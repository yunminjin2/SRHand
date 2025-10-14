import os
from os.path import join, exists
import glob
import json
import numpy as np
import cv2
import trimesh
import torch
import smplx
import nvdiffrast.torch as dr

import zipfile
import pillow_avif
from io import BytesIO
import pandas as pd
import itertools
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image

from models.utils import *
from models.get_rays import get_ray_directions, get_rays

import torch.nn.functional as F

mano_layer = {'right': smplx.create('./', 'mano', use_pca=False, is_rhand=True), 'left': smplx.create('./', 'mano', use_pca=False, is_rhand=False)}


# fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
if torch.sum(torch.abs(mano_layer['left'].shapedirs[:,0,:] - mano_layer['right'].shapedirs[:,0,:])) < 1:
    print('Fix shapedirs bug of MANO')
    mano_layer['left'].shapedirs[:,0,:] *= -1

def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area

def img_contrast_bright(img, a, b, g):
    h, w, c = img.shape
    blank = np.zeros([h, w, c], img.dtype)
    dst = cv2.addWeighted(img, a, blank, b, g)
    return dst

def img_adjust(img):
    stone_gama = np.power(img.astype(np.float32), 0.75)  # 图像较暗，若采用幂率变换，γ<1，拉伸低灰度级,交互式选择
    temp = stone_gama - np.min(stone_gama)
    stone_gama = temp / np.max(temp)

    img_cmy = 1 - cv2.cvtColor(stone_gama, cv2.COLOR_BGR2RGB)
    c, m, y = cv2.split(img_cmy)
    # print(m.shape)
    m_gama = np.power(m.astype(np.float32), 0.88)  # 深红色较多，压缩一下
    temp_m = m_gama - np.min(m_gama)
    m_gama = (temp_m / (np.max(temp_m)))
    out_stone = 1 - cv2.merge((c, m_gama, y))

    adjusted = cv2.addWeighted(out_stone * 255, 1.3, out_stone * 255, 0, 5) / 255
    return cv2.cvtColor(adjusted, cv2.COLOR_RGB2BGR)


def sample_frames(frame_list, camera_list, data_path, extension='.avif'):
    common_frames = []
    
    for frame in frame_list:
        frame = f'{frame:06d}'
        
        frame_in_all_cameras = all(os.path.exists(os.path.join(data_path, 'cam' + camera, frame +extension)) for camera in camera_list)
        if frame_in_all_cameras:
            common_frames.append(frame)

    return sorted(common_frames)



def get_interhand_cam_seqdatabyframe(data_path, res=(336, 512), data_name='0003_fake_gun', lr_size=16, 
                                 capture_name='Capture9', drop_cam=[], split='train', cam_id=None, test_num=30,
                                 return_ray=False, num_frame=20, adjust=True):
    mano_layer['right'] = mano_layer['right'].cpu()
    mano_layer['left'] = mano_layer['left'].cpu()
    capture_idx = capture_name.replace('Capture', '')

    with open(join(data_path, 'annotations/%s' % split, 'InterHand2.6M_%s_camera.json' % split)) as f:
        cam_params = json.load(f)
    with open(join(data_path, 'annotations/%s' % split, 'InterHand2.6M_%s_MANO_NeuralAnnot.json' % split)) as f:
        mano_params = json.load(f)
    cam_param = cam_params[capture_idx]
    if data_name == 'all':
        data_names = sorted(os.listdir(join(data_path, 'images/%s' % split, capture_name)))[:5]
    else:
        data_names = [data_name]

    imgs_t, lr_imgs_t, grayimgs_t, masks_t, w2cs_t, projs_t, mano_out_t , cam_exs_t, cam_ins_t = [], [], [], [], [], [], [], [], []
    for data_name in data_names:
        if cam_id is None or len(cam_id) == 0:
            camera_names = [i for i in sorted(os.listdir(join(data_path, 'images/%s' % split, capture_name, data_name))) if i not in drop_cam and '400' in i][:test_num]
        else:
            camera_names = cam_id if isinstance(cam_id, list) else [cam_id]
        num = len(camera_names)
        img_files = os.listdir(join(data_path, 'images/%s' % split, capture_name, data_name, camera_names[0]))
        img_names = sorted([file for file in img_files if file.endswith(".jpg")])
        
        img_names = img_names[::max(len(img_names) // num_frame, 1)][:num_frame]
        print('image views num: %d, frames num: %d' % (num, len(img_names)))
        print(data_name, img_names, camera_names)
        for img_name in img_names:
            mano_param = mano_params[capture_idx][str(int(img_name[5:-4]))]
            vertices = []
            faces = []
            mano_out = []
            for hand_type in ['left', 'right']:
                if mano_param[hand_type] is not None:
                    mano_pose = torch.FloatTensor(mano_param[hand_type]['pose']).view(-1, 3)
                    
                    root_pose = mano_pose[0].view(1, 3)
                    hand_pose = mano_pose[1:, :].view(1, -1)
                    shape = torch.FloatTensor(mano_param[hand_type]['shape']).view(1, -1)
                    trans = torch.FloatTensor(mano_param[hand_type]['trans']).view(1, 3)
                    output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape,
                                                   transl=trans)

                    vertices.append(output.vertices)
                    mano_out.append({'type': hand_type, 'pose': mano_pose, 'shape': shape, 'trans': trans, 'vertices': vertices})
                    if len(faces) == 0:
                        faces.append(mano_layer[hand_type].faces)
                    else:
                        faces.append(mano_layer[hand_type].faces + output.vertices.shape[1])

            vertices = torch.cat(vertices, 1).cuda()
            faces = np.concatenate(faces, 0)
            # mesh = trimesh.Trimesh(vertices=vertices[0].detach().cpu().numpy(), faces=faces)
            # mesh.export('test.obj')
            faces = torch.from_numpy(faces.astype(np.int32)).int().cuda()

            w2cs = []
            projs = []
            imgs = []
            grayimgs = []
            cam_exs = []
            cam_ins = []

            for i, cam_name in enumerate(camera_names):
                cam_idx = cam_name.replace('cam', '')
                
                t, R = np.array(cam_param['campos'][str(cam_idx)], dtype=np.float32).reshape(3), np.array(
                    cam_param['camrot'][str(cam_idx)], dtype=np.float32).reshape(3, 3)
                scale_mats = np.eye(4)
                scale_mats[:3, :3] = R
                cam_t = -np.dot(R, t.reshape(3, 1)).reshape(3) / 1000
                scale_mats[:3, 3] = cam_t

                cam_ex = torch.cat([torch.from_numpy(R), torch.from_numpy(cam_t).reshape(3, 1)], dim=-1).float()

                focal = np.array(cam_param['focal'][cam_idx], dtype=np.float32).reshape(2)
                princpt = np.array(cam_param['princpt'][cam_idx], dtype=np.float32).reshape(2)
                cameraIn = np.array([[focal[0], 0, princpt[0]],
                                     [0, focal[1], princpt[1]],
                                     [0, 0, 1]])

                img = cv2.imread(
                    join(data_path, 'images/%s' % split, capture_name, data_name, 'cam' + cam_idx, img_name))
                img = img_adjust(img) * 255 if adjust else img
                
                
                hand = vertices[0].cpu().numpy() @ R.T + cam_t
                
                hand2d = hand @ cameraIn.T
                hand2d = hand2d[:, :2] / hand2d[:, 2:]

                [img], _, cameraIn = cut_img([img], np.array(hand2d[None,]), camera=cameraIn, radio=0.8, img_size=res[0])
                cam_in = torch.from_numpy(cameraIn).float()
            
            
                P = cameraIn @ scale_mats[:3]
                proj, w2c = load_K_Rt_from_P(P[:3])

                proj[0, 0] = proj[0, 0] / (res[0] / 2.)
                proj[0, 2] = proj[0, 2] / (res[0] / 2.) - 1.
                proj[1, 1] = proj[1, 1] / (res[1] / 2.)
                proj[1, 2] = proj[1, 2] / (res[1] / 2.) - 1.
                proj[2, 2] = 0.
                proj[2, 3] = -0.1
                proj[3, 2] = 1.
                proj[3, 3] = 0.
                projs.append(proj.astype(np.float32))
                w2cs.append(w2c.astype(np.float32))
                cam_exs.append(cam_ex)
                cam_ins.append(cam_in)

                # img = img_contrast_bright(img, 1.2, -0.2, 30)

                grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                img = cv2.resize(img, res)
                grayimg = cv2.resize(grayimg, res)

                img = torch.from_numpy((img / 255.)).float()
                grayimg = torch.from_numpy((grayimg / 255.)).float()

                imgs.append(img)
                grayimgs.append(grayimg)
                                
            w2cs = torch.from_numpy(np.stack(w2cs)).permute(0, 2, 1).cuda()
            projs = torch.from_numpy(np.stack(projs)).permute(0, 2, 1).cuda()
            cam_exs = torch.stack(cam_exs).cuda()
            cam_ins = torch.stack(cam_ins).cuda()

            glctx = dr.RasterizeCudaContext()
            vertsw = torch.cat([vertices, torch.ones_like(vertices[:, :, 0:1])], axis=2).expand(num, -1, -1)
            rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2cs)
            proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, projs)
            

            rast_out, _ = dr.rasterize(glctx, proj_verts, faces, resolution=(res[1], res[0]))
            
            feat = torch.ones_like(vertsw[:, :, :1])
            feat, _ = dr.interpolate(feat, rast_out, faces)
            masks = feat[:, :, :, :1].contiguous().squeeze(-1).cpu()
            # masks = dr.antialias(masks, rast_out, proj_verts, faces).squeeze(-1)
            # mask_list = []
            # for b in range(masks.shape[0]):
            #     img = imgs[b].cpu().numpy()
            #     mask = masks[b].unsqueeze(-1).numpy()
                
            #     mask_bool = mask[..., 0] > 0.5
            #     sel_img = img[mask_bool].mean(axis=-1)

            #     sel_img = np.bitwise_and(sel_img>(10/255), sel_img<(200/ 255))
            #     mask_bool[mask_bool] = sel_img.astype('int32')
            #     mask = mask * mask_bool[..., None]

                
            #     contours, _ = cv2.findContours(mask[..., 0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #     contours = list(contours)
            #     contours.sort(key=cnt_area, reverse=True)
            #     poly = contours[0].transpose(1, 0, 2).astype(np.int32)
            #     poly_mask = np.zeros_like(img)
            #     poly_mask = cv2.fillPoly(poly_mask, poly, (1,1,1))
            #     mask = mask * poly_mask

            #     mask_list.append(torch.from_numpy(mask[..., 0]).float())
            
            
            # masks = torch.stack(mask_list, 0).cuda()
            
            imgs = torch.stack(imgs, dim=0)
            grayimgs = torch.stack(grayimgs, dim=0)
            
            
            imgs[masks == 0] = 0
            grayimgs[masks == 0] = 0
            
            lr_img = F.interpolate(imgs.permute(0, 3, 1, 2), lr_size).permute(0, 2, 3, 1)

            imgs_t.append(imgs)
            lr_imgs_t.append(lr_img)
            grayimgs_t.append(grayimgs)
            masks_t.append(masks)
            w2cs_t.append(w2cs)
            projs_t.append(projs)
            cam_exs_t.append(cam_exs)
            cam_ins_t.append(cam_ins)
            mano_out_t.append(mano_out)

    imgs_t = torch.stack(imgs_t, dim=0)
    lr_imgs_t = torch.stack(lr_imgs_t, dim=0)
    grayimgs_t = torch.stack(grayimgs_t, dim=0)
    masks_t = torch.stack(masks_t, dim=0).cpu()
    w2cs_t = torch.stack(w2cs_t, dim=0).cpu()
    projs_t = torch.stack(projs_t, dim=0).cpu()
    cam_exs_t = torch.stack(cam_exs_t, dim=0).cpu()
    cam_ins_t = torch.stack(cam_ins_t, dim=0).cpu()

    hand_types = []
    pose_t = []
    shape_t = []
    trans_t = []
    vertices_t = []
    for mano_out in mano_out_t:
        if len(mano_out) == 2:
            hand_types = ['left', 'right']
            pose_t.append(torch.cat([mano_out[0]['pose'], mano_out[1]['pose']], 0).unsqueeze(0))
            shape_t.append(torch.cat([mano_out[0]['shape'], mano_out[1]['shape']], 1))
            trans_t.append(torch.cat([mano_out[0]['trans'], mano_out[1]['trans']], 1))
        else:
            hand_types = [mano_out[0]['type']]
            pose_t.append(mano_out[0]['pose'].unsqueeze(0))
            shape_t.append(mano_out[0]['shape'])
            trans_t.append(mano_out[0]['trans'])
            vertices_t.append(mano_out[0]['vertices'][0])

    pose_t = torch.cat(pose_t, 0)
    shape_t = torch.cat(shape_t, 0)
    trans_t = torch.cat(trans_t, 0)
    vertices_t = torch.cat(vertices_t, 0)

    if return_ray:
        ray_directions = []

        c2ws = torch.inverse(w2cs)
        for i, cam_name in enumerate(camera_names):
            cam_idx = cam_name.replace('cam', '')
            cam_ray_direction = get_ray_directions(res[1], res[0], cam_param['focal'][cam_idx][0],
                                                   cam_param['focal'][cam_idx][1],
                                                   cam_param['princpt'][cam_idx][0],
                                                   cam_param['princpt'][cam_idx][1], ).cuda()

            tmp_ray_direction, _ = get_rays(cam_ray_direction, c2ws[i])

            ray_direction = tmp_ray_direction.reshape(res[1], res[0], 3).cpu()
            ray_directions.append(ray_direction)
        ray_directions = torch.stack(ray_directions)
        return imgs_t, lr_imgs_t, grayimgs_t, masks_t, w2cs_t, projs_t, cam_exs_t, cam_ins_t, pose_t, shape_t, trans_t, hand_types, vertices_t, ray_directions

    return imgs_t, lr_imgs_t, grayimgs_t, masks_t, w2cs_t, projs_t, cam_exs_t, cam_ins_t, pose_t, shape_t, trans_t, vertices_t, hand_types



def get_interhand_cam_test_seqdatabyframe(data_path, res=(334, 512), lr_size=16, data_name='0003_fake_gun', capture_name='Capture9', drop_cam=[], split='train', cam_id=None, test_num=30, num_frame=-1, return_ray=False, adjust=True):
    mano_layer['right'] = mano_layer['right'].cpu()
    mano_layer['left'] = mano_layer['left'].cpu()
    capture_idx = capture_name.replace('Capture', '')

    with open(join(data_path, 'annotations/%s' % split, 'InterHand2.6M_%s_camera.json' % split)) as f:
        cam_params = json.load(f)
    with open(join(data_path, 'annotations/%s' % split, 'InterHand2.6M_%s_MANO_NeuralAnnot.json' % split)) as f:
        mano_params = json.load(f)
    cam_param = cam_params[capture_idx]
    if data_name == 'all':
        data_names = sorted(os.listdir(join(data_path, 'images/%s' % split, capture_name)))[:20]
    else:
        data_names = [data_name]

    imgs_t, lr_imgs_t, grayimgs_t, masks_t, w2cs_t, projs_t, mano_out_t, cam_exs_t, cam_ins_t, vertices_t = [], [], [], [], [], [], [], [], [], []

    for data_name in data_names:
        if cam_id is None:
            camera_names = [i for i in sorted(os.listdir(join(data_path, 'images/%s' % split, capture_name, data_name))) if i not in drop_cam and '400' in i][:test_num]
        else:
            
            camera_names = cam_id if isinstance(cam_id, list) else [cam_id]
        num = len(camera_names)
        img_files = os.listdir(join(data_path, 'images/%s' % split, capture_name, data_name, camera_names[0]))
        img_names = sorted([file for file in img_files if file.endswith(".jpg")])
        if num_frame != -1:
            img_names = img_names[:num_frame]

        print('image views num: %d, frames num: %d' % (num, len(img_names)))
        # print(data_name, img_names, camera_names)
        for img_name in img_names:
            mano_param = mano_params[capture_idx][str(int(img_name[5:-4]))]
            vertices = []
            faces = []
            mano_out = []
            for hand_type in ['left', 'right']:
                if mano_param[hand_type] is not None:
                    mano_pose = torch.FloatTensor(mano_param[hand_type]['pose']).view(-1, 3)
                    root_pose = mano_pose[0].view(1, 3)
                    hand_pose = mano_pose[1:, :].view(1, -1)
                    shape = torch.FloatTensor(mano_param[hand_type]['shape']).view(1, -1)
                    trans = torch.FloatTensor(mano_param[hand_type]['trans']).view(1, 3)
                    output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)

                    vertices.append(output.vertices)
                    mano_out.append({'type': hand_type, 'pose': mano_pose, 'shape': shape, 'trans': trans, 'vertices': vertices})
                    if len(faces) == 0:
                        faces.append(mano_layer[hand_type].faces)
                    else:
                        faces.append(mano_layer[hand_type].faces + output.vertices.shape[1])

            vertices = torch.cat(vertices, 1).cuda()
            faces = np.concatenate(faces, 0)
            faces = torch.from_numpy(faces.astype(np.int32)).int().cuda()

            
            w2cs = []
            projs = []
            imgs = []
            grayimgs = []
            cam_exs = []
            cam_ins = []
            for i, cam_name in enumerate(camera_names):
                cam_idx = cam_name.replace('cam', '')
                t, R = np.array(cam_param['campos'][str(cam_idx)], dtype=np.float32).reshape(3), np.array(
                    cam_param['camrot'][str(cam_idx)], dtype=np.float32).reshape(3, 3)
                scale_mats = np.eye(4)
                scale_mats[:3, :3] = R
                cam_t = -np.dot(R, t.reshape(3, 1)).reshape(3) / 1000
                scale_mats[:3, 3] = cam_t

                cam_ex = torch.cat([torch.from_numpy(R), torch.from_numpy(cam_t).reshape(3, 1)], dim=-1).float()
                
                focal = np.array(cam_param['focal'][cam_idx], dtype=np.float32).reshape(2)
                princpt = np.array(cam_param['princpt'][cam_idx], dtype=np.float32).reshape(2)
                cameraIn = np.array([[focal[0], 0, princpt[0]],
                                     [0, focal[1], princpt[1]],
                                     [0, 0, 1]])

                img = cv2.imread(
                    join(data_path, 'images/%s' % split, capture_name, data_name, 'cam' + cam_idx, img_name))
                img = img_adjust(img) * 255 if adjust else img
                
                                
                hand = vertices[0].cpu().numpy() @ R.T + cam_t
                hand2d = hand @ cameraIn.T
                hand2d = hand2d[:, :2] / hand2d[:, 2:]

                [img], _, cameraIn = cut_img([img], np.array(hand2d[None,]), camera=cameraIn, radio=0.8, img_size=res[0])
                cam_in = torch.from_numpy(cameraIn).float()
                
            
                P = cameraIn @ scale_mats[:3]
                proj, w2c = load_K_Rt_from_P(P[:3])

                proj[0, 0] = proj[0, 0] / (res[0] / 2.)
                proj[0, 2] = proj[0, 2] / (res[0] / 2.) - 1.
                proj[1, 1] = proj[1, 1] / (res[1] / 2.)
                proj[1, 2] = proj[1, 2] / (res[1] / 2.) - 1.
                proj[2, 2] = 0.
                proj[2, 3] = -0.1
                proj[3, 2] = 1.
                proj[3, 3] = 0.
                projs.append(proj.astype(np.float32))
                w2cs.append(w2c.astype(np.float32))

                cam_exs.append(cam_ex)
                cam_ins.append(cam_in)

                # img = img_contrast_bright(img, 1.2, -0.2, 30)

                grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                img = cv2.resize(img, res)
                grayimg = cv2.resize(grayimg, res)

                img = torch.from_numpy((img / 255.)).float().cuda()
                grayimg = torch.from_numpy((grayimg / 255.)).float().cuda()

                imgs.append(img)
                grayimgs.append(grayimg)
                

            w2cs = torch.from_numpy(np.stack(w2cs)).permute(0, 2, 1).cuda()
            projs = torch.from_numpy(np.stack(projs)).permute(0, 2, 1).cuda()
            cam_exs = torch.stack(cam_exs).cuda()
            cam_ins = torch.stack(cam_ins).cuda()

            glctx = dr.RasterizeCudaContext()
            vertsw = torch.cat([vertices, torch.ones_like(vertices[:, :, 0:1])], axis=2).expand(num, -1, -1)
            rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2cs)
            proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, projs)
            

            rast_out, _ = dr.rasterize(glctx, proj_verts, faces, resolution=(res[1], res[0]))
            
            feat = torch.ones_like(vertsw[:, :, :1])
            feat, _ = dr.interpolate(feat, rast_out, faces)
            masks = feat[:, :, :, :1].contiguous().squeeze(-1).cpu()
            # masks = feat[:, :, :, :1].contiguous().cpu()
           
            imgs = torch.stack(imgs, dim=0)
            grayimgs = torch.stack(grayimgs, dim=0)
            
            
            # mask_list = []
            # for b in range(masks.shape[0]):
            #     img = imgs[b].cpu().numpy()
            #     mask = masks[b].numpy()
                
            #     mask_bool = mask[..., 0] > 0.5
            #     sel_img = img[mask_bool].mean(axis=-1)

            #     sel_img = np.bitwise_and(sel_img>(10/255), sel_img<(200/ 255))
            #     mask_bool[mask_bool] = sel_img.astype('int32')
            #     mask = mask * mask_bool[..., None]

                
            #     contours, _ = cv2.findContours(mask[..., 0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #     contours = list(contours)
            #     contours.sort(key=cnt_area, reverse=True)
            #     poly = contours[0].transpose(1, 0, 2).astype(np.int32)
            #     poly_mask = np.zeros_like(img)
            #     poly_mask = cv2.fillPoly(poly_mask, poly, (1,1,1))
            #     mask = mask * poly_mask

            #     mask_list.append(torch.from_numpy(mask[..., 0]).float())
            # masks = torch.stack(mask_list, 0).cuda()
            
            
            lr_img = F.interpolate(imgs.permute(0, 3, 1, 2), lr_size).permute(0, 2, 3, 1)
            
            imgs_t.append(imgs)
            lr_imgs_t.append(lr_img)
            grayimgs_t.append(grayimgs)
            masks_t.append(masks)
            w2cs_t.append(w2cs)
            projs_t.append(projs)
            cam_exs_t.append(cam_exs)
            cam_ins_t.append(cam_ins)
            mano_out_t.append(mano_out)

    imgs_t = torch.stack(imgs_t, dim=0)
    lr_imgs_t = torch.stack(lr_imgs_t, dim=0)
    masks_t = torch.stack(masks_t, dim=0)
    grayimgs_t = torch.stack(grayimgs_t, dim=0)
    w2cs_t = torch.stack(w2cs_t, dim=0)
    projs_t = torch.stack(projs_t, dim=0)
    cam_exs_t = torch.stack(cam_exs_t, dim=0).cpu()
    cam_ins_t = torch.stack(cam_ins_t, dim=0).cpu()

    hand_types = []
    pose_t = []
    shape_t = []
    trans_t = []
    vertices_t = []
    for mano_out in mano_out_t:
        if len(mano_out) == 2:
            hand_types = ['left', 'right']
            pose_t.append(torch.cat([mano_out[0]['pose'], mano_out[1]['pose']], 0).unsqueeze(0))
            shape_t.append(torch.cat([mano_out[0]['shape'], mano_out[1]['shape']], 1))
            trans_t.append(torch.cat([mano_out[0]['trans'], mano_out[1]['trans']], 1))
        else:
            hand_types = [mano_out[0]['type']]
            pose_t.append(mano_out[0]['pose'].unsqueeze(0))
            shape_t.append(mano_out[0]['shape'])
            trans_t.append(mano_out[0]['trans'])
            vertices_t.append(mano_out[0]['vertices'][0])

    pose_t = torch.cat(pose_t, 0)
    shape_t = torch.cat(shape_t, 0)
    trans_t = torch.cat(trans_t, 0)
    vertices_t = torch.cat(vertices_t, 0)

    if return_ray:
        ray_directions = []

        c2ws = torch.inverse(w2cs)
        for i, cam_name in enumerate(camera_names):
            cam_idx = cam_name.replace('cam', '')
            cam_ray_direction = get_ray_directions(res[1], res[0], cam_param['focal'][cam_idx][0],
                                                   cam_param['focal'][cam_idx][1],
                                                   cam_param['princpt'][cam_idx][0],
                                                   cam_param['princpt'][cam_idx][1], ).cuda()

            tmp_ray_direction, _ = get_rays(cam_ray_direction, c2ws[i])

            ray_direction = tmp_ray_direction.reshape(res[1], res[0], 3)
            ray_directions.append(ray_direction)
        ray_directions = torch.stack(ray_directions)
        return imgs_t, lr_imgs_t, grayimgs_t, masks_t, w2cs_t, projs_t, cam_exs_t, cam_ins_t, pose_t, shape_t, trans_t, hand_types, ray_directions, vertices_t, img_names

    return imgs_t, grayimgs_t, w2cs_t, projs_t, cam_exs_t, cam_ins_t, pose_t, shape_t, trans_t, hand_types, img_names
