import torch
import smplx
import numpy as np
import os
from models.XHand import XHand
import json
import cv2
import nvdiffrast.torch as dr
from models.utils import load_K_Rt_from_P
from models.get_rays import get_ray_directions, get_rays
from torchvision.utils import save_image
drop_cam = ["cam400006,cam400008,cam400015,cam400035,cam400049,cam400323,cam410015,cam410018,cam410028,cam410062,cam410063,cam410068,cam410210,cam410211,cam410216,cam410218,cam410236,cam410238"]

mano_layer = {'right': smplx.create('./', 'mano', use_pca=False, is_rhand=True),
              'left': smplx.create('./', 'mano', use_pca=False, is_rhand=False)}
mano_layer['right'] = mano_layer['right'].cpu()
mano_layer['left'] = mano_layer['left'].cpu()

data_path = '/workspace/datasets/InterHand/'

def cut_img(img_list, label2d_list, camera=None, radio=0.7, img_size=256):
    Min = []
    Max = []
    for label2d in label2d_list:
        Min.append(np.min(label2d, axis=0))
        Max.append(np.max(label2d, axis=0))
    Min = np.min(np.array(Min), axis=0)
    Max = np.max(np.array(Max), axis=0)

    mid = (Min + Max) / 2
    L = np.max(Max - Min) / 2 / radio
    M = img_size / 2 / L * np.array([[1, 0, L - mid[0]],
                                     [0, 1, L - mid[1]]])

    img_list_out = []
    for img in img_list:
        img_list_out.append(cv2.warpAffine(img, M, dsize=(img_size, img_size)))

    label2d_list_out = []
    for label2d in label2d_list:
        x = np.concatenate([label2d, np.ones_like(label2d[:, :1])], axis=-1)
        x = x @ M.T
        label2d_list_out.append(x)

    if camera is not None:
        camera[0, 0] = camera[0, 0] * M[0, 0]
        camera[1, 1] = camera[1, 1] * M[1, 1]
        camera[0, 2] = camera[0, 2] * M[0, 0] + M[0, 2]
        camera[1, 2] = camera[1, 2] * M[1, 1] + M[1, 2]

    return img_list_out, label2d_list_out, camera

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

def cut_image(res=(256, 256), data_name='ROM03_RT_No_Occlusion', capture_name='Capture0', split='test', return_ray=True):
    capture_idx = capture_name.replace('Capture', '')

    with open(os.path.join(data_path, 'annotations/%s' % split, 'InterHand2.6M_%s_camera.json' % split)) as f:
        cam_params = json.load(f)
    with open(os.path.join(data_path, 'annotations/%s' % split, 'InterHand2.6M_%s_MANO_NeuralAnnot.json' % split)) as f:
        mano_params = json.load(f)
    
    
    camera_names = [i for i in sorted(os.listdir(os.path.join(data_path, 'images/%s' % split, capture_name, data_name))) if i not in drop_cam and '400' in i][:3]
    
    num = len(camera_names)
    
    img_files = os.listdir(os.path.join(data_path, 'images/%s' % split, capture_name, data_name, camera_names[0]))
    img_names = sorted([file for file in img_files if file.endswith(".jpg")])
    img_names = img_names[::max(len(img_names) // 10, 1)][:10]
    cam_param = cam_params[capture_idx]
    
    imgs_t, grayimgs_t, masks_t, w2cs_t, projs_t, mano_out_t = [], [], [], [], [], []
    
    for img_name in img_names:

        mano_param = mano_params[capture_idx][str(int(img_name[5:-4]))]
        
        hand_type = 'right' 
        if mano_param[hand_type] is None:
            hand_type = 'left'
            if mano_param[hand_type] is None:
                continue
        
        mano_pose = torch.FloatTensor(mano_param[hand_type]['pose']).view(-1, 3)
        root_pose = mano_pose[0].view(1, 3)
        hand_pose = mano_pose[1:, :].view(1, -1)
        shape = torch.FloatTensor(mano_param[hand_type]['shape']).view(1, -1)
        trans = torch.FloatTensor(mano_param[hand_type]['trans']).view(1, 3)
        mano_out = [{'type': hand_type, 'pose': mano_pose, 'shape': shape, 'trans': trans}]
        
        output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)

        vertices = output.vertices.cuda()
        faces = mano_layer[hand_type].faces
        faces = torch.from_numpy(faces.astype(np.int32)).int().cuda()
        handV = output.vertices[0]
        handJ = output.joints[0]
    
        imgs, grayimgs, projs, w2cs = [], [], [], []

        for i, cam_name in enumerate(camera_names):
            cam_idx = cam_name.replace('cam', '')
            t = np.array(cam_param['campos'][str(cam_idx)], dtype=np.float32).reshape(3)
            R =np.array(cam_param['camrot'][str(cam_idx)], dtype=np.float32).reshape(3, 3)
            scale_mats = np.eye(4)
            scale_mats[:3, :3] = R
            cam_t = -np.dot(R, t.reshape(3, 1)).reshape(3) / 1000
            scale_mats[:3, 3] = cam_t

            focal = np.array(cam_param['focal'][cam_idx], dtype=np.float32).reshape(2)
            princpt = np.array(cam_param['princpt'][cam_idx], dtype=np.float32).reshape(2)
            cameraIn = np.array([[focal[0], 0, princpt[0]],
                                [0, focal[1], princpt[1]],
                                [0, 0, 1]]
                        )


            img = cv2.imread(os.path.join(data_path, 'images/%s' % split, capture_name, data_name, 'cam' + cam_idx, img_name))
            img = img_adjust(img) * 255
        
            hand = handV @ R.T + cam_t
            hand2d = hand @ cameraIn.T
            hand2d = hand2d[:, :2] / hand2d[:, 2:]

            [img], _, cameraIn = cut_img([img], np.array(hand2d[None,]), camera=cameraIn, radio=0.8, img_size=res[0])
            
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
            
            handV2d = handV @ cameraIn.T
            handV2d = handV2d[:, :2] / handV2d[:, 2:]
            handJ2d = handJ @ cameraIn.T
            handJ2d = handJ2d[:, :2] / handJ2d[:, 2:]

            grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            img = torch.from_numpy((img / 255.)).float()
            grayimg = torch.from_numpy((grayimg / 255.)).float()

            projs.append(proj.astype(np.float32))
            w2cs.append(w2c.astype(np.float32))
            
            imgs.append(img)
            grayimgs.append(grayimg)
            
        w2cs = torch.from_numpy(np.stack(w2cs)).permute(0, 2, 1).cuda()
        projs = torch.from_numpy(np.stack(projs)).permute(0, 2, 1).cuda()
                
    
        glctx = dr.RasterizeCudaContext()
        vertsw = torch.cat([vertices, torch.ones_like(vertices[:, :, 0:1])], axis=2).expand(num, -1, -1)
        rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2cs)
        proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, projs)

        rast_out, _ = dr.rasterize(glctx, proj_verts, faces, resolution=(res[1], res[0]))
        feat = torch.ones_like(vertsw[:, :, :1])
        feat, _ = dr.interpolate(feat, rast_out, faces)
        masks = feat[:, :, :, :1].contiguous().squeeze(-1)
        # masks = dr.antialias(masks, rast_out, proj_verts, faces).squeeze(-1)

        imgs = torch.stack(imgs, dim=0)
        grayimgs = torch.stack(grayimgs, dim=0)
        imgs[masks == 0] = 0
        grayimgs[masks == 0] = 0

        imgs_t.append(imgs)
        grayimgs_t.append(grayimgs)
        masks_t.append(masks)
        w2cs_t.append(w2cs)
        projs_t.append(projs)
        mano_out_t.append(mano_out)
    
        
        
        
    imgs_t = torch.stack(imgs_t, dim=0)
    grayimgs_t = torch.stack(grayimgs_t, dim=0)
    masks_t = torch.stack(masks_t, dim=0).cpu()
    w2cs_t = torch.stack(w2cs_t, dim=0).cpu()
    projs_t = torch.stack(projs_t, dim=0).cpu()

    hand_types = []
    pose_t = []
    shape_t = []
    trans_t = []
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

    pose_t = torch.cat(pose_t, 0)
    shape_t = torch.cat(shape_t, 0)
    trans_t = torch.cat(trans_t, 0)

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
        return imgs_t, grayimgs_t, masks_t, w2cs_t, projs_t, pose_t, shape_t, trans_t, hand_types, ray_directions

    return imgs_t, grayimgs_t, masks_t, w2cs_t, projs_t, pose_t, shape_t, trans_t, hand_types

        
        
        
        
        
if __name__=='__main__':
    xhand_pth = torch.load('./interhand_out/Capture0_ROM03_RT_No_Occlusion_test/xhand/model.pth')
    xhand = XHand(xhand_pth.verts, xhand_pth.faces, xhand_pth.delta_net.x_mean,
           xhand_pth.color_net.x_mean, xhand_pth.lbs_net.x_mean,
           xhand_pth.template_v, xhand_pth.sh_coeffs, latent_num=20,
           hand_type=xhand_pth.hand_type, render_nettype='mlp', use_pe=True,
           use_x_pos=True, use_ray=True, use_emb=False, wo_latent=False, mlp_use_pose=True, use_rotpose=False
        ).cuda()

    glctx = dr.RasterizeCudaContext()
    resolution = [256, 256]


    imgs, grayimgs, masks, w2cs, projs, poses, shapes, transs, hand_types, rays = cut_image()
    scales = torch.ones(transs.shape[0], 1)
    
    j = 0
    k = 0
    pose = poses[j:j + 1].reshape(1, -1).cuda()
    shape = shapes[j:j + 1].cuda()
    trans = transs[j:j + 1].cuda()
    scale = scales[j:j + 1].cuda()

    k = 0
    batch = 1
    n = min(50, k + batch) - k
    w2c = w2cs[j][k:k + n].cuda()
    proj = projs[j][k:k + n].cuda()
    img = imgs[j][k:k + n].cuda()
    mask = masks[j][k:k + n].cuda()
    ray = rays[k:k + n].cuda()
    # valid_mask = valid_masks[perm[k:k+batch]]
    sh_coeff = xhand.sh_coeffs[k:k + n]
    
    data_input = pose, trans, scale, w2c, proj, None, ray, sh_coeff
    render_imgs, mesh_imgs, pred_imgs, vertices_new, pred_mask, pred_albedo = (
        xhand(data_input, glctx, resolution, is_train=False))
    
    
    import pdb; pdb.set_trace()