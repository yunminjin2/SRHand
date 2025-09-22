import os
from os.path import join, exists
import sys
from typing import Tuple, Union

import glob
import json
import numpy as np
import cv2
import trimesh
import torch
import smplx
import nvdiffrast.torch as dr
from models.utils import load_K_Rt_from_P, cut_img
from models.get_rays import get_ray_directions, get_rays
from projection import *

from torch.utils.data import Dataset
from copy import deepcopy
from sklearn.utils import shuffle
import torch.nn.functional as F

from models.utils import get_normals, draw_joint
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops.packed_to_padded import packed_to_padded
from pytorch3d.renderer.mesh.rasterizer import Fragments as MeshFragments
from pytorch3d.structures import Meshes

import matplotlib.pyplot as plt

mano_layer = {'right': smplx.create('./', 'mano', use_pca=False, is_rhand=True), 'left': smplx.create('./', 'mano', use_pca=False, is_rhand=False)}

# fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
if torch.sum(torch.abs(mano_layer['left'].shapedirs[:,0,:] - mano_layer['right'].shapedirs[:,0,:])) < 1:
    print('Fix shapedirs bug of MANO')
    mano_layer['left'].shapedirs[:,0,:] *= -1

def _rand_barycentric_coords(
    size1, size2, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper function to generate random barycentric coordinates which are uniformly
    distributed over a triangle.

    Args:
        size1, size2: The number of coordinates generated will be size1*size2.
                      Output tensors will each be of shape (size1, size2).
        dtype: Datatype to generate.
        device: A torch.device object on which the outputs will be allocated.

    Returns:
        w0, w1, w2: Tensors of shape (size1, size2) giving random barycentric
            coordinates
    """
    uv = torch.rand(2, size1, size2, dtype=dtype, device=device)
    u, v = uv[0], uv[1]
    u_sqrt = u.sqrt()
    w0 = 1.0 - u_sqrt
    w1 = u_sqrt * (1.0 - v)
    w2 = u_sqrt * v
    # pyre-fixme[7]: Expected `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]` but
    #  got `Tuple[float, typing.Any, typing.Any]`.
    return w0, w1, w2


def sample_points_from_meshes(
    meshes,
    num_samples: int = 10000,
    return_normals: bool = False,
    return_textures: bool = False,
    return_sample_info: bool = False,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    Convert a batch of meshes to a batch of pointclouds by uniformly sampling
    points on the surface of the mesh with probability proportional to the
    face area.

    Args:
        meshes: A Meshes object with a batch of N meshes.
        num_samples: Integer giving the number of point samples per mesh.
        return_normals: If True, return normals for the sampled points.
        return_textures: If True, return textures for the sampled points.

    Returns:
        3-element tuple containing

        - **samples**: FloatTensor of shape (N, num_samples, 3) giving the
          coordinates of sampled points for each mesh in the batch. For empty
          meshes the corresponding row in the samples array will be filled with 0.
        - **normals**: FloatTensor of shape (N, num_samples, 3) giving a normal vector
          to each sampled point. Only returned if return_normals is True.
          For empty meshes the corresponding row in the normals array will
          be filled with 0.
        - **textures**: FloatTensor of shape (N, num_samples, C) giving a C-dimensional
          texture vector to each sampled point. Only returned if return_textures is True.
          For empty meshes the corresponding row in the textures array will
          be filled with 0.

        Note that in a future releases, we will replace the 3-element tuple output
        with a `Pointclouds` datastructure, as follows

        .. code-block:: python

            Pointclouds(samples, normals=normals, features=textures)
    """
    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")

    if return_textures and meshes.textures is None:
        raise ValueError("Meshes do not contain textures.")

    faces = meshes.faces_packed()
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
    num_meshes = len(meshes)
    num_valid_meshes = torch.sum(meshes.valid)  # Non empty meshes.

    # Initialize samples tensor with fill value 0 for empty meshes.
    samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)

    # Only compute samples for non empty meshes
    with torch.no_grad():
        areas, _ = mesh_face_areas_normals(verts, faces)  # Face areas can be zero.
        max_faces = meshes.num_faces_per_mesh().max().item()
        areas_padded = packed_to_padded(
            areas, mesh_to_face[meshes.valid], max_faces
        )  # (N, F)

        # TODO (gkioxari) Confirm multinomial bug is not present with real data.
        sample_face_idxs = areas_padded.multinomial(
            num_samples, replacement=True
        )  # (N, num_samples)
        sample_face_idxs += mesh_to_face[meshes.valid].view(num_valid_meshes, 1)
    
    # Get the vertex coordinates of the sampled faces.
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Randomly generate barycentric coords.
    w0, w1, w2 = _rand_barycentric_coords(
        num_valid_meshes, num_samples, verts.dtype, verts.device
    )

    # Use the barycentric coords to get a point on each sampled face.
    a = v0[sample_face_idxs]  # (N, num_samples, 3)
    b = v1[sample_face_idxs]
    c = v2[sample_face_idxs]
    samples[meshes.valid] = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c

    if return_normals:
        # Initialize normals tensor with fill value 0 for empty meshes.
        # Normals for the sampled points are face normals computed from
        # the vertices of the face in which the sampled point lies.
        normals = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)
        vert_normals = (v1 - v0).cross(v2 - v1, dim=1)
        vert_normals = vert_normals / vert_normals.norm(dim=1, p=2, keepdim=True).clamp(
            min=sys.float_info.epsilon
        )
        vert_normals = vert_normals[sample_face_idxs]
        normals[meshes.valid] = vert_normals

    if return_textures:
        # fragment data are of shape NxHxWxK. Here H=S, W=1 & K=1.
        pix_to_face = sample_face_idxs.view(len(meshes), num_samples, 1, 1)  # NxSx1x1
        bary = torch.stack((w0, w1, w2), dim=2).unsqueeze(2).unsqueeze(2)  # NxSx1x1x3
        # zbuf and dists are not used in `sample_textures` so we initialize them with dummy
        dummy = torch.zeros(
            (len(meshes), num_samples, 1, 1), device=meshes.device, dtype=torch.float32
        )  # NxSx1x1
        fragments = MeshFragments(
            pix_to_face=pix_to_face, zbuf=dummy, bary_coords=bary, dists=dummy
        )
        textures = meshes.sample_textures(fragments)  # NxSx1x1xC
        textures = textures[:, :, 0, 0, :]  # NxSxC

    # return
    # TODO(gkioxari) consider returning a Pointclouds instance [breaking]
    if return_normals and return_textures:
        # pyre-fixme[61]: `normals` may not be initialized here.
        # pyre-fixme[61]: `textures` may not be initialized here.
        return samples, normals, textures
    if return_normals:  # return_textures is False
        # pyre-fixme[61]: `normals` may not be initialized here.
        return samples, normals
    if return_textures:  # return_normals is False
        # pyre-fixme[61]: `textures` may not be initialized here.
        return samples, textures
    if return_sample_info:
        return samples, (sample_face_idxs, w0, w1, w2)
    return samples


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


class LiifTrainDataset(Dataset):
    def __init__(self, data_path, res=(336, 512), drop_cam=[], split='train', cam_id=None, test_num=30, return_ray=False, num_frame=20, adjust=True, mano_layer=mano_layer, hand_type='right', limit=None, subject_limit=None):
        self.data_path = data_path
        self.adjust = adjust
        self.res = res
        self.hand_type = hand_type
        self.mano_layer = mano_layer
        self.faces = mano_layer[self.hand_type].faces
        self.limit = limit
        
        with open(join(data_path, 'annotations/%s' % 'train', 'InterHand2.6M_%s_camera.json' % 'train')) as f:
            cam_params = json.load(f)
        with open(join(data_path, 'annotations/%s' % 'train', 'InterHand2.6M_%s_MANO_NeuralAnnot.json' % 'train')) as f:
            mano_params = json.load(f)
        
        if subject_limit is None:
            subject_limit = -2
        
        capture_names = sorted(os.listdir(join(data_path, 'images/%s' % 'train')))
        capture_names = capture_names[:subject_limit] if split == 'train' else capture_names[-4:]
            
        self.datas = []
        for capture_name in capture_names:
            capture_idx = capture_name.replace('Capture', '')
            cam_param = cam_params[capture_idx]
            data_names = sorted(os.listdir(join(data_path, 'images/%s' % 'train', capture_name)))[:87]
            # data_names = sorted(os.listdir(join(data_path, 'images/%s' % 'train', capture_name)))[:8]
            

            self.two_hand = False
            for data_name in data_names:
                
                camera_names = [i for i in sorted(os.listdir(join(data_path, 'images/%s' % 'train', capture_name, data_name))) if i not in drop_cam and '400' in i]
                
                num = len(camera_names)
                # import pdb; pdb.set_trace()
                
                for cam_name in camera_names:
                    img_files = os.listdir(join(data_path, 'images/%s' % 'train', capture_name, data_name, cam_name))
                    img_names = sorted([file for file in img_files if file.endswith(".jpg")])
                    
                    # img_names = img_names[::max(len(img_names) // num_frame, 1)][:num_frame]
                    # print('image views num: %d, frames num: %d' % (num, len(img_names)))
                    # print(data_name, img_names, camera_names)
                    for img_name in img_names:
                        mano_param = mano_params[capture_idx][str(int(img_name[5:-4]))]
                        both_hand = True if mano_param['right'] is not None and mano_param['left'] is not None else False
                        if mano_param[hand_type] is None or both_hand:
                            continue
                        
                        datum = {}
                        datum['mano_param'] = mano_param[hand_type]
                        
                        # for i , cam_name in enumerate(camera_names):
                        cam_idx = cam_name.replace('cam', '')
                        datum['cam_idx'] = cam_idx
                        datum['hand_type'] = hand_type
                        img_path = join(data_path, 'images/%s' % 'train', capture_name, data_name, 'cam' + cam_idx, img_name)
                        if not exists(img_path):
                            continue
                        datum['img_path'] = img_path
                        
                        t, R = np.array(cam_param['campos'][str(cam_idx)], dtype=np.float32).reshape(3), np.array(cam_param['camrot'][str(cam_idx)], dtype=np.float32).reshape(3, 3)
                        focal = np.array(cam_param['focal'][str(cam_idx)], dtype=np.float32).reshape(2)
                        princpt = np.array(cam_param['princpt'][str(cam_idx)], dtype=np.float32).reshape(2)
                        
                        scale_mats = np.eye(4)
                        scale_mats[:3, :3] = R
                        cam_t = -np.dot(R, t.reshape(3, 1)).reshape(3) / 1000
                        scale_mats[:3, 3] = cam_t

                        cameraIn = np.array([[focal[0], 0, princpt[0]],
                                            [0, focal[1], princpt[1]],
                                            [0, 0, 1]])
                        
                        datum['cam_param'] = {
                            'R': R,
                            't': t,
                            'cam_t': cam_t,
                            'scale_mats': scale_mats,
                            'focal': focal,
                            'princpt': princpt,
                            'In': cameraIn,
                        }    
                        self.datas.append(datum)        
        
        if self.limit:
            self.datas = self.datas[:limit]
        
        self.glctx = dr.RasterizeCudaContext()
        
    def __len__(self,):
        return len(self.datas)
    
    
    def __getitem__(self, idx):
    
        data = deepcopy(self.datas[idx])
        
        vertices = []
        mano_out = []
        mano_param = data['mano_param']
        hand_type = data['hand_type']
    
        mano_pose = torch.FloatTensor(mano_param['pose']).view(-1, 3)
        root_pose = mano_pose[0].view(1, 3)
        hand_pose = mano_pose[1:, :].view(1, -1)
        shape = torch.FloatTensor(mano_param['shape']).view(1, -1)
        trans = torch.FloatTensor(mano_param['trans']).view(1, 3)
        
        output = self.mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
        joints = output.joints
        vertices = output.vertices
        faces = torch.from_numpy(self.faces.astype(np.int32)).int().cuda()
        
        cam_ex = torch.cat([torch.from_numpy(data['cam_param']['R']), torch.from_numpy(data['cam_param']['cam_t']).reshape(3, 1)], dim=-1).float()
        hand = rotate(vertices[0].cpu().permute(1, 0), cam_ex).permute(1, 0)
        hand2d = projection(hand.permute(1, 0), torch.from_numpy(data['cam_param']['In']).float()).permute(1, 0)[:, :2]
        
        img = cv2.imread(data['img_path'])
        img = img_adjust(img) * 255 if self.adjust else img

        [img], _, cameraIn = cut_img([img], np.array(hand2d[None,]), camera=data['cam_param']['In'], radio=0.8, img_size=self.res[0])
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        cam_i = torch.from_numpy(cameraIn).float()
        
        P = cameraIn @ data['cam_param']['scale_mats'][:3]
        proj, w2c = load_K_Rt_from_P(P[:3])
        
        proj[0, 0] = proj[0, 0] / (self.res[0] / 2.)
        proj[0, 2] = proj[0, 2] / (self.res[0] / 2.) - 1.
        proj[1, 1] = proj[1, 1] / (self.res[1] / 2.)
        proj[1, 2] = proj[1, 2] / (self.res[1] / 2.) - 1.
        proj[2, 2] = 0.
        proj[2, 3] = -0.1
        proj[3, 2] = 1.
        proj[3, 3] = 0.

        img = torch.from_numpy((img / 255.)).float().permute(2, 0, 1)
        grayimg = torch.from_numpy((grayimg / 255.)).float()
        
        w2cs = torch.from_numpy(w2c).unsqueeze(0).permute(0, 2, 1).cuda()
        projs = torch.from_numpy(proj).float().unsqueeze(0).permute(0, 2, 1).cuda()    
        
        vertsw = torch.cat([vertices, torch.ones_like(vertices[:, :, 0:1])], axis=2).expand(1, -1, -1).cuda()
        rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2cs)
        proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, projs)

        ## Joints
        rot_joints = rotate(joints[0].cpu().permute(1, 0), cam_ex).permute(1, 0)
        joints2d = projection(rot_joints.permute(1, 0), cam_i).permute(1, 0) 
        joints2d[:, :2] = joints2d[:, :2] / 256.0
        align_joints = rot_joints - rot_joints[0]
        joints25d = np.concatenate((joints2d[:, :2], align_joints[:, 2:]), axis=1)
        
        joint_img = draw_joint(joints25d, torch.zeros_like(img))
        
        ## Queries
        queries = sample_points_from_meshes(meshes=Meshes(vertices, faces.unsqueeze(0).cpu()), num_samples=2000, )[0].float()
    
        # rot_queries = rotate(queries.permute(1, 0), cam_ex).permute(1, 0)
        # q_2d = projection(rot_queries.permute(1, 0), cam_i).permute(1, 0) 
        # q_2d[:, :2] = q_2d[:, :2] / 256.0
        # align_queries = rot_queries - min(rot_queries[:, -1])
        # q_25d = torch.cat([q_2d, align_queries[:, -1:]], dim=-1)
        
        
        ## Depth
        rast_out, _ = dr.rasterize(self.glctx, proj_verts, faces, resolution=(self.res[1], self.res[0]))
        
        depth = rast_out[0, ..., 2:3]
        depth[depth != 0] = depth[depth != 0] - depth[depth != 0].min()
        depth = depth / depth.max()
        depth = depth.repeat(1, 1, 1, 3)


        ## Normal
        normals = get_normals(vertsw[:, :, :3], faces.long())
        
        feat = torch.ones_like(vertsw[:, :, :1])
        feat = torch.cat([feat, normals], dim=2)
        feat, _ = dr.interpolate(feat, rast_out, faces)
        masks = feat[:, :, :, :1].contiguous().squeeze(-1)
        gt_normals = feat[:, :, :, 1:4].contiguous()
        
        valid_idx = torch.where((masks > 0) & (rast_out[:, :, :, 3] > 0))
        
        gt_normals = dr.antialias(gt_normals, rast_out, proj_verts, faces)
        normal_img = torch.zeros_like(gt_normals)
        normal_img[valid_idx] = gt_normals[valid_idx]
        
        ray_directions = []
        c2ws = torch.inverse(w2cs)
        
        
        cam_ray_direction = get_ray_directions(self.res[1], self.res[0], data['cam_param']['focal'][0],
                                                data['cam_param']['focal'][1],
                                                data['cam_param']['princpt'][0],
                                                data['cam_param']['princpt'][1], ).cuda()
        
        tmp_ray_direction, _ = get_rays(cam_ray_direction, c2ws[0])

        ray_direction = tmp_ray_direction.reshape(self.res[1], self.res[0], 3).cpu()
        ray_directions.append(ray_direction)
        ray_directions = torch.stack(ray_directions)
        
        
        return {
            'img': img.cuda(),
            'img_path': data['img_path'],
            'masks': masks.cuda(), 
            'w2c': torch.from_numpy(w2c).permute(1, 0).cuda(), 
            'proj': torch.from_numpy(proj).float().permute(1, 0).cuda(), 
            'joint_img': joint_img.cuda(),
            'joints25d': torch.from_numpy(joints25d).float().cuda(),
            'mano_pose': mano_pose.cuda().reshape(-1), 
            'shape': shape[0].cuda(), 
            'trans': trans[0].cuda(), 
            'ray_directions': ray_directions[0].cuda(),
            'vertices': vertices[0].float().cuda(),
            'hand_types': hand_type, 
            'normals' : normal_img.squeeze(0).permute(2, 0, 1).cuda(),
            'depth' : depth.squeeze(0).permute(2, 0, 1).cuda(),
            'queries' : queries.cuda(),
            'cam': {
                'ex': cam_ex.cuda(),
                'in': cam_i.cuda()    
            }
        }