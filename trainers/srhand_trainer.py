import os
import pdb

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
from os.path import join
from glob import glob
from tqdm import tqdm
import argparse
import pickle
from pyhocon import ConfigFactory
import numpy as np
import cv2
import lpips
import smplx
import time
import trimesh

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import GaussianBlur

import nvdiffrast.torch as dr
from get_data import *
from models.MVIHand import MVIHand
from models.giif import GIIF
from models.GAN import UNetDiscriminatorSN
from models.utils import *

from repose import lbs, lbs_pose, pose2rot
import json
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR 
from matplotlib import pyplot as plt
import pytorch_wavelets as wavelet
from chamfer_distance import ChamferDistance
import wandb

ih_cam_names = ['cam400262', 'cam400263', 'cam400264', 'cam400265', 'cam400266', 'cam400267', 'cam400268', 'cam400269','cam400270', 'cam400271', 'cam400272', 'cam400273', 'cam400274', 'cam400275', 'cam400276', 'cam400279','cam400280', 'cam400281', 'cam400282', 'cam400283', 'cam400284', 'cam400285', 'cam400287', 'cam400288','cam400289', 'cam400290', 'cam400291', 'cam400292', 'cam400293', 'cam400294', 'cam400296', 'cam400297','cam400298', 'cam400299', 'cam400300', 'cam400301', 'cam400310', 'cam400312', 'cam400314', 'cam400315','cam400316', 'cam400317', 'cam400319', 'cam400320', 'cam400321', 'cam400322', 'cam400323', 'cam400324','cam400326', 'cam400327']
go_cam_names = ['401106', '401107', '401108', '401643', '401645', '401646', '401647', '401650', '401652', '401653', '401654', '401655', '401658', '401659', '401660', '401765', '401891', '401892', '401893', '401894']

MEAN_HAND_ALBEDO = torch.tensor([0.31996773101, 0.36127372, 0.44126652]).cuda()
# MEAN_HAND_ALBEDO = torch.tensor([73/ 255, 112/255, 117/255]).cuda()


def unconfidence_to_color(queries):
    min_dist = np.min(queries)
    max_dist = np.max(queries)
    norm_dist = (queries - min_dist) / (max_dist - min_dist + 1e-8)
    
    cmap = plt.get_cmap("jet")
    colors = np.zeros((*queries.shape, 3))
    for b in range(queries.shape[0]):
        colors[b] = cmap(norm_dist[b])[:, :3] 
        
    
    return colors


def cal_query_mean_var(query, ref_imgs):
    visible_counts = query[..., 2:].sum(0)
    visible_counts_clamped =  visible_counts.clamp(min=1)
                    
    query_color = torch.zeros_like(query)
    B = query.shape[0]
    for b in range(B):
        coords = (query[b, query[b, :, 2] == 1] * 255).long().clamp(0, 255)
        query_color[b, query[b, :, 2] == 1] = ref_imgs[b, :, coords[:, 1], coords[:, 0]].permute(1, 0)
   
    mean_sr_queries = query_color.detach().sum(0) / visible_counts_clamped
    squared_diff = (query_color - mean_sr_queries.unsqueeze(0)) ** 2
    variance_visible = (squared_diff * query[..., 2:]).sum(dim=[0, -1]) / visible_counts_clamped.squeeze()
    variance_visible[visible_counts.squeeze() == 0] = 0

    return mean_sr_queries, variance_visible

class SRHandTrainer:
    def __init__(self, conf, weight, mano_layer, is_continue=False, model_path=None, optimizer_path=None, implicit_path=None, implicit_optimizer_path=None):
        self.type = conf.get_string('data_type')
        
        self.drop_cam = conf.get_string('drop_cam').split(',')
        self.drop_frame = conf.get_string('drop_frame', ['a,b']).split(',')
        self.cam_id = conf.get_string('cam_id')
        if "," in self.cam_id:
            self.cam_id = self.cam_id.split(',')
        elif len(self.cam_id) > 0:
            self.cam_id = [self.cam_id]
        self.using_view = conf.get_int('using_view')
        self.num_view = conf.get_int('num_view')
        self.num_frame = conf.get_int('num_frame')
        self.num = conf.get_int('num')
        self.capture_name = conf.get_string('capture_name')
        self.adjust = conf.get_bool('adjust')
        self.w = conf.get_int('w')
        self.h = conf.get_int('h')
        self.lr_size = conf.get_int('lr_size')
        self.net_type = conf.get_string('net_type')
        self.use_pe = conf.get_bool('use_pe')
        self.use_x_pos = conf.get_bool('use_x_pos')
        self.use_ray = conf.get_bool('use_ray')
        self.use_emb = conf.get_bool('use_emb')
        self.mlp_use_pose = conf.get_bool('mlp_use_pose')
        self.use_rotpose = conf.get_bool('use_rotpose')
        self.pose_consist = conf.get_bool('pose_consist', False)
        self.pose_weight = conf.get_float('pose_weight', False)
        self.use_template = conf.get_bool('use_template', False)
        self.wo_latent = conf.get_bool('wo_latent')
        self.latent_num = conf.get_int('latent_num')
        self.resolution = (self.h, self.w)
        self.epoch_albedo = conf.get_int('epoch_albedo')
        self.epoch_sfs = conf.get_int('epoch_sfs')
        self.epoch_train = conf.get_int('epoch_train')
        self.sfs_weight = conf.get_float('sfs_weight')
        self.geo_weight = conf.get_float('geo_weight', 0)
        self.lap_weight = conf.get_float('lap_weight')
        self.albedo_weight = conf.get_float('albedo_weight')
        self.mask_weight = conf.get_float('mask_weight')
        self.edge_weight = conf.get_float('edge_weight')
        self.delta_weight = conf.get_float('delta_weight')
        self.use_enh = conf.get_bool('use_enh', False)
        self.use_consist = conf.get_bool('use_consist', False)
        self.delta_scale = conf.get_float('delta_scale', 1)
        
        
        if not self.use_enh: # original xhand
            self.geo_weight = 0
            # self.mask_weight = 0
            self.delta_scale = 1
            
        self.part_smooth = conf.get_float('part_smooth')
        self.use_sum = conf.get_float('use_sum')
        self.degree = conf.get_int('degree')
        self.batch = conf.get_int('batch')
        self.lr = conf.get_float('lr')
        self.albedo_lr = conf.get_float('albedo_lr')
        self.sh_lr = conf.get_float('sh_lr')
        self.lbs_weight = weight
        self.implicit_update_term = conf.get_int('implicit_update_term', 50)
        self.hand_type = conf.get_string('hand_type')
        self.z_channels = conf.get_int('z_channels')
        self.ch_mult = conf.get_list('ch_mult')
        
        self.exp_name = conf.get_string('exp_name')
        self.use_liif = conf.get_bool('use_liif', False)
        self.liif_use_3d = conf.get_bool('liif_use_3d', False)
        self.liif_pretrain_path = conf.get_string('liif_model', None)
        self.liif_epoch = conf.get_int('liif_epoch', 0)
        self.epoch_implicit = conf.get_int('epoch_implicit', 10)
        self.wavelet_weight = conf.get_float('wavelet_weight', 1)
        self.disc_weight = conf.get_float('disc_weight', 1)
        self.cons_weight = conf.get_float('cons_weight', 1)
        
        self.is_continue = is_continue
        self.model_path = model_path
        self.optimizer_path = optimizer_path
        self.implicit_path = implicit_path
        self.implicit_optimizer_path = implicit_optimizer_path
        
        self.use_disc = conf.get_bool('use_disc', False)
        if self.use_disc:
            self.disc_path = conf.get_string('disc_path')
        
        self.lpips_loss = lpips.LPIPS(net='vgg').cuda()
        
        self.use_wandb = conf.get_bool('use_wandb', False)
        self.log_step = conf.get_int('log_step', 10)
                
        self.mano_layer = {}
        self.mano_layer['right'] = mano_layer['right'].cuda()
        self.mano_layer['left'] = mano_layer['left'].cuda()
        
        self.dataset = {}
        self.dataloader = {}

        self.perm = torch.randperm(self.using_view)
        
        self.blur = GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        self.glctx = dr.RasterizeCudaContext()
        if self.use_wandb:
            wandb.init(project='XHand', name=conf.get_string('exp_name'), job_type='train')
            wandb.run.log_code(root='/workspace/datasets/XHand/', include_fn=lambda p: any(p.endswith(ext) for ext in ('.py', '.json', '.yaml', '.md', '.txt.', '.gin')), exclude_fn=lambda p: any(s in p for s in ('output', 'tmp', 'wandb', '.git', '.vscode')))
        
        self.dwt2d = wavelet.DWTForward(J=3, wave='db1', mode='zero').cuda()
        self.chamfer_distance = ChamferDistance()
        
    def finish(self, ):
        if self.use_wandb:
            wandb.finish()
        time.sleep(3)
        
    def prepare_data(self, data_path, data_type='interhand', data_name='ROM03_RT_No_Occulsion', split='test'):
        self.data_name = data_name
        
        if split == 'train':
            if data_type == 'interhand':
                self.imgs, self.lr_imgs, self.grayimgs, self.masks, self.w2cs, self.projs, self.cam_exs, self.cam_ins, self.poses, self.shapes, self.transs, self.hand_types, self.mano_vertices, self.rays = get_interhand_cam_seqdatabyframe(
                    data_path, res=(self.w, self.h), lr_size=self.lr_size, data_name=data_name, capture_name=self.capture_name, drop_cam=self.drop_cam, split='test', return_ray=True, cam_id=self.cam_id, test_num=self.num_view, num_frame=self.num_frame, adjust=self.adjust
                )
                
        if split == 'test':
            if data_type == 'interhand':
                self.imgs, self.lr_imgs, self.grayimgs, self.masks, self.w2cs, self.projs, self.cam_exs, self.cam_ins, self.poses, self.shapes, self.transs, self.hand_types, self.rays, self.mano_vertices, self.img_names = get_interhand_cam_test_seqdatabyframe(
                data_path, res=(self.w, self.h), data_name=data_name, capture_name=self.capture_name, drop_cam=self.drop_cam, cam_id=self.cam_id, split=split, return_ray=True, adjust=self.adjust, num_frame=self.num_frame)   # , cam_id='cam400267')
        
        
        
        self.scales = torch.ones(self.transs.shape[0], 1)
        self.num_frame = self.imgs.shape[0]
        self.num_view =  self.imgs.shape[1]
        self.ori_using_view = self.using_view
        
    def load_subdivide_smpl(self, type='mano'):
        ori_v = []
        ori_f = []
        vertices = []
        faces = []
        mano_vertices = []
        mano_faces = []
        weights = []
        J_regressor_new = []
        mean_shape = self.shapes.mean(0, keepdim=True).cpu()
        self.mano_faces = self.mano_layer['right'].faces
        for i, hand_type in enumerate(self.hand_types):
            vertices_T = self.mano_layer[hand_type].v_template
            vertices_T = torch.einsum('bl,mkl->bmk', [mean_shape, self.mano_layer[hand_type].shapedirs.cpu()]) + vertices_T.cpu()
            
            mano_vertices.append(vertices_T[0].cuda())
            faces_T = self.mano_layer[hand_type].faces
            ori_v.append(vertices_T[0].cuda())
            ori_f.append(torch.from_numpy(faces_T.astype(np.int32) + vertices_T.shape[1] * i).cuda())
            v, f = trimesh.remesh.subdivide_loop(vertices_T[0].numpy(), faces_T.astype(np.int64), iterations=3)
            f = f + i * v.shape[0]
            self.len_v = v.shape[0]
            self.len_f = f.shape[0]
            
            vertices.append(torch.from_numpy(v.astype(np.float32)).cuda())
            mano_faces.append(torch.from_numpy(faces_T.astype(np.int32)).cuda())
            faces.append(torch.from_numpy(f.astype(np.int32)).cuda())
            weights.append(torch.from_numpy(self.lbs_weight[hand_type]['weights']).float().cuda())
            
            new_J_regressor = torch.zeros(16, self.len_v).to(self.mano_layer[hand_type].J_regressor.dtype).to(self.mano_layer[hand_type].J_regressor.device)
            J_regressor = torch.cat((self.mano_layer[hand_type].J_regressor, new_J_regressor), dim=1)
            J_regressor_new.append(J_regressor)
    
        self.ori_v = torch.cat(ori_v, 0)
        self.ori_f = torch.cat(ori_f, 0)
        self.vertices = torch.cat(vertices, 0)
        self.faces = torch.cat(faces, 0)
        self.mano_faces = torch.cat(mano_faces, 0)
        self.weights = torch.cat(weights, 0)
        self.J_regressor = torch.cat(J_regressor_new, 0)
        
    def initialize_model(self,):
        self.load_subdivide_smpl()

        self.np_faces = self.faces.squeeze().detach().cpu().numpy()
        self.np_mano_faces = self.mano_faces.squeeze().detach().cpu().numpy()
        
        self.sh_coeffs = torch.zeros(self.num_view, 27).cuda()
        self.albedo = (torch.zeros_like(self.vertices)).unsqueeze(0)
        self.delta = torch.zeros_like(self.vertices)

        self.vertices_tmp = torch.clone(self.vertices)
        self.sh_coeffs_not_initialized = True       
        
        self.implicit_net = None
        if self.use_liif:
            if self.liif_use_3d:
                self.implicit_net = GIIF(
                    encoder_spec={
                        'name': 'rdn-baseline',
                        'args': {
                            'n_feats': self.z_channels,
                            'no_upsampling': True
                        }
                    },
                    imnet_spec={
                        'name': 'mlp',
                        'args': {
                            'out_dim': 3,
                            'hidden_list': [256 ,256, 256, 256],
                        }
                    },
                    local_ensemble=True,
                    img_size = self.h,
                    mano_faces = self.mano_faces,
                ).cuda()
      
        
        
            liif_pretrain_path = join(self.liif_pretrain_path, 'giif.pth')
            if self.is_continue and os.path.exists(join(self.out_mesh_dire, 'implicit_model.pth')):
                liif_pretrain_path = join(self.out_mesh_dire, 'implicit_model.pth')
            
            print('Loading Pretrained GIIF from '+ liif_pretrain_path)
            state_dict = torch.load(liif_pretrain_path).state_dict()
            self.implicit_net.load_state_dict(state_dict, strict=False)
            self.implicit_net.eval()
            
            if self.use_disc:
                state_dict = torch.load(self.disc_path).state_dict()
                self.disc_net = UNetDiscriminatorSN(
                    num_in_ch = 3,
                    num_feat=64,
                    skip_connection=True    
                ).cuda()
                self.disc_net.load_state_dict(state_dict, strict=False)
        else:
            self.implict_net = None
        
    def set_log_path(self, out_dire, out_mesh_dire):
        self.out_dire = out_dire
        self.out_mesh_dire = out_mesh_dire
       
    def pretrain(self,):
        os.makedirs(join(self.out_mesh_dire, 'rerender'), exist_ok=True)
        sig = nn.Sigmoid()

        self.sh_coeffs.requires_grad_(True)
        self.albedo.requires_grad_(True)
        self.delta.requires_grad_(False)
        self.poses.requires_grad_(True)
        self.weights.requires_grad_(False)
        
        
        self.vertices_tmp = torch.clone(self.vertices)
        a = self.vertices[self.faces[:, 0].long()]
        b = self.vertices[self.faces[:, 1].long()]
        c = self.vertices[self.faces[:, 2].long()]

        self.edge_length_mean = torch.cat([((a - b) ** 2).sum(1), ((c - b) ** 2).sum(1), ((a - c) ** 2).sum(1)])
        

        self.step = 0
        self.update_sr(ori_step=True)
        
        if not self.is_continue or not os.path.exists(join(self.out_mesh_dire, 'seq.pt')):
            # init albedo/sh/delta
            for i in range(1):
                vertices = self.vertices_tmp + self.delta
                # compute sphere harmonic coefficient as initialization
                optimizer = Adam([{'params': self.albedo, 'lr': self.albedo_lr},
                                {'params': self.sh_coeffs, 'lr': self.sh_lr}, 
                                {'params': self.poses, 'lr': 0.0005}]
                            )

                pbar = tqdm(range(self.epoch_albedo))
                rendered_img = []
                for i in pbar:
                    perm = torch.randperm(self.using_view)
                    for k in range(0, self.using_view, self.batch):
                        n = min(self.using_view, k + self.batch) - k
                        img = self.imgs[0][perm[k:k + self.batch]].cuda()
                        sr_img = self.sr_imgs[0][perm[k:k + self.batch]].cuda()
                        lr_img = F.interpolate(self.lr_imgs[0][perm[k:k + self.batch]].permute(0, 3, 1, 2).cuda(), size=self.w).permute(0, 2 ,3 ,1)
                        w2c = self.w2cs[0][perm[k:k + self.batch]].cuda()
                        proj = self.projs[0][perm[k:k + self.batch]].cuda()
                        mask = self.masks[0][perm[k:k + self.batch]].cuda()
                        sh_coeff = self.sh_coeffs[perm[k:k + self.batch]]
                        pose = self.poses[0:1].cuda()
                        shape = self.shapes[0:1].cuda()
                        trans = self.transs[0:1].cuda()
                        scale = self.scales[0:1].cuda()
                        vertices_n = vertices.unsqueeze(0)
                        

                        vertices_new = []
                        for idx, hand_type in enumerate(self.hand_types):
                            pose_new = pose[:, idx * 16:(idx + 1) * 16, :].view(1, -1)
                            shape_new = shape[:, idx * 10:(idx + 1) * 10].view(1, -1)

                            trans_new = trans[:, idx * 3:(idx + 1) * 3].view(1, -1)
                            weights_new =self. weights[idx * self.len_v:(idx + 1) * self.len_v]
                            verts_new = lbs_pose(
                                pose_new.clone(), 
                                self.ori_v[idx * 778: (idx + 1) * 778].unsqueeze(0), 
                                weights_new, 
                                vertices_n[:, idx * self.len_v:(idx + 1) * self.len_v], 
                                hand_type=hand_type
                            )
                            verts_new = verts_new * scale + trans_new.unsqueeze(1)
                            vertices_new.append(verts_new)
                        vertices_new = torch.cat(vertices_new, 1)

                        vertsw = torch.cat([vertices_new, torch.ones_like(vertices_new[:, :, 0:1])], axis=2).expand(n, -1, -1)
                        rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2c)
                        proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, proj)
                        normals = get_normals(rot_verts[:, :, :3], self.faces.long())

                        rast_out, _ = dr.rasterize(self.glctx, proj_verts, self.faces, resolution=self.resolution)
                        feat = torch.cat([normals, self.albedo.expand(n, -1, -1), torch.ones_like(vertsw[:, :, :1])], dim=2)
                        feat, _ = dr.interpolate(feat, rast_out, self.faces)
                        pred_normals = feat[:, :, :, :3].contiguous()
                        rast_albedo = feat[:, :, :, 3:6].contiguous()
                        pred_mask = feat[:, :, :, 6:7].contiguous()
                        pred_normals = dr.antialias(pred_normals, rast_out, proj_verts, self.faces)
                        pred_normals = F.normalize(pred_normals, p=2, dim=3)
                        rast_albedo = dr.antialias(rast_albedo, rast_out, proj_verts, self.faces)
                        pred_mask = dr.antialias(pred_mask, rast_out, proj_verts, self.faces).squeeze(-1)
                        
                        valid_idx = torch.where((mask > 0) & (rast_out[:, :, :, 3] > 0))
                        valid_normals = pred_normals[valid_idx]
                        valid_shcoeff = sh_coeff[valid_idx[0]]
                        valid_albedo = sig(rast_albedo[valid_idx])

                        valid_img = sr_img[valid_idx]
                        pred_img = torch.clip(
                            compute_color(valid_albedo.unsqueeze(0), valid_normals.unsqueeze(0), valid_shcoeff)[0], 0, 1)
                        

                        sfs_loss = self.sfs_weight * F.l1_loss(pred_img, valid_img)
                        albedo_loss = F.mse_loss(MEAN_HAND_ALBEDO, valid_albedo.mean(0))
                        # albedo_loss = albedo_weight * laplacian_smoothing(albedo.squeeze(0), faces.long(), method="uniform")
                        # lap_w = lap_weight * laplacian_smoothing(weights, faces.long(), method="uniform") / 10
                        mask_loss = self.mask_weight * F.mse_loss(pred_mask, mask)

                        loss = sfs_loss + mask_loss + albedo_loss

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        des = 'sfs:%.4f' % sfs_loss.item() + ' mask:%.4f' % mask_loss.item()
                        pbar.set_description(des)

            self.delta.requires_grad_(True)
            self.sh_coeffs.requires_grad_(True)
            
            optimizer = Adam([{'params': self.delta, 'lr': self.lr},
                              {'params': self.sh_coeffs, 'lr': self.sh_lr},
                              {'params': self.poses, 'lr': 0.0005},
                              {'params': self.albedo, 'lr': self.albedo_lr}])

            pbar = tqdm(range(self.epoch_sfs))
            for i in pbar:
                perm = torch.randperm(self.using_view)
                if i == self.epoch_sfs // 2:
                    self.lap_weight = self.lap_weight * 10
                for k in range(0, self.using_view, self.batch):
                    vertices = self.vertices_tmp + self.delta
                    n = min(self.using_view, k + self.batch) - k
                    w2c = self.w2cs[0][perm[k:k + self.batch]].cuda()
                    proj = self.projs[0][perm[k:k + self.batch]].cuda()
                    img = self.imgs[0][perm[k:k + self.batch]].cuda()
                    sr_img = self.sr_imgs[0][perm[k:k + self.batch]].cuda()
                    lr_img = F.interpolate(self.lr_imgs[0][perm[k:k + self.batch]].permute(0, 3, 1, 2).cuda(), size=self.w).permute(0, 2 ,3 ,1)
                    mask = self.masks[0][perm[k:k + self.batch]].cuda()
                    sh_coeff = self.sh_coeffs[perm[k:k + self.batch]]
                    pose = self.poses[0:1].cuda()
                    shape = self.shapes[0:1].cuda()
                    trans = self.transs[0:1].cuda()
                    scale = self.scales[0:1].cuda()
                    verts_mano = self.mano_vertices[0:1].cuda() 
                    vertices_n = vertices.unsqueeze(0)
                    
                    template_vertices = self.template_vertices[0:1].cuda() if self.use_template else None

                    # get posed verts
                    vertices_new = []
                    for idx, hand_type in enumerate(self.hand_types):
                        pose_new = pose[:, idx * 16:(idx + 1) * 16, :].view(1, -1)
                        trans_new = trans[:, idx * 3:(idx + 1) * 3].view(1, -1)
                        weights_new = self.weights[idx * self.len_v:(idx + 1) * self.len_v]

                        verts_new = lbs_pose(pose_new.clone(), self.ori_v[idx * 778: (idx + 1) * 778].unsqueeze(0), weights_new, vertices_n[:, idx * self.len_v:(idx + 1) * self.len_v], hand_type=hand_type)
                        verts_new = verts_new * scale + trans_new.unsqueeze(1)
                        vertices_new.append(verts_new)
                    vertices_new = torch.cat(vertices_new, 1)
                    
                    vertsw = torch.cat([vertices_new, torch.ones_like(vertices_new[:, :, 0:1])], axis=2).expand(n, -1, -1)
                    rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2c)
                    proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, proj)
                    normals = get_normals(rot_verts[:, :, :3], self.faces.long())

                    rast_out, _ = dr.rasterize(self.glctx, proj_verts, self.faces, resolution=self.resolution)
                    feat = torch.cat([normals, self.albedo.expand(n, -1, -1), torch.ones_like(vertsw[:, :, :1])], dim=2)
                    feat, _ = dr.interpolate(feat, rast_out, self.faces)
                    pred_normals = feat[:, :, :, :3].contiguous()
                    rast_albedo = feat[:, :, :, 3:6].contiguous()
                    pred_mask = feat[:, :, :, 6:7].contiguous()
                    pred_normals = F.normalize(pred_normals, p=2, dim=3)
                    pred_mask = dr.antialias(pred_mask, rast_out, proj_verts, self.faces).squeeze(-1)

                    valid_idx = torch.where((mask > 0) & (rast_out[:, :, :, 3] > 0))
                    valid_normals = pred_normals[valid_idx]
                    valid_shcoeff = sh_coeff[valid_idx[0]]
                    valid_albedo = sig(rast_albedo[valid_idx])

                    valid_img = sr_img[valid_idx]
                    pred_img = torch.clip(
                        compute_color(valid_albedo.unsqueeze(0), valid_normals.unsqueeze(0), valid_shcoeff)[0], 0, 1)

                    tmp_img = torch.zeros_like(img)
                    tmp_img[valid_idx] = pred_img
                    tmp_img = dr.antialias(tmp_img, rast_out, proj_verts, self.faces)

                    sfs_loss = self.sfs_weight * (F.l1_loss(tmp_img[valid_idx], valid_img))

                    lap_delta_loss = self.lap_weight * laplacian_smoothing(self.delta, self.faces.long(), method="uniform", return_sum=False)
                    lap_delta_loss = lap_delta_loss[lap_delta_loss > torch.quantile(lap_delta_loss, 0.25)].sum()
                    lap_vert_loss = self.lap_weight * laplacian_smoothing(vertices, self.faces.long(), method="uniform", return_sum=False)
                    lap_vert_loss = lap_vert_loss[lap_vert_loss < torch.quantile(lap_vert_loss, 0.25)].sum()
                    albedo_loss = self.albedo_weight * laplacian_smoothing(self.albedo.squeeze(0), self.faces.long(), method="uniform", return_sum=False)
                    albedo_loss = albedo_loss[albedo_loss > torch.quantile(albedo_loss, 0.25)].sum() + F.mse_loss(MEAN_HAND_ALBEDO, valid_albedo.mean(0)) * 100

                    lap_w = laplacian_smoothing(self.weights, self.faces.long(), method="uniform")

                    # normal_loss = 0.0 * normal_consistency(vertices, faces.long())
                    normal_loss = torch.zeros_like(albedo_loss)
                    mask_loss = self.mask_weight * F.mse_loss(pred_mask, mask)
                    a = vertices[self.faces[:, 0].long()]
                    b = vertices[self.faces[:, 1].long()]
                    c = vertices[self.faces[:, 2].long()]
                    edge_length = torch.cat(
                        [((a - b) ** 2).sum(1), ((c - b) ** 2).sum(1), ((a - c) ** 2).sum(1)])

                    edge_loss = torch.clip(edge_length - self.edge_length_mean, 0, 1).mean() * self.edge_weight
                    delta_loss = (self.delta ** 2).sum(1).mean() * self.delta_weight
                    
                    cd_loss = torch.tensor(0).cuda()
                    if self.use_template:
                        dist1, dist2,_, _ = self.chamfer_distance(template_vertices, vertices[None,]) 
                        cd_loss = (torch.mean(dist1)) + (torch.mean(dist2)) 
                    
                    loss = sfs_loss + lap_delta_loss + lap_vert_loss + albedo_loss + mask_loss + normal_loss + delta_loss + edge_loss + lap_w + cd_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    des = 'sfs:%.4f' % sfs_loss.item() + ' lap:%.4f' % lap_delta_loss.item() + ' albedo:%.4f' % albedo_loss.item() + ' mask:%.4f' % mask_loss.item() + ' normal:%.4f' % normal_loss.item() + ' edge:%.4f' % edge_loss.item() + ' delta:%.4f' % delta_loss.item() + ' weight:%.4f' % lap_w.item() 
                    
                    pbar.set_description(des)
              
            del optimizer
   
            # init pretrain implicit
            self.delta.requires_grad_(False)
            self.sh_coeffs.requires_grad_(False)

            torch.save({'sh_coeff': self.sh_coeffs, 'albedo': self.albedo, 'delta': self.delta}, join(self.out_mesh_dire, 'seq.pt'))

            np_verts = vertices.squeeze().detach().cpu().numpy()
            
            mesh = trimesh.Trimesh(np_verts, self.np_faces, process=False, maintain_order=True)
            mesh.export(join(self.out_mesh_dire, 'seq.obj'))

            save_obj_mesh_with_color(join(self.out_mesh_dire, 'seq_c.obj'), np_verts, self.np_faces, (sig(self.albedo).detach().cpu().numpy()[0])[:, 2::-1])
            
        else:
            state = torch.load(join(self.out_mesh_dire, 'seq.pt'))
            self.sh_coeffs = state['sh_coeff']
            self.albedo = state['albedo']
            self.delta = state['delta']
            
                    
    def update_sr(self, ori_step=False):
        os.makedirs(join(self.out_mesh_dire, 'rerender', '%02d'  % (self.step)), exist_ok=True)
        if self.use_liif:
            self.implicit_net.eval()
        
        rendered_img = []
        psnrs = []
        
        self.img_name = []
        self.sr_imgs = []
        
        print("Update SR")
        pbar = tqdm(range(self.num_frame))
        for i in pbar:
            mv_imgs = []
            
            pose =  self.poses[i:i + 1].cuda()
            shape = self.shapes[i:i + 1].cuda()
            trans = self.transs[i:i + 1].cuda()
            scale = self.scales[i:i + 1].cuda()
            
            verts_mano = self.mano_vertices[i:i+1].cuda() 
            vertices_new = []
            
            for idx, hand_type in enumerate(self.hand_types):
                pose_new = pose[:, idx * 16:(idx + 1) * 16, :].view(1, -1)
                trans_new = trans[:, idx * 3:(idx + 1) * 3].view(1, -1)
                weights_new = self.weights[idx * self.len_v:(idx + 1) * self.len_v]
                verts_new = verts_mano
                vertices_new.append(verts_new)
            
            vertsw = torch.cat(vertices_new, 0)
            for k in range(0, self.num_view):
                n = min(self.using_view, k + 1) - k
                
                w2c = self.w2cs[i][k:k + 1].cuda()
                proj = self.projs[i][k:k + 1].cuda()
                img = self.imgs[i][k:k + 1].cuda()
                lr_img = self.lr_imgs[i][k:k + 1].permute(0, 3, 1, 2).cuda()
                mask = self.masks[i][k:k + 1].cuda()
                
                cam_exs = self.cam_exs[i][k: k + 1].cuda()
                cam_ins = self.cam_ins[i][k: k + 1].cuda()
                
                sh_coeff = self.sh_coeffs[k:k + 1]
                
                im_feat = None
                render_img = torch.cat([img, F.interpolate(lr_img, self.w).permute(0, 2, 3, 1)], dim=2)
                if self.use_liif:
                    with torch.no_grad():   
                        if self.liif_use_3d:
                            src_dict = {
                                'img': lr_img,
                                'vertices': verts_mano,
                                'w2c': w2c,
                                'proj': proj,
                                'cam': {
                                    'ex': cam_exs,
                                    'in': cam_ins
                                }
                                
                            }
                            sr_img, vert_img = self.implicit_net(src_dict, hr_size=(self.w, self.h), glctx=self.glctx)
                            sr_img = sr_img.permute(0, 2, 3, 1)
                            vert_img = vert_img.permute(0, 2, 3, 1)
                            sr_img[mask==0] = 0
                            
                            # vert_img, _, _ = self.implicit_net.render_hand(vertsw, self.mano_verts_color.unsqueeze(0), w2c, proj, self.glctx, out_size=(self.w, self.h))                   
                            # sr_img = self.implicit_net.forward_img(lr_img, vert_img, im_feat, out_size=(self.w, self.h)).permute(0, 2, 3, 1)

                            board = torch.zeros_like(vert_img[..., :3]).cuda()
                            normal = board + (0.5 + (vert_img[..., :3]) / 2)* mask.unsqueeze(-1)
                            render_img = torch.cat([render_img, normal], 2)
                        else:
                            src_dict = {
                                'img': lr_img,
                                'vertices': verts_mano,
                                'w2c': w2c,
                                'proj': proj,
                                'cam': {
                                    'ex': cam_exs,
                                    'in': cam_ins
                                }    
                            }
                            sr_img, _ = self.implicit_net(src_dict, hr_size=(self.w, self.h))      
                            sr_img = sr_img.permute(0, 2, 3, 1)
                            sr_img[mask==0] = 0
                    
                    
                else:
                    sr_img = F.interpolate(lr_img, size=self.h, mode='bicubic').permute(0, 2, 3, 1)
                
                render_img = torch.cat([render_img, sr_img], 2).clamp(0, 1).cpu()
                
                
                psnrs.append(calculate_psnr(img[0].detach().cpu().numpy(), np.clip(sr_img[0].detach().cpu().numpy(), 0, 1))) # , mask[0].detach().cpu().numpy()))
                
                rendered_img.append(
                    render_img
                )
                mv_imgs.append(sr_img[0].detach().cpu())
                self.img_name.append('%02d_%02d.png' % (k, i))
            
            self.sr_imgs.append(torch.stack(mv_imgs, dim=0))
            
        self.sr_imgs = torch.stack(self.sr_imgs, dim=0)
        
        if ori_step:
            self.ori_sr_imgs = self.sr_imgs.clone()
        print("Implicit PSNRS : {}".format(np.array(psnrs).mean()))
        
        rendered_img = torch.cat(rendered_img, 0)

        for i in range(rendered_img.shape[0]):
            cv2.imwrite(join(self.out_mesh_dire, 'rerender', '%02d'  % (self.step), self.img_name[i]), tensor2img(rendered_img, i))
        
        if self.use_wandb:
            log_dict = {
                'implicit_psnr': np.array(psnrs).mean(),
            }
            wandb.log(log_dict, step=self.step)
        
    def update_mean(self,):
        
        weights_mean = []
        delta_mean = []
        albedo_mean = []
        
        print('update mean')
        with torch.no_grad():
            for j in range(self.num_frame):
                pose =  self.poses[j:j + 1].reshape(1, -1).cuda()
                shape = self.shapes[j:j + 1].cuda()

                if self.use_rotpose:
                    matrix = pose2rot(pose.view(-1, 48).clone()).view([1, -1, 3, 3])
                else:
                    matrix = pose.clone()
                condition = matrix.reshape(1, -1) / np.pi
                pred_weights = self.model.forward_lbs(condition, False)[0][0]
                pred_albedo = self.model.forward_color(condition, False)[0][0]
                pred_delta = self.model.forward_delta(condition, pred_albedo.unsqueeze(0) if self.model.use_cond else None)[0][0]
                
                weights_mean.append(pred_weights)
                delta_mean.append(pred_delta)
                albedo_mean.append(pred_albedo)

        self.model.renew_mean(torch.stack(delta_mean, 0).mean(0).detach(),
                        torch.stack(albedo_mean, 0).mean(0).detach(),
                        torch.stack(weights_mean, 0).mean(0).detach())

    def implicit_update(self, sr_optimizer, epoch=0, render_resolution=None):
        os.makedirs(join(self.out_mesh_dire, 'unconfidence', '%02d'  % (self.step)), exist_ok=True)

        mano_anno = []
        cam_anno = []
        
        implicit_batch = 8
        
        self.implicit_net.train()
        render_resolution = render_resolution if render_resolution else self.resolution
        
        dwt_2d = wavelet.DWTForward(J=3, wave='db1', mode='zero').cuda()
        bce_loss = nn.BCELoss()
        
        
        # prev_mu = torch.zeros(self.vertices.shape[0]).cuda() 
        prev_mu = None
        perm_frame = torch.randperm(self.num_frame)
        ac_query_color = []
        for impl_e in range(self.epoch_implicit):
            pbar = tqdm(range(min(20, self.num_frame)))
            for j_perm in pbar:
                j = perm_frame[j_perm]
                pose =  self.poses[j:j + 1].reshape(1, -1).cuda()
                shape = self.shapes[j:j + 1].cuda()
                trans = self.transs[j:j + 1].cuda()
                scale = self.scales[j:j + 1].cuda()

                mano_anno.append({
                    'poses': np.round(np.asarray(pose.cpu().detach().numpy(), np.float64), 6).tolist(),
                    'shape': np.round(np.asarray(shape.cpu().detach().numpy(), np.float64), 6).tolist(),
                    'trans': np.round(np.asarray(trans.cpu().detach().numpy(), np.float64), 6).tolist(),
                    'scale': scale.cpu().tolist()
                })
                
                perm = self.perm
                
                for k in range(0, min(20, self.using_view), implicit_batch):
                    n = min(self.using_view, k + implicit_batch) - k
                    
                    ori_sr_img = self.ori_sr_imgs[j][perm[k:k + n]].cuda().permute(0, 3, 1, 2)
                    lr_img = self.lr_imgs[j][perm[k:k + n]].cuda() 
                                        
                    mask = self.masks[j][perm[k:k + n]].cuda()
                    
                    w2c =  self.w2cs[j][perm[k:k + n]].cuda()
                    proj = self.projs[j][perm[k:k + n]].cuda()
                    verts_mano = self.mano_vertices[j:j+1].cuda()
                                                            
                    sh_coeff = self.model.sh_coeffs[perm[k:k + n]]
                    ray =  self.rays[perm[k:k + n]].cuda()
                    cam_exs = self.cam_exs[j][k: k + 1].cuda()
                    cam_ins = self.cam_ins[j][k: k + 1].cuda()
                

                    data_input = pose, trans, scale, w2c, proj, mask, ray, sh_coeff
                    
                    with torch.no_grad():
                        render_imgs, mesh_imgs, normal_imgs, pred_imgs, albedo_imgs, vertices_new, pred_weights, pred_mask, pred_albedo, queries = (
                        self.model(data_input, self.glctx, self.resolution, is_train=False))
                    
                        # sr_img_xhand = self.implicit_net.forward_img( lr_img=lr_img, feat=torch.cat([pred_imgs, normal_imgs], dim=3), out_size=(self.w, self.h))
                    
                    src_dict = {
                        'img': lr_img.permute(0, 3, 1, 2),
                        'vertices': verts_mano,
                        'w2c': w2c,
                        'proj': proj,
                        'cam': {
                            'ex': cam_exs,
                            'in': cam_ins
                        }
                    }
                    sr_img, normal_img = self.implicit_net(src_dict, hr_size=(self.w, self.h), glctx=self.glctx)
                    loss = 0    
                    
                    des = 'Implicit update -- '
                    
                    # Color Loss
                    # color_loss = F.l1_loss(
                    #     (F.interpolate(lr_img.permute(0, 3, 1, 2), self.w // 4) * F.interpolate(mask.unsqueeze(1), self.w//4)), 
                    #     (F.interpolate(sr_img, self.w // 4) * F.interpolate(mask.unsqueeze(1), self.w//4))
                    # )
                    # color_loss = F.l1_loss(
                    #     (F.interpolate(lr_img.permute(0, 3, 1, 2), self.w).permute(0, 2, 3, 1) * mask.unsqueeze(-1)).mean(dim=(1, 2, 3)), 
                    #     sr_img.mean(dim=(1, 2, 3))
                    # )
                    # loss = loss + color_loss * 0.3
                    # des = des + 'color: %.4f ' % color_loss.item()
                    
                    # Fourier Freq Loss
                    # gt_freq = torch.fft.fft2(ori_sr_img, norm='ortho')
                    # pred_freq = torch.fft.fft2(sr_img, norm='ortho')
                    
                    # freq_loss = F.l1_loss(gt_freq, pred_freq)
                    # loss = loss + freq_loss * 0.5
                    # des = des + 'freq: %.4f ' % freq_loss.item()
                    
                    # sr_loss = F.l1_loss(sr_img_xhand, sr_img)
                    # loss = loss + sr_loss * 0.1
                    # des = des + 'sr: %.4f ' % sr_loss.item()
                    
                    
                    
                    # Wavelet Loss
                    
                    # The above code snippet is performing the following operations in Python:
                    xl, xh = dwt_2d(sr_img)
                    # lr_yl, lr_yh = dwt_2d(ori_sr_img[..., :3].permute(0, 3, 1, 2))
                    yl, yh = dwt_2d(ori_sr_img)
                    
                    freq_loss = F.l1_loss(xl, yl) * 0.2 + (F.l1_loss(xh[0].sum(0), yh[0].sum(0)) + F.l1_loss(xh[1].sum(0), yh[1].sum(0)) + F.l1_loss(xh[2].sum(0), yh[2].sum(0)))
                    loss = loss + freq_loss * self.wavelet_weight
                    des = des + 'freq: %.4f ' % freq_loss.item()
                    
                    # Total Variation Loss
                    # total_variation_loss = F.l1_loss(
                    #     tv_loss(ori_sr_img),
                    #     tv_loss(sr_img)
                    # )     
                    # loss = loss + total_variation_loss * 0.5
                    # des = des + 'tv: %.4f ' % total_variation_loss.item()
            
                    # Discriminator Loss
                    fake_pred = self.disc_net(sr_img.detach()).squeeze(1)
                    
                    disc_loss = bce_loss(fake_pred, 0.5 * (1 - mask) + torch.ones_like(fake_pred) * mask)
                    loss = loss + disc_loss * self.disc_weight
                    des = des + 'disc: %.4f ' % fake_pred[mask == 1].mean() 
            
            
                    # Query Mean Consistency Loss
                    mean_query, var_query = cal_query_mean_var(queries, sr_img)
                    
                    if self.pose_consist:
                        pose_diff = F.l1_loss(mean_query, (mean_query if prev_mu is None else prev_mu))
                    
                    pred_weights = 1 - pred_weights.max(2)[0]
                    board = torch.zeros(n, 1, self.h, self.w).cuda()
                    for b in range(n):
                        coords = (queries[b, queries[b, :, 2] == 1] * 255).long().clamp(0, 255)
                        board[b, 0, coords[:, 1], coords[:, 0]] = var_query[queries[b, :, 2] == 1] * 100
                        # board[b, 0, coords[:, 1], coords[:, 0]] += pred_weights[0, queries[b, :, 2] == 1] * 0.5
                    prev_mu = mean_query.detach()
                    unconfidence_map = board.repeat(1, 3, 1, 1).clamp(0, 1)
                    imgs = torch.cat([sr_img, unconfidence_map], dim=-1).permute(1, 0, 2, 3).reshape(3, n * self.h, -1)
                    
                    cv2.imwrite(join(self.out_mesh_dire, 'unconfidence', '%02d'  % (self.step), '%02d_%02d.png' % (j, k)), tensor2img(imgs, 0))
                    # import pdb; pdb.set_trace()
                    sr_loss = pose_diff * self.pose_weight + F.l1_loss(pred_imgs.permute(0, 3, 1, 2) * unconfidence_map, sr_img * unconfidence_map).mean() # * 0.8 + self.lpips_loss(pred_imgs.permute(0, 3, 1, 2) * unconfidence_map, sr_img * unconfidence_map).mean() * 0.2
                    loss = loss + sr_loss  * self.cons_weight
                    des = des + 'sr: %.4f ' % sr_loss.item() + 'pose_diff: %.4f' % pose_diff.item()
                
                    
                    # cons_loss = 0
                    # cons_loss = F.l1_loss(query_color, mean_sr_queries.unsqueeze(0).repeat(query_color.shape[0], 1, 1))
                    # loss = loss + cons_loss * 0.3
                    # des = des + 'cons: %.4f ' % cons_loss.item()
                    
                    sr_optimizer.zero_grad()
                    loss.backward()
                    sr_optimizer.step()
                    
                    pbar.set_description(des)

                    ac_query_color.append(var_query)

        mean_var = torch.mean(torch.stack(ac_query_color, dim=0).reshape(self.epoch_implicit, -1, 49281), dim=1) 
        co_dist = unconfidence_to_color(mean_var.detach().cpu().numpy())
        
        for i in [0, self.epoch_implicit-1]:
            save_obj_mesh_with_color(join(self.out_mesh_dire, 'unconfidence', '%03d_unconfidence_%02d.obj' % (self.step, i)), self.vertices.cpu().numpy(), self.faces.cpu().numpy(), co_dist[i])
            


    # sr_img_blur = self.implicit_net.forward_img( lr_img=lr_img, feat=vert_img.clone(), out_size=(self.w, self.h))
                
    # mean_queries = sr_queries.sum(0) / queries[..., 2:].sum(0)
    
    
    
    def xhand_train_step(self, optimizer, epoch, pbar, train_render=False, render_resolution=None):
        perm_frame = torch.randperm(self.num_frame)
        mano_anno = []
        cam_anno = []
        
        loss_dict = {
            'sfs_loss': [], 
            'lap_delta_loss' : [],
            'albedo_loss' : [],
            'edge_loss' : [], 
            'delta_loss': [], 
            'weight_loss': [],
            'albedo_loss': [],
            'render_loss': [],
            'geo_acc': [],
        }   
        
        render_resolution = render_resolution if render_resolution else self.resolution
        
        for j_perm in range(self.num_frame):
            j = perm_frame[j_perm]
            rendered_img = []
            proj_list = []
            w2c_list = []
            perm = torch.randperm(self.using_view)

            pose =  self.poses[j:j + 1].reshape(1, -1).cuda()
            shape = self.shapes[j:j + 1].cuda()
            trans = self.transs[j:j + 1].cuda()
            scale = self.scales[j:j + 1].cuda()
            template_vertices = self.template_vertices[j:j+1].cuda() if self.use_template else None
            
            mano_anno.append({
                'poses': np.round(np.asarray(pose.cpu().detach().numpy(), np.float64), 6).tolist(),
                'shape': np.round(np.asarray(shape.cpu().detach().numpy(), np.float64), 6).tolist(),
                'trans': np.round(np.asarray(trans.cpu().detach().numpy(), np.float64), 6).tolist(),
                'scale': scale.cpu().tolist()
            })
            
            for k in range(0, self.using_view, self.batch):
                n = min(self.using_view, k + self.batch) - k
                w2c =  self.w2cs[j][perm[k:k + n]].cuda()
                proj = self.projs[j][perm[k:k + n]].cuda()
                
                proj_list.append(proj)
                w2c_list.append(w2c)
                
                img =  self.imgs[j][perm[k:k + n]].cuda()
                lr_img = self.lr_imgs[j][perm[k:k + n]].cuda() 
                sr_img = self.sr_imgs[j][perm[k:k + n]].cuda()
                mask = self.masks[j][perm[k:k + n]].cuda()
                ray =  self.rays[perm[k:k + n]].cuda()
                sh_coeff = self.model.sh_coeffs[perm[k:k + n]]
                # valid_mask = valid_masks[perm[k:k+batch]]

                data_input = pose, trans, scale, w2c, proj, mask, ray, sh_coeff
                
                valid_idx, render_imgs, tmp_img, mean_img, normal_img, pred_delta, vertices_new, pred_weights, pred_albedo, pred_mask, queries = (self.model(data_input, self.glctx, render_resolution, is_train=True, train_render=train_render))
                
                valid_img = sr_img[valid_idx]
                sr_img[mask == 0] = 0
                
                sfs_loss = self.sfs_weight * ((F.l1_loss(tmp_img[valid_idx], valid_img)) * 0.8 + 0.2 * self.lpips_loss(tmp_img.permute(0, 3, 1, 2), sr_img.permute(0, 3, 1, 2)).mean())  
                
                geo_acc = 0
                # geo_acc = self.lpips_loss(mean_img.permute(0, 3, 1, 2).sum(axis=1, keepdim=True), sr_img.permute(0, 3, 1, 2).sum(axis=1, keepdim=True)).mean() * 0.3
                            
                geo_acc = self.lpips_loss(mean_img.permute(0, 3, 1, 2).sum(axis=1, keepdim=True), sr_img.permute(0, 3, 1, 2).sum(axis=1, keepdim=True)).mean()
                
                # _, xh = self.dwt2d(mean_img)
                # _, yh = self.dwt2d(sr_img)
                # for i in range(len(xh)):
                #     geo_acc = geo_acc + F.l1_loss(xh[i], yh[i]) * 0.1
                geo_acc = geo_acc * self.geo_weight
                            
                
                if train_render:  
                    render_loss = ((F.l1_loss(render_imgs[valid_idx], valid_img)) * 0.8 + 0.2 * self.lpips_loss(render_imgs.permute(0, 3, 1, 2), sr_img.permute(0, 3, 1, 2)).mean()) * self.sfs_weight
                else:
                    render_loss = torch.zeros_like(sfs_loss)

                if self.part_smooth:
                    lap_delta_loss = self.lap_weight * laplacian_smoothing(pred_delta[0], self.faces.long(), method="uniform", return_sum=False) / 10
                    lap_delta_loss = lap_delta_loss[lap_delta_loss > torch.quantile(lap_delta_loss, 0.25)].sum()
                    lap_vert_loss = torch.zeros_like(lap_delta_loss)
                else:
                    lap_delta_loss = self.lap_weight * laplacian_smoothing(pred_delta[0], self.faces.long(), method="uniform", return_sum=True) / 10
                    lap_vert_loss = self.lap_weight * laplacian_smoothing(vertices_new[0], self.faces.long(), method="uniform", return_sum=True) / 100

                lap_w = laplacian_smoothing(pred_weights.squeeze(0), self.faces.long(), method="uniform")

                albedo_loss = laplacian_smoothing(pred_albedo.squeeze(0), self.faces.long(), method="uniform", return_sum=False)
                
                albedo_loss = albedo_loss[albedo_loss > torch.quantile(albedo_loss, 0.25)].sum() + F.mse_loss(MEAN_HAND_ALBEDO, pred_albedo.mean(0)) * 0.2
                normal_loss = torch.zeros_like(albedo_loss)
                
                mask_loss = self.mask_weight * F.mse_loss(pred_mask, mask)
                weight_loss = 100 * F.mse_loss(pred_weights.squeeze(0), self.weights)
                a = vertices_new[0, self.faces[:, 0].long()]
                b = vertices_new[0, self.faces[:, 1].long()]
                c = vertices_new[0, self.faces[:, 2].long()]
                edge_length = torch.cat(
                    [((a - b) ** 2).sum(1), ((c - b) ** 2).sum(1), ((a - c) ** 2).sum(1)])

                edge_loss = torch.clip(edge_length - self.edge_length_mean , 0, 1).mean() * self.edge_weight
                delta_loss = F.relu((pred_delta[0] ** 2).sum(1).mean() - 0.0001) * self.delta_weight
                delta2_loss = F.l1_loss(pred_delta[0], self.delta)

                loss_dict['sfs_loss'].append(sfs_loss.item())
                loss_dict['lap_delta_loss'].append(lap_delta_loss.item())
                loss_dict['albedo_loss'].append(albedo_loss.item())
                loss_dict['edge_loss'].append(edge_loss.item())
                loss_dict['delta_loss'].append(delta_loss.item())
                loss_dict['weight_loss'].append(weight_loss.item())
                loss_dict['albedo_loss'].append(albedo_loss.item())
                loss_dict['render_loss'].append(render_loss.item())
                loss_dict['geo_acc'].append(geo_acc.item())
                
                # fig =plt.figure()
                # ax = fig.add_subplot(projection='3d')
                # np_t = template_vertices[0].cpu().numpy()
                # np_v = vertices_new[0].detach().cpu().numpy()
                # plt.scatter(np_t[:, 0], np_t[: , 1], np_t[:, 2])
                # plt.scatter(np_v[:, 0], np_v[: , 1], np_v[:, 2])

                
                                
                cd_loss = torch.tensor(0).cuda()
                if self.use_template:
                    dist1, dist2,_, _ = self.chamfer_distance(template_vertices, vertices[None,]) 
                    cd_loss = (torch.mean(dist1)) + (torch.mean(dist2))  * 100
                                
                loss = sfs_loss + lap_delta_loss + lap_vert_loss + mask_loss + normal_loss + delta_loss + delta2_loss + edge_loss + lap_w + weight_loss + albedo_loss + render_loss + geo_acc + cd_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                des = 's:%.3f' % sfs_loss.item() + ' l:%.3f' % lap_delta_loss.item() + ' a:%.3f' % albedo_loss.item() + ' m:%.3f' % mask_loss.item() + ' n:%.3f' % normal_loss.item() + ' e:%.3f' % edge_loss.item() + ' d:%.3f,%.3f' % (delta_loss.item(), delta2_loss.item()) + ' w:%.3f,%.3f' % (weight_loss.item(), lap_w.item()) + ' r:%.3f' % render_loss.item() + ' c:%.3f' % cd_loss.item()
                
                pbar.set_description(des)
                
                if epoch % 100 == 0:
                    rendered_img.append(torch.cat([tmp_img, render_imgs, sr_img, img], 2))
                    perm_last = perm.cpu().numpy()
            
            cam_anno.append({
                'projs': torch.cat(proj_list, 0).cpu().tolist(),
                'w2cs' : torch.cat(w2c_list, 0).cpu().tolist()
            })
            
            if epoch % 100 == 0:
                rendered_img = torch.cat(rendered_img, 0)
                
                os.makedirs(join(self.out_mesh_dire, 'objs'), exist_ok=True)
                os.makedirs(join(self.out_mesh_dire, 'images'), exist_ok=True)
                os.makedirs(join(self.out_mesh_dire, 'result_%d' % epoch), exist_ok=True)
                
                for i_, idx in enumerate(perm_last):
                    cv2.imwrite(join(self.out_mesh_dire, 'result_%d' % epoch, 'train_%02d_%02d.png' % (j.item(), idx)), tensor2img(rendered_img, i_))
                
                
                save_obj_mesh_with_color(join(self.out_mesh_dire, 'objs', '%d.obj' % (j.item())),
                                        vertices_new[0].detach().cpu().numpy(),
                                        self.np_faces, (pred_albedo.detach().cpu().numpy()[0])[:, 2::-1])
                save_obj_mesh_with_color(join(self.out_mesh_dire, 'result_%d' % epoch, 'seq_%d.obj' % (j.item())), vertices_new[0].detach().cpu().numpy(), self.np_faces, (pred_albedo.detach().cpu().numpy()[0])[:, 2::-1])
            
            if epoch % 100 == 0:
                with open(join(self.out_mesh_dire, 'mano_anno.json'), 'w') as f:
                    json.dump(mano_anno, f)
                with open(join(self.out_mesh_dire, 'cam_anno.json'), 'w') as f:
                    json.dump(cam_anno, f)
        
        
        for keys, vals in loss_dict.items():
            loss_dict[keys] = np.array(vals).mean()
        
        vis_dict = {
            'img': img,
            'sr': sr_img.clamp(0, 1),
            'pred': torch.cat([tmp_img, render_imgs], dim=2),
        }
        
        return vis_dict, loss_dict      
        
    def train(self, ):
        # start frame by frame
        self.albedo.requires_grad_(False)
        self.poses.requires_grad_(False)
        self.delta.requires_grad_(False)
        self.sh_coeffs.requires_grad_(False)
        self.delta = self.delta.clone().detach()

        self.model = MVIHand(
            self.vertices_tmp, 
            self.faces, 
            self.delta, 
            self.albedo[0], 
            self.weights, 
            self.ori_v, 
            self.sh_coeffs,
            delta_scale=self.delta_scale,
            hand_type=self.hand_type, 
            render_nettype=self.net_type, 
            use_pe=self.use_pe, 
            latent_num=self.latent_num, 
            use_x_pos=self.use_x_pos, 
            use_ray=self.use_ray, 
            use_emb=self.use_emb, 
            wo_latent=self.wo_latent,
            mlp_use_pose=self.mlp_use_pose, 
            use_rotpose=self.use_rotpose, 
            use_sum=self.use_sum, 
            mano_layer=self.mano_layer,
            use_cond=True if self.use_enh else False     # False basic
        )

        self.model = self.model.cuda()
                    
        optimizer = Adam([{'params': self.model.parameters(), 'lr': 0.001}])
        if self.use_liif and self.use_consist:
            sr_optimizer = Adam([{'params': self.implicit_net.imnet.parameters(), 'lr': 0.001}])
        start_epoch = 1
        train_render = False
        if self.is_continue:
            checkpoint_state_dict = torch.load(self.model_path)
            start_epoch = checkpoint_state_dict['epoch']
            state_dict = checkpoint_state_dict['model'].state_dict()
            
            if start_epoch > self.epoch_train // 2 - 1:
                train_render = True
            
            if train_render:       
                for key in list(state_dict.keys()):
                    if 'renderer.' in key or 'render' in key:
                        del state_dict[key]
            self.model.load_state_dict(state_dict, strict=False)
                        
            print('continue from ' + self.model_path + ' epoch : %d' % start_epoch)
            
            if self.optimizer_path and os.path.exists(self.optimizer_path):
                optimizer.load_state_dict(torch.load(self.optimizer_path).state_dict())
            if self.use_liif and self.use_consist:                   
                if self.implicit_optimizer_path and os.path.exists(self.implicit_optimizer_path):
                    sr_optimizer.load_state_dict(torch.load(self.implicit_optimizer_path).state_dict())
        
              
        pbar = tqdm(range(start_epoch, self.epoch_train + 1))
        render_resolution = [128, 256, 384]
        for i in pbar:
            
            if i % 200 == 0:
                torch.save({
                    'epoch': i,
                    'model': self.model
                }, join(self.out_mesh_dire, 'model.pth'))
                torch.save(optimizer, join(self.out_mesh_dire, 'optimizer.pth'))
                if self.use_liif and self.use_consist:
                    torch.save(self.implicit_net, join(self.out_mesh_dire, 'implicit_model.pth'))
                    torch.save(sr_optimizer, join(self.out_mesh_dire, 'implicit_optimizer.pth'))
            
            if i >= self.epoch_train // 4:
                self.model.sh_coeffs.requires_grad_(True)
                if i >= self.epoch_train // 2:
                    train_render = True
                    if self.use_enh:
                        self.lap_weight = 9000
                        self.geo_weight = 1
                    if self.use_consist:
                        # self.mask_weight = 70
                        None
                    if i % 50 == 1:
                        self.update_mean()
            
        
            if self.use_consist and i > (self.epoch_train // 2)-1 and i % self.implicit_update_term == 0 and i < self.epoch_train - 100:
            #if self.use_consist and i % self.implicit_update_term == 1:
                self.implicit_update(
                    sr_optimizer,
                    epoch=i,
                )
                self.implicit_net.eval()
                self.update_sr()
            
            vis_dict, loss_dict = self.xhand_train_step(
                optimizer=optimizer, 
                epoch=i, 
                pbar=pbar,
                train_render=train_render
            )
                
            self.step += 1
                
            if self.step % self.log_step == 0:
                loss_dict['gt'] =   wandb.Image(to_pil_image( vis_dict['img'][0].permute(2, 0, 1).flip(0)))
                loss_dict['sr'] =   wandb.Image(to_pil_image( vis_dict['sr'][0].permute(2, 0, 1).flip(0)))
                loss_dict['pred_img'] = wandb.Image(to_pil_image(vis_dict['pred'][0].permute(2, 0, 1).flip(0)))
                # loss_dict['implicit_img'] = wandb.Image(to_pil_image(vis_dict['implicit_img'][0].clamp(0, 1).permute(2, 0, 1).flip(0)))
                if self.use_wandb:
                    wandb.log(loss_dict, step=self.step)
            
        # save parameter
        torch.save({
            'epoch': i,
            'model': self.model
        }, join(self.out_mesh_dire, 'model.pth'))
        torch.save(optimizer, join(self.out_mesh_dire, 'optimizer.pth'))
        if self.use_liif and self.use_consist:
            torch.save(self.implicit_net, join(self.out_mesh_dire, 'implicit_model.pth'))
            torch.save(sr_optimizer, join(self.out_mesh_dire, 'implicit_optimizer.pth'))
        print('Finished {}'.format(self.exp_name))
        
    @torch.no_grad()
    def eval(self, xhand_path=None, implicit_path=None, save_vis=True, save_mesh=False,):
        
        if self.use_liif and implicit_path:
            state_dict = torch.load(join(implicit_path)).state_dict()
            print('Loading Opimized LIIF from ', implicit_path)
            self.implicit_net.load_state_dict(state_dict, strict=False)

        if xhand_path is None:
            xhand_path = os.path.join(self.out_mesh_dire, 'model.pth')
        
        if xhand_path is None:
            print('Please input model path')
        out_dire = self.out_dire
        if xhand_path:
            out_dire = '/'.join(xhand_path.split('/')[:-1])
        
        if save_vis:
            os.makedirs(out_dire + '/outs', exist_ok=True)
        if save_mesh:
            os.makedirs(out_dire + '/mesh_outs', exist_ok=True)
        print('Loading from ', xhand_path)
        xhand_pth = torch.load(xhand_path)['model']
        
        self.model = MVIHand(
            xhand_pth.verts, 
            xhand_pth.faces, 
            xhand_pth.delta_net.x_mean, 
            xhand_pth.color_net.x_mean, 
            xhand_pth.lbs_net.x_mean,
            xhand_pth.template_v, 
            xhand_pth.sh_coeffs, 
            delta_scale=self.delta_scale,
            latent_num=self.latent_num,
            hand_type=xhand_pth.hand_type, 
            render_nettype=self.net_type, 
            use_pe=self.use_pe, 
            use_x_pos=self.use_x_pos, 
            use_ray=self.use_ray, 
            use_emb=self.use_emb, 
            wo_latent=self.wo_latent, 
            mlp_use_pose=self.mlp_use_pose, 
            use_rotpose=self.use_rotpose, 
            use_cond=True if self.use_enh else False ,
        )
        sh_coeffs = self.model.sh_coeffs
        if xhand_path:
            self.model.load_state_dict(xhand_pth.state_dict(), strict=False)
        self.model = self.model.cuda().eval()
        
        loss_fn_alex = lpips.LPIPS(net='alex', version=0.1).cuda()
        
        gt_tq = tensorQueue2(8)
        sr_tq = tensorQueue2(8)
        
        dwt_2d = wavelet.DWTForward(J=3, wave='db1', mode='zero').cuda()
        
        infer_speed = []
        output_imgs = []
        total_psnr = []
        total_sr_psnr = []
        total_ssim = []
        total_sr_ssim = []
        total_lpips = []
        total_sr_lpips = []
        total_lw= []
        total_hw= []
        total_variation = []
        
        cam_names = ih_cam_names if self.type == 'interhand' else go_cam_names
        unseen_sh_coeff = sh_coeffs.mean(0, keepdim=True)

        for j in range(self.num_frame):
            pose =  self.poses[j:j + 1].reshape(1, -1).cuda()
            shape = self.shapes[j:j + 1].cuda()
            trans = self.transs[j:j + 1].cuda()
            scale = self.scales[j:j + 1].cuda()
            
            for k in range(0, self.num_view):
                w2c = self.w2cs[j][k:k + 1].cuda()
                proj= self.projs[j][k:k + 1].cuda()
                img = self.imgs[j][k:k + 1].cuda()
                mask = self.masks[j][k:k + 1].cuda()
                ray = self.rays[k:k + 1].cuda()
                cam_id = self.cam_id[k]
                sh_id = cam_names.index(cam_id) if cam_id in cam_names else go_cam_names.index(cam_id)
                sh_coeff = sh_coeffs[sh_id:sh_id+1]
                
                # import pdb; pdb.set_trace()
                verts_mano = self.mano_vertices[j:j+1].cuda()
                
                cam_exs = self.cam_exs[j][k: k + 1].cuda()
                cam_ins = self.cam_ins[j][k: k + 1].cuda()

                data_input = pose, trans, scale, w2c, proj, None, ray, sh_coeff

                start_time = time.time()
                render_imgs, mesh_imgs, normal_img, pred_imgs, albedo_imgs, vertices_new, pred_weights, pred_mask, pred_albedo, queries = (
                    self.model(data_input, self.glctx, self.resolution, is_train=False))
                render_imgs = pred_imgs.clone()
                if j > 0:
                    infer_speed.append(time.time() - start_time)
                
                pred_mask = pred_mask > 0.5
                
                sr_comp_img = img.clone()
                sr_comp_img[mask == 0] = 0
                
                # pred_mask = pred_mask * mask
                render_imgs[pred_mask==0] = 0
                img[pred_mask == 0] = 0
                
                lr_img = F.interpolate(sr_comp_img.permute(0, 3, 1, 2), self.lr_size).permute(0, 2, 3, 1)
                if self.use_liif:
                    src_dict = {
                        'img': lr_img.permute(0, 3, 1, 2),
                        'vertices': verts_mano,
                        'w2c': w2c,
                        'proj': proj,
                        'cam': {
                            'ex': cam_exs,
                            'in': cam_ins
                        }
                    }
                    mano_img = torch.zeros_like(img).cuda()
                    if self.liif_use_3d:
                        # mano_img, _, mano_mask = self.implicit_net.render_hand(verts_mano, mano_verts_color.unsqueeze(0), w2c, proj, self.glctx, self.mano_faces, out_size=(self.w, self.h))
                        # sr_img = self.implicit_net.forward_img(lr_img, mano_img.permute(0, 3, 1, 2), out_size=(self.w, self.h))
                        sr_img, _ = self.implicit_net(src_dict, self.glctx, hr_size=(self.w, self.h))
                        # normal_img = normal_img.permute(0, 2,  3, 1)
                    else:
                        sr_img, _ = self.implicit_net(src_dict, hr_size=(self.w, self.h))
                    
                    sr_img[0, :, mask[0] == 0] = 0
                    
                else:
                    sr_img =F.interpolate(lr_img.permute(0, 3, 1, 2), self.h)
                    mano_img = torch.zeros_like(img).cuda()
                sr_img = sr_img.permute(0, 2, 3, 1)
                
                gt_img_t, queries_t = gt_tq.next(img.permute(0, 3, 1, 2), queries)
                mean_query, var_query = cal_query_mean_var(queries_t, gt_img_t)
                board = torch.zeros(1, 1, self.h, self.w).cuda()       
                coords = (queries[0, queries[0, :, 2] == 1] * 255).long().clamp(0, 255)
                board[0, 0, coords[:, 1], coords[:, 0]] = var_query[queries[0, :, 2] == 1] 
                gt_unconfidence_map = board.repeat(1, 3, 1, 1).permute(0, 2, 3, 1)
                
                sr_img_t, queries_t = sr_tq.next(sr_img.permute(0, 3, 1, 2), queries)
                mean_query, var_query = cal_query_mean_var(queries_t, sr_img_t)
                board = torch.zeros(1, 1, self.h, self.w).cuda()       
                coords = (queries[0, queries[0, :, 2] == 1] * 255).long().clamp(0, 255)
                vis_query_sum = 0
                
                board[0, 0, coords[:, 1], coords[:, 0]] = var_query[queries[0, :, 2] == 1] * mask[0, coords[:, 1], coords[:, 0]]
                vis_query_sum = mask[0, coords[:, 1], coords[:, 0]].sum()
                unconfidence_map = board.repeat(1, 3, 1, 1).permute(0, 2, 3, 1)
                unconfidence_map[mask == 0] = 0 
                
                
                outs = tensor2img(torch.cat([sr_comp_img, F.interpolate(lr_img.permute(0, 3, 1, 2), size=self.h, mode='bicubic').permute(0, 2, 3, 1), sr_img, mesh_imgs, normal_img, albedo_imgs, mano_img[..., :3], pred_imgs, gt_unconfidence_map * 20, unconfidence_map * 20], 2))
                width = outs.shape[1]
                img_num = width / img.shape[2]

                if save_vis:
                    output_imgs.append(outs)

                sr_psnr = calculate_psnr(sr_comp_img[0].detach().cpu().numpy(), sr_img[0].detach().cpu().numpy())  # PSNR(img[0].detach().cpu().numpy(), render_imgs[0].detach().cpu().numpy())
                sr_ssim = SSIM(sr_comp_img[0].detach().cpu().numpy(), sr_img[0].detach().cpu().numpy(), channel_axis=2, data_range=1, multichannel=True)
                sr_lpips_loss = loss_fn_alex(sr_img.permute(0, 3, 1, 2), sr_comp_img.permute(0, 3, 1, 2))
                
                psnr = calculate_psnr(img[0].detach().cpu().numpy(), render_imgs[0].detach().cpu().numpy()) # , mask[0].detach().cpu().numpy())
                # from torchmetrics import PeakSignalNoiseRatio

                # import pdb; pdb.set_trace()
                # calculate_psnr(img[0].detach().cpu().numpy(), render_imgs[0].detach().cpu().numpy(), mask[0].detach().cpu().numpy())  # PSNR(img[0].detach().cpu().numpy(), render_imgs[0].detach().cpu().numpy())
                ssim = SSIM(img[0].detach().cpu().numpy(), render_imgs[0].detach().cpu().numpy(), channel_axis=2, data_range=1, multichannel=True)
                lpips_loss = loss_fn_alex(render_imgs.permute(0, 3, 1, 2), img.permute(0, 3, 1, 2))
                    
                xl, xh = self.dwt2d(render_imgs)
                yl, yh = self.dwt2d(img)
                    
                l_w = F.l1_loss(xl, yl)
                s_w = (F.l1_loss(xh[0], yh[0]) + F.l1_loss(xh[1], yh[1]) + F.l1_loss(xh[2], yh[2]))
                

                variation = (unconfidence_map.sum() / vis_query_sum ) / 3
                
                total_sr_psnr.append( sr_psnr)
                total_sr_ssim.append( sr_ssim)
                total_sr_lpips.append(sr_lpips_loss.item())
                
                total_psnr.append(psnr)
                total_ssim.append(ssim)
                total_lpips.append(lpips_loss.item())
                
                total_variation.append(variation.detach().cpu().numpy())
                total_lw.append(l_w.item()) 
                total_hw.append(s_w.item())
                if save_vis:
                    img_name = f"{self.img_names[j]:06d}" if isinstance(self.img_names[j], int) else self.img_names[j][:10]
                    cv2.imwrite(out_dire + '/outs' + '/%s_%s.png' % (str(self.cam_id[:10]), img_name), outs)
                    
                if save_mesh:
                    save_obj_mesh_with_color(out_dire + '/mesh_outs' + '/%s.obj' % (self.img_names[j][:-4]), vertices_new.cpu().numpy()[0], self.model.faces.cpu().numpy(), (pred_albedo.cpu().numpy()[0])[:, 2::-1])
            

        print('inference fps:', 1 / np.mean(infer_speed))
        print('PSNR:', np.mean(total_psnr), "SSIM:", np.mean(total_ssim), "LPIPS:", np.mean(total_lpips), "Variation:", np.mean(total_variation), "LW:", np.mean(total_lw), "HW:", np.mean(total_hw))
        print('SR_PSNR:', np.mean(total_sr_psnr), "SR_SSIM:", np.mean(total_sr_ssim), "LPIPS:", np.mean(total_sr_lpips))
        
        if save_vis:
            convert2video(out_dire + '/%s_%s_%s' % (self.capture_name, self.data_name, cam_id[:10]), output_imgs, fps=8, ext=['mp4', 'gif'])