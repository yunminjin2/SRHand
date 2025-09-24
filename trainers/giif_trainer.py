import os

from os.path import join
from glob import glob
from tqdm import tqdm

from pyhocon import ConfigFactory
import numpy as np
import cv2
import lpips

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.transforms import GaussianBlur
from torchvision.transforms.functional import to_pil_image
import nvdiffrast.torch as dr
from models.utils import tensor2img
from dataset import GiifTrainDataset
from models.giif import *
from models.GAN import UNetDiscriminatorSN
import random
from models.utils import calculate_psnr, draw_joint

import wandb

MEAN_HAND_ALBEDO = torch.tensor([0.31996773, 0.36127372, 0.44126652]).cuda()


class GiifTrainer:
    def __init__(self, conf, mano_layer):
        self.type = conf.get_string('data_type')
        
        self.drop_cam = conf.get_string('drop_cam').split(',')
        self.cam_id = conf.get_string('cam_id')
        self.num_view = conf.get_int('num_view')
        self.num_frame = conf.get_int('num_frame')
        self.num = conf.get_int('num')
        self.adjust = conf.get_bool('adjust')
        self.w = conf.get_int('w')
        self.h = conf.get_int('h')
        self.lr_size = conf.get_int('lr_size')
        self.use_x_pos = conf.get_bool('use_x_pos')
        self.use_ray = conf.get_bool('use_ray')
        self.use_emb = conf.get_bool('use_emb')
        self.mlp_use_pose = conf.get_bool('mlp_use_pose')
        self.use_rotpose = conf.get_bool('use_rotpose')
        self.resolution = (self.h, self.w)
        self.batch = conf.get_int('batch')
        self.lr = conf.get_float('lr')
        self.use_blur = conf.get_bool('use_blur')
        if self.use_blur:
            self.blurer = GaussianBlur(3)
        self.use_noise = conf.get_bool('use_noise')
        
        self.giif_continue = conf.get_bool('giif_continue')
        self.giif_epoch = conf.get_int('giif_epoch')
        self.z_channels = conf.get_int('z_channels')
        self.ch_mult = conf.get_list('ch_mult')
        self.use_js = conf.get_bool('use_js')
        self.use_joint = conf.get_bool('use_joint', False)
        self.use_normals = conf.get_bool('use_normals', False)
        self.use_depths = conf.get_bool('use_depths', False)
        self.use_rays = conf.get_bool('giif_use_rays', False)
        self.use_rdn = conf.get_bool('use_rdn', False)
        self.use_3d = conf.get_bool('use_3d', False)
        self.use_gan = conf.get_bool('use_gan', False)
        self.use_mano = conf.get_bool('use_mano', False)
        self.use_multiscale = conf.get_bool('use_multiscale', False)
        
        self.use_wandb = conf.get_bool('use_wandb', False)
        self.log_step = conf.get_int('log_step', 10)
        
        self.mano_layer = {}
        self.mano_layer['right'] = mano_layer['right'].cuda()
        self.mano_layer['left'] = mano_layer['left'].cuda()
        
        self.dataset = {}
        self.dataloader = {}

        self.glctx = dr.RasterizeCudaContext()
        if self.use_wandb:
            wandb.init(project='XHand', name=conf.get_string('exp_name'), job_type='train')
            wandb.run.log_code(root='/workspace/datasets/XHand/', include_fn=lambda p: any(p.endswith(ext) for ext in ('.py', '.json', '.yaml', '.md', '.txt.', '.gin')), exclude_fn=lambda p: any(s in p for s in ('output', 'tmp', 'wandb', '.git', '.vscode')))
    
    def finish(self, ):
        if self.use_wandb:
            wandb.finish()
        time.sleep(3)
        
    def prepare_data(self,data_path, data_name='interhand', split='test'):
        self.dataset[split] = GiifTrainDataset(
            data_path=data_path,
            res=(self.w, self.h), 
            drop_cam=self.drop_cam, 
            split=split, 
            return_ray=True, 
            cam_id=self.cam_id, 
            test_num=self.num_view, 
            num_frame=self.num_frame, 
            adjust=self.adjust,
            limit=None,
            subject_limit=15
        )

        self.mano_faces = torch.from_numpy(self.mano_layer['right'].faces.astype(np.int32)).cuda()
        self.dataloader[split] = DataLoader(self.dataset[split], batch_size=self.batch, shuffle=True if split=='train' else False)
    
        print(f"Split : {split}, total length: {len(self.dataloader[split])}" )
        
    def initialize_model(self, pretrain_path=None):
        self.model = GIIF(
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
            use_normal = self.use_normals,
            use_depth = self.use_depths,
        ).cuda()
    
                
        if self.use_gan:
            self.disc = UNetDiscriminatorSN(
            num_in_ch = 3,
            num_feat=64,
            skip_connection=True    
        ).cuda()
        if pretrain_path:
            state_dict = torch.load(pretrain_path).state_dict()
            print('Loading Pretrained GIIF')
            self.model.load_state_dict(state_dict, strict=False)
        
    def set_log_path(self, out_dire, out_mesh_dire):
        self.out_dire = out_dire
        self.out_mesh_dire = out_mesh_dire
       
    def pretrain(self, giif_model=None):
        gan_loss = nn.BCELoss()
        lpips_loss = lpips.LPIPS(net='vgg').cuda()
        
        giif_optimizer = Adam([{'params': self.model.parameters(), 'lr': 0.0001}])
        if self.use_gan:
            disc_optimizer = Adam([{'params': self.disc.parameters(), 'lr': 0.001}])
            
        scheduler = torch.optim.lr_scheduler.LambdaLR(giif_optimizer, lr_lambda=lambda x: max(1e-3, 10**(-x*0.0002)))
        os.makedirs(join(self.out_mesh_dire, 'giif_images'), exist_ok=True)
        
        use_gan = False 
        for epoch in range(self.giif_epoch):
            # if epoch > self.giif_epoch // 2 and self.use_gan:
            #     use_gan = True
            pbar = tqdm(enumerate(self.dataloader['train']), total=len(self.dataloader['train']))
            psnrs = []
            save_imgs = []
            up_imgs = []
            for i, data in pbar:

                src_ori_img =   data['img']
                src_img_path =  data['img_path']
                src_mask =      data['masks']
                src_joints25d = data['joints25d']
                src_joints_img =data['joint_img']
                src_pose =      data['mano_pose']
                src_shape =     data['shape']
                src_trans =     data['trans']
                src_w2c =       data['w2c']
                src_proj =      data['proj']
                src_rays =      data['ray_directions']
                src_verts =     data['vertices']
                src_normals =   data['normals']
                
                src_queries =   data['queries']
                src_cam =       data['cam']

                src_img = src_ori_img * src_mask

                lr_size = self.lr_size
                if self.use_multiscale:
                    lr_size = random.choice([self.lr_size // 2, self.lr_size, self.lr_size * 2])
                lr_img = F.interpolate(src_img, lr_size)
                
                if self.use_blur:
                    lr_img = self.blurer(lr_img)
                if self.use_noise:
                    noise = torch.randn_like(lr_img) * 0.01
                    lr_img = lr_img + noise
               
                src_dict = {
                    'img': lr_img,
                    'pose': src_pose,
                    'joints25d': src_joints25d,
                    'joint_img': src_joints_img,
                    'w2c': src_w2c,
                    'rays': src_rays,
                    'proj': src_proj,
                    'vertices': src_verts,
                    'normals': src_normals,
                    'queries': src_queries,
                    'cam' : src_cam
                }
                
                self.model.train()
                
                out, render_img = self.model(src_dict, hr_size=(self.w, self.h), glctx=self.glctx)
                
                tot_loss = 0
                loss_l1 = F.l1_loss(out, src_img) + lpips_loss(src_img.sum(axis=1, keepdim=True), out.sum(axis=1, keepdim=True)).mean()
                tot_loss = tot_loss + loss_l1
                if use_gan:
                    self.disc.eval()
                    disc_loss = 0
                    fake_pred = self.disc(out)
                    disc_loss = gan_loss(fake_pred, (torch.ones_like(fake_pred) / 2) * (1 - src_mask) + torch.ones_like(fake_pred) * src_mask)
                    tot_loss = tot_loss + disc_loss
                            
                giif_optimizer.zero_grad()
                tot_loss.backward()
                giif_optimizer.step()
                scheduler.step()
                
                psnr = calculate_psnr(src_img[0].permute(1, 2, 0).detach().cpu().numpy(), np.clip(out[0].permute(1, 2, 0).detach().cpu().numpy(), 0, 1), src_mask[0].detach().cpu().numpy()) 
                psnrs.append(psnr)
                
                des = 'Epoch:%02d' % epoch + 'l1:%.3f' % loss_l1.item() + ' psnr:%.3f' % psnr
                
                real_loss, fake_loss = torch.tensor(0), torch.tensor(0)
                if use_gan:
                    self.disc.train()
                    self.model.eval()
                    
                    disc_loss = 0
                    real_loss = 0
                    fake_loss = 0

                    B = out.shape[0]
                    
                    disc_optimizer.zero_grad()
                    fake_pred = self.disc(out.detach())
                    real_pred = self.disc(src_img)
                    
                    real_loss = gan_loss(real_pred, (torch.ones_like(real_pred) / 2) * (1 - src_mask) + torch.ones_like(real_pred) * src_mask)
                    real_loss.backward()

                    fake_loss = gan_loss(fake_pred, (torch.ones_like(fake_pred) / 2) * (1 - src_mask))
                    fake_loss.backward()
                    
                    disc_optimizer.step()
                    
                    out_real_pred = torch.mean((real_pred[0, src_mask[0] == 1]).detach())
                    out_fake_pred = torch.mean((fake_pred[0, src_mask[0] == 1]).detach())
                
                    des += ' disc_loss: R : %.3f F : %.3f' % (real_loss.mean().item(), fake_loss.mean().item())+ 'out_pred : R : %.3f, F : %.3f' %(out_real_pred, out_fake_pred)
                
                pbar.set_description(des)
                
                if i % 200 == 0:
                    board = torch.zeros_like(src_ori_img)
                    # pair_25d_img = []
                    # for b in range(src_ori_img.shape[0]):
                    #     pair_25d_img.append(draw_joint(pair_joints25d[b], board[0]))
                    # pair_25d_img = torch.stack(pair_25d_img)
                        
                    lr_img = F.interpolate(lr_img, src_img.shape[2])
                    src_cond_img = []
                    if self.use_joint:
                        for b in range(src_ori_img.shape[0]):
                            src_cond_img.append(draw_joint(src_joints25d[b], board[0]))
                        src_cond_img = torch.stack(src_cond_img)
                    elif self.use_normals:                        
                        for b in range(src_ori_img.shape[0]):
                            src_cond_img.append(board[0] + (0.5 + (src_normals[b]) / 2)* src_mask[b])
                        src_cond_img = torch.stack(src_cond_img)
                    else: 
                        src_cond_img = torch.zeros_like(src_ori_img)
                    
                        
                    lr_img = F.interpolate(lr_img, src_img.shape[2])
                    save_img = torch.cat([src_img, lr_img, out], 3).clamp(0, 1)
                    if self.use_3d:
                        save_img = torch.cat([save_img, render_img[:, :3]], 3).clamp(0, 1)
                    # save_img = torch.cat([pair_img, pair_25d_img, src_ori_img, src_25d_img, lr_img, out], 3)
                    for b in range(save_img.shape[0]):
                        cv2.imwrite(join(self.out_mesh_dire, 'giif_images', '%02d_%02d.png' % (epoch, b)), tensor2img(save_img, b))

                    if i % self.log_step == 0 or i == self.giif_epoch -1:
                        self.step = epoch * len(self.dataloader['train']) + i
                        log_dict = {
                            'disc_loss': (real_loss + fake_loss).item(),
                            'real_loss': real_loss.item(),
                            'fake_loss': fake_loss.item(),
                        }

                        if self.use_wandb:
                            log_dict['render_img'] =   wandb.Image(to_pil_image(save_img[b].flip(0)))
                            wandb.log(log_dict, step=self.step)
            
            val_psnr, val_lpips = self.eval()
            log_dict = {
                'val_psnr': val_psnr,
                'val_lpips': val_lpips,
            }
            if self.use_wandb:
                wandb.log(log_dict, step=self.step)

            torch.save(self.model, join(self.out_mesh_dire, 'giif.pth'))
            if self.use_gan:
                torch.save(self.disc, join(self.out_mesh_dire, 'disc.pth'))
            
            print('\nPSNR: %.3f'%np.array(psnrs).mean())
            
        print("Finished Disc Pretraining")
        torch.save(self.model, join(self.out_mesh_dire, 'giif.pth'))
        if self.use_gan:
            print("Finished giif Pretraining")
            torch.save(self.disc, join(self.out_mesh_dire, 'disc.pth'))
        
    @torch.no_grad()
    def eval(self, ):
        lpips_loss = lpips.LPIPS(net='vgg').cuda()
        self.model.eval()
        
        os.makedirs(join(self.out_mesh_dire, 'eval_images'), exist_ok=True)
        
        pbar = tqdm(enumerate(self.dataloader['eval']), total=len(self.dataloader['eval']))
        psnrs = []
        lpipss = []
        save_imgs = []
        up_imgs = []
        for i, data in pbar:
            
            src_ori_img =   data['img']
            src_img_path =  data['img_path']
            src_mask =      data['masks']
            src_joints25d = data['joints25d']
            src_joints_img =data['joint_img']
            src_pose =      data['mano_pose']
            src_shape =     data['shape']
            src_trans =     data['trans']
            src_w2c =       data['w2c']
            src_proj =      data['proj']
            src_rays =      data['ray_directions']
            src_verts =     data['vertices']
            src_normals =   data['normals']
            src_queries =   data['queries']
            src_cam =       data['cam']
                        
            src_img = src_ori_img * src_mask

            gt_img = F.interpolate(src_img, self.w)
            
            # lr_img = src_img
            lr_img = F.interpolate(src_img, self.lr_size)
            if self.use_blur:
                lr_img = self.blurer(lr_img)
            if self.use_noise:
                noise = torch.randn_like(lr_img) * 0.01
                lr_img = lr_img + noise
                
            src_dict = {
                'img': lr_img,
                'pose': src_pose,
                'joints25d': src_joints25d,
                'joint_img': src_joints_img,
                'w2c': src_w2c,
                'rays': src_rays,
                'proj': src_proj,
                'vertices': src_verts,
                'normals': src_normals,
                'queries': src_queries,
                'cam' : src_cam
            }
                        
            out, render_img = self.model(src_dict, self.glctx, hr_size=(self.w, self.h)) 
            # out = out * src_mask
            if not self.use_3d:
                render_img = out
            psnr = calculate_psnr(gt_img[0].permute(1, 2, 0).detach().cpu().numpy(), np.clip(out[0].permute(1, 2, 0).detach().cpu().numpy(), 0, 1), src_mask[0].detach().cpu().numpy()) 
                        
            psnrs.append(psnr)
            lpipss.append(lpips_loss(gt_img.sum(axis=1, keepdim=True), out.sum(axis=1, keepdim=True)).mean().cpu())

            des = 'PSNR:%.3f' % (psnr)
            pbar.set_description(des)
            
            if i % 10 == 0: 
                board = torch.zeros_like(src_ori_img)
                lr_img = F.interpolate(lr_img, size=src_ori_img.shape[-1], mode='bicubic')
                src_cond_img = []
                if self.use_joint:
                    for b in range(src_ori_img.shape[0]):
                        src_cond_img.append(draw_joint(src_joints25d[b], board[0]))
                    src_cond_img = torch.stack(src_cond_img)
                elif self.use_normals:                        
                    for b in range(src_ori_img.shape[0]):
                        src_cond_img.append(board[0] + (0.5 + (src_normals[b]) / 2)* src_mask[b])
                    src_cond_img = torch.stack(src_cond_img)
                else: 
                    src_cond_img = torch.zeros_like(src_ori_img)
                        
                save_img = torch.cat([src_ori_img, lr_img, src_cond_img, out, render_img[:, :3]], 3).clamp(0, 1)
                # save_img = torch.cat([pair_img, pair_joints_img, src_ori_img, src_joints_img, lr_img, out], 3)
                for b in range(save_img.shape[0]):
                    cv2.imwrite(join(self.out_mesh_dire, 'eval_images', '%02d_%02d.png' % (i, b)), tensor2img(save_img, b))
            

        result = 'PSNR: {}, LPIPS: {}'.format(np.array(psnrs).mean(), np.array(lpipss).mean())
        f = open(os.path.join(self.out_mesh_dire, 'result.txt'), 'w')
        f.write(result)
        f.close()
        
        return np.array(psnrs).mean(), np.array(lpipss).mean()