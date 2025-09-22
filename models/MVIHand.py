import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr

from repose import lbs, lbs_pose, pose2rot
from models.mlp import ConditionNetwork
from models.unet import UNet
from models.utils import get_normals, compute_color, proj_verts2coord
from torchvision.utils import save_image



class MVIHand(nn.Module):
    def __init__(self, vertices, faces, init_delta, init_albedo, init_weights, template_v, sh_coeffs, delta_scale=1,
                 hand_type='left', render_nettype='mlp', use_pe=True, use_x_pos=True, use_ray=True, use_emb=True,
                 wo_latent=False, latent_num=20, use_rotpose=False, mlp_use_pose=True, use_sum=False, res=[512, 334], albedo_level=3, mano_layer=None, implicit_net=None, sample_query=5000, use_cond=False):
        '''

        :param init_delta: n_vertices, 3
        :param init_albedo: n_vertices, 3
        :param init_weights: n_vertices, num_joints
        '''
        super(MVIHand, self).__init__()
        init_weights = torch.log(1000 * (init_weights + 0.000001))
        self.template_v = template_v
        self.hand_type = hand_type
        self.verts = vertices
        self.faces = faces
        # self.glctx = dr.RasterizeGLContext()
        self.mano_layer = mano_layer
        self.use_rotpose = use_rotpose
        if self.use_rotpose:
            pose_num = 16 * 3 * 3
        else:
            pose_num = 16 * 3
        
        self.use_emb = use_emb        
        self.implicitNet = implicit_net
        self.use_cond = use_cond
        if self.use_emb:
            self.delta_net = ConditionNetwork(init_delta, pose_num, 3, 10, 512, 8, use_condition=self.use_cond,learnable_mean=False)
            self.color_net = ConditionNetwork(init_albedo, pose_num, 3, 10, 128, 5, learnable_mean=False)
            self.lbs_net =   ConditionNetwork(init_weights, pose_num, init_weights.shape[1], 10, 128, 5)
        self.render_nettype = render_nettype

        self.albedo_level = albedo_level
        self.use_pe = use_pe
        self.use_x_pos = use_x_pos
        self.use_ray = use_ray
        self.wo_latent = wo_latent

        self.sh_coeffs = nn.Parameter(sh_coeffs, requires_grad=True)
        self.sig = nn.Sigmoid()
        self.use_sum = use_sum
        
        self.delta_scale = delta_scale
        
        
    def forward_delta(self, condition, add_condition=None):
        return self.delta_net(input=condition, condition=None if add_condition is None else add_condition.detach())

    def forward_color(self, condition, if_sig=True):
        if if_sig:
            pred_albedo, code = self.color_net(condition)
            return self.sig(pred_albedo), code
        else:
            return self.color_net(condition)

    def forward_lbs(self, condition, if_softmax=True):
        pred_weights, code = self.lbs_net(condition)
        if if_softmax:
            pred_weights = F.softmax(pred_weights, -1)
        return pred_weights, code


    def renew_mean(self, delta_mean, albedo_mean, weights_mean):
        # weights_mean = torch.log(1000 * (weights_mean + 0.000001))
        if self.use_emb:
            self.lbs_net.x_mean = nn.Parameter(weights_mean.detach(), requires_grad=False)
            self.delta_net.x_mean = nn.Parameter(delta_mean.detach(), requires_grad=False)
            self.color_net.x_mean = nn.Parameter(albedo_mean.detach(), requires_grad=False)



    def forward(self, data_input, glctx, resolution, is_train=True, train_render=True, is_right=None):
        pose, trans, scale, w2c, proj, mask, ray, sh_coeff = data_input    

        n = w2c.shape[0]
        pose_n = pose.shape[0]
        if self.use_rotpose:
            matrix = pose2rot(pose.view(-1, 48).clone()).view([pose_n, -1, 3, 3])
        else:
            matrix = pose.clone()
        condition = matrix.reshape(pose_n, -1) / np.pi
        
        pred_weights, _ = self.forward_lbs(condition)
        pred_albedo, code_albedo = self.forward_color(condition)
        pred_delta, code_delta = self.forward_delta(condition, pred_albedo if self.use_cond else None)
        
        
        vertices_n = self.verts.unsqueeze(0) + pred_delta * self.delta_scale
        verts_new = lbs_pose(pose, self.template_v.unsqueeze(0),
                             pred_weights,
                             vertices_n, hand_type=self.hand_type)
        if is_right is not None:
            verts_new[:, :, 0] = (2 * is_right - 1) * verts_new[:, :, 0]
        vertices_new = verts_new * scale + trans.unsqueeze(1)
                
        # get posed verts
        vertsw = torch.cat([vertices_new, torch.ones_like(vertices_new[:, :, 0:1])], axis=2).expand(n, -1, -1)
        rot_verts = torch.einsum('ijk,ikl->ijl', vertsw, w2c)
        proj_verts = torch.einsum('ijk,ikl->ijl', rot_verts, proj)
        normals = get_normals(vertsw[:, :, :3], self.faces.long())
        
        rast_out, _ = dr.rasterize(glctx, proj_verts, self.faces, resolution=resolution)
        face_id = rast_out[..., -1].long()
        
        feat = torch.cat([normals, pred_albedo.expand(n, -1, -1), torch.ones_like(vertsw[:, :, :1]), vertices_new.expand(n, -1, -1), get_normals(rot_verts[:, :, :3], self.faces.long())], dim=2)
        feat, _ = dr.interpolate(feat, rast_out, self.faces)
        pred_normals = feat[:, :, :, :3].contiguous()
        rast_albedo = feat[:, :, :, 3:6].contiguous()
        pred_mask = feat[:, :, :, 6:7].contiguous()
        pred_vert = feat[:, :, :, 7:10].contiguous()
        gt_normals = feat[:, :, :, 10:13].contiguous()

        pred_normals = F.normalize(pred_normals, p=2, dim=3)
        pred_mask = dr.antialias(pred_mask, rast_out, proj_verts, self.faces).squeeze(-1)
        valid_idx = torch.where((mask > 0) & (rast_out[:, :, :, 3] > 0)) if mask is not None else torch.where(rast_out[:, :, :, 3] > 0)
        valid_normals = pred_normals[valid_idx]
        if is_right is not None:
            valid_normals = valid_normals * (2 * is_right - 1)
        
        coord = proj_verts2coord(proj_verts)
        
        v_mask = torch.zeros((n, vertices_new.shape[1]), dtype=torch.bool, device=vertices_new.device)
        
        for b in range(n):
            v_ids = self.faces[(face_id[b][pred_mask[b] > 0] - 1).unique()].unique().long()
            v_mask[b].index_fill_(0, v_ids, 1)
        queries = coord * v_mask.unsqueeze(-1)
        queries = torch.cat([queries, v_mask.unsqueeze(-1)], -1)
        gt_normals = dr.antialias(gt_normals, rast_out, proj_verts, self.faces)
            
        normal_img = torch.zeros_like(pred_normals)
        normal_img[valid_idx] = 0.5 + (gt_normals[valid_idx] / 2)
        # import pdb; pdb.set_trace()
        valid_shcoeff = sh_coeff[valid_idx[0]]
        valid_albedo = rast_albedo[valid_idx]
        
        valid_albedo_mean = valid_albedo.mean().expand(valid_albedo.unsqueeze(0).shape)
        
        mean_pix = torch.clip(compute_color(valid_albedo_mean, valid_normals.unsqueeze(0), valid_shcoeff)[0], 0, 1)
        mean_img = torch.zeros_like(pred_normals)
        mean_img[valid_idx] = mean_pix
        mean_img = dr.antialias(mean_img, rast_out, proj_verts, self.faces)
        
        pred_img = torch.clip(compute_color(valid_albedo.unsqueeze(0), valid_normals.unsqueeze(0), valid_shcoeff)[0], 0, 1)
        render_imgs = torch.zeros_like(pred_normals)
        tmp_img = torch.zeros_like(pred_normals)
        tmp_img[valid_idx] = pred_img
        tmp_img = dr.antialias(tmp_img, rast_out, proj_verts, self.faces)
        
        render_imgs = tmp_img
        if self.use_sum:
            render_imgs = render_imgs + tmp_img
        
        if is_train:
            return valid_idx, render_imgs, tmp_img, mean_img, normal_img, pred_delta, vertices_new, pred_weights, pred_albedo, pred_mask, queries
        
        else:
            gt_normals = dr.antialias(gt_normals, rast_out, proj_verts, self.faces)
            
            normal_img = torch.zeros_like(pred_normals)
            normal_img[valid_idx] = 0.5 + (gt_normals[valid_idx] / 2)
            
            gt_normals = F.normalize(gt_normals, p=2, dim=3)
            gt_normals = gt_normals[valid_idx]
            if is_right is not None:
                gt_normals = gt_normals * (2 * is_right - 1)
            light_direction = torch.zeros_like(gt_normals)
            light_direction[:, 2] = -1
            reflect = (-light_direction) - 2 * gt_normals * torch.sum(gt_normals * (-light_direction), dim=1, keepdim=True)
            dot = torch.sum(reflect * light_direction, dim=1, keepdim=True)  # n 1
            specular = 0.2 * torch.pow(torch.maximum(dot, torch.zeros_like(dot)), 16)
            color = torch.sum(gt_normals * light_direction, dim=1, keepdim=True) + specular
            color = torch.clamp(color, 0, 1)
            # color = color.squeeze().detach().cpu().numpy()
            mesh_img = torch.zeros_like(pred_normals)
            mesh_img[valid_idx] = color

            # pred_color = compute_color(pred_albedo, get_normals(rot_verts[:, :, :3], self.faces.long()), sh_coeff)
            
            return render_imgs, mesh_img, normal_img, tmp_img, rast_albedo, vertices_new, pred_weights, pred_mask, pred_albedo, queries

