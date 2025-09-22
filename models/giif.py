import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from einops import rearrange
from PIL import Image

import copy
from models.esdr import make_edsr_baseline
from models.RDN import make_rdn
from models.giif_modules import *
from models.modules.unet import UNet, UNet_no_t
from models.encoder import Encoder


from projection import *

from models.positional_encoding import PostionalEncoding
from models.modules.depthNormalizer import DepthNormalizer
from models.utils import get_normals


models = {}

def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator

def make_coord(shape, ranges=None, flatten=True, align_corners=False):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        
        if align_corners:
            seq = v0 + (2 * r) * torch.arange(n+1).float()
        else:
            seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def make_coord_cell(bs, h, w):
    coord = make_coord((h, w)).cuda()
    
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w

    coord = coord.repeat((bs,) + (1,) * coord.dim())
    cell = cell.repeat((bs,) + (1,) * cell.dim())
    return coord, cell

def to_pixel_samples(b, h, w):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """

    coord, cell = make_coord_cell(b, h, w)
    return coord, cell


def make(model_spec, args=None, load_sd=False):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model = models[model_spec['name']](**model_args)
    if load_sd:
        model.load_state_dict(model_spec['sd'])
    return model



from models.modules.HGFilter import HGFilter
import nvdiffrast.torch as dr
    
class GIIF(nn.Module):
    def __init__(self, encoder_spec, imnet_spec=None, n_feat=128, use_normal=True, use_depth=False,
                 local_ensemble=True, feat_unfold=True, cell_decode=True, img_size=256, mano_faces=None):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.img_size = img_size
        self.n_feat = encoder_spec['args']['n_feats']
        
        self.use_normal = use_normal
        self.use_depth = use_depth
        print('use_normal: ', self.use_normal)
        print('use_depth: ', self.use_depth)

        in_channel = 3
        if self.use_normal and self.use_depth:
            in_channel += 3
        # self.encoder = MLP(in_dim=5, out_dim=self.n_feat, hidden_list=[256, 256])
        self.encoder = make_rdn(G0=128, no_upsampling=encoder_spec['args']['no_upsampling'])
        self.im_encoder = HGFilter(
            in_channel=in_channel,
            n_feat=self.n_feat, 
            hourglass_dim=128
        )
        # self.im_encoder = HGFilter(
        #     in_channel=3
        # )
        # self.encoder = HGFilter(in_channel=3)
        self.normalizer = DepthNormalizer(
            loadSize=img_size,
            z_size=100
        )       
        if imnet_spec is not None:
            imnet_in_dim = n_feat * 2
            # imnet_in_dim = n_feat * 4
            # if self.feat_unfold:
            #     imnet_in_dim *= 9
            imnet_in_dim += 2 # attach coord
            imnet_in_dim += 2 # attach coord
            if self.cell_decode:
                imnet_in_dim += 2
            
            self.imnet = MLP(in_dim=imnet_in_dim, out_dim=imnet_spec['args']['out_dim'], hidden_list=imnet_spec['args']['hidden_list'])
        else:
            self.imnet = None
            
        self.mano_faces = mano_faces
        self.rotate =  False
        self.pe = PostionalEncoding(min_deg=0, max_deg=1, scale=0.1)
        # self.verts_net = MLP(in_dim=87 + self.n_feat, out_dim=3, hidden_list=imnet_spec['args']['hidden_list'])
        # self.verts_net = MLP(in_dim=32768, out_dim=778*3, hidden_list=imnet_spec['args']['hidden_list'])
        
   
    def index(self, feat, query):
        if query.shape[-1] != 2:
            query = query.permute(0, 2, 1)
        query = query.unsqueeze(2)  # [B, N, 1, 2]
        samples = torch.nn.functional.grid_sample(feat, query, align_corners=True)  # [B, C, N, 1]
        return samples[..., 0]   # [B, 128, 16]
    

    def encode(self, inp):
        n_batch = inp.shape[0]
        # [4, 64, 32, 32] = self.encoder(inp)
        im_feat = self.encoder(inp)       # [B, n_feat, 256, 256]
        
        return im_feat
        
    def query_image(self, feat, lr_im_feat, coord, cell=None):
        
        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        # if self.feat_unfold:
        #     feat = F.unfold(feat, 3, padding=1).view(
        #         feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        lr_feat_coord = make_coord(lr_im_feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(lr_im_feat.shape[0], 2, *lr_im_feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                lq_feat = F.grid_sample(
                    lr_im_feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                lq_coord = F.grid_sample(
                    lr_feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord, lq_feat, lq_coord], dim=-1)
                
                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret
    
    def render_hand(self, vertices, w2c, proj, glctx, faces=None, out_size=(256, 256)):
        B = vertices.shape[0]
        vertsw = vertices
        faces = faces if faces is not None else self.mano_faces
        
        vertsw = torch.cat([vertices, torch.ones_like(vertices[:, :, 0:1]).cuda()], axis=2)
        rot_verts = rotate_n(vertsw, w2c)
        proj_verts = projection_n(rot_verts, proj)

        rast_out, _ = dr.rasterize(glctx, proj_verts, faces, resolution=(out_size[0], out_size[1]))
        depth = rast_out[..., 2:3]
        
        # depth[depth != 0] = depth[depth != 0] - depth[depth != 0].min()
        depth[depth != 0] = depth[depth != 0] - depth[depth != 0].min()
        depth = depth / depth.max()
        depth = depth.repeat(1, 1, 1, 3)


        face_id = rast_out[..., -1].long()
        normals = get_normals(vertices[:, :, :3], faces.long())# .flip(-1)
        whites = torch.ones_like(vertices[:, :, :1])
        feat = torch.cat([normals, whites], dim=2)
        feat, _ = dr.interpolate(feat, rast_out, faces)
        
        pred_feat = feat[..., :3].contiguous()
        mask = feat[..., 3:4].contiguous().squeeze(-1)
        # valid_idx = torch.where((mask > 0) & (rast_out[:, :, :, 3] > 0))
        
        # pred_img = torch.gather(self.verts_color.repeat(4, 1, 1), 1, face_id).reshape(B, out_size[0], out_size[1], 3)
        pred_feat = dr.antialias(pred_feat, rast_out,proj_verts, faces)
        
        v_mask = []
        for b in range(B):
            visible_vertices = torch.zeros(vertices.shape[1], dtype=torch.bool, device=vertices.device)
            v_ids = faces[(face_id[b][mask[b] > 0] - 1).unique()].unique().long()
            visible_vertices[v_ids] = 1
            v_mask.append(visible_vertices) 
        
        v_mask = torch.stack(v_mask, dim=0)
        
        if self.use_normal and self.use_depth:
            pred_feat = torch.cat([pred_feat, depth], dim=-1)
        elif self.use_depth:
            pred_feat = depth
        pred_feat = pred_feat.permute(0, 2, 1, 3).flip(1) if self.rotate else pred_feat
        return pred_feat, face_id, mask
    
    def forward_img(self, lr_img, feat, lr_im_feat=None, queries=None, out_size=(256, 256), use_pdb=False):
        if lr_img.shape[-1] == 3:
            lr_img = lr_img.permute(0, 3, 1, 2)
        if feat.shape[-1] != out_size[0]:
            feat = feat.permute(0, 3, 1, 2)

        B = feat.shape[0]
        LW, LH = lr_img.shape[2], lr_img.shape[3]
        
        # image = feat[:, :3]
        # image = (image - 0.5) / 0.5
        
        
        normal = (feat[:, :3] - 0.5) / 0.5
        input_feat = torch.cat([normal, feat[:, 3:]], dim=1)
        
        lr_img = (lr_img - 0.5) / 0.5
        
        if lr_im_feat is None:
            lr_im_feat = self.encoder(lr_img)

        if queries is None:
            coord, cell = to_pixel_samples(B, out_size[0], out_size[1])
        else:
            coord = queries
        
            cell = torch.ones_like(coord)
            cell[:, :, 0] *= 2 / out_size[0]
            cell[:, :, 1] *= 2 / out_size[1]           
        
        im_feat_list, _, _ = self.im_encoder(input_feat)
        # im_feat = torch.cat(im_feat_list, dim=1)
        im_feat = im_feat_list[-1]
        # im_feat = self.encoder(image)
        
        ret = self.query_image(im_feat, lr_im_feat, coord, cell)

        if queries is None:
            # ret = ret + image.reshape(B, 3, -1).permute(0, 2, 1)
            ret = ret * 0.5 + 0.5
            return ret.reshape(B, out_size[0], out_size[1], -1).permute(0, 3, 1, 2)
        # # B, H, W, C = image.shape
        # # query = query.permute(0, 2, 1)

        return ret

    def forward(self, src_dict, glctx, hr_size=(256, 256)):
        # img [8, 3, 256, 256]
        # pose [1, 48]
        # w2c [ 8, 4, 4]
        # proj [8, 4, 4]
        lr_img, vertices, w2c, proj, cam_exs, cam_ins = src_dict['img'], src_dict['vertices'], src_dict['w2c'], src_dict['proj'], src_dict['cam']['ex'], src_dict['cam']['in']
        B = vertices.shape[0]
        

        lr_img = lr_img.permute(0, 1, 3, 2).flip(2) if self.rotate else lr_img
        pred_feat, face_id, depth  = self.render_hand(vertices.clone(), w2c, proj, glctx, out_size=hr_size)
        
        # , im_feat = self.forward_hand_color(images=lr_img, vertices=vertices, w2c=w2c, proj=proj, cam_ex=cam_exs, cam_in=cam_ins)
        # coord = torch.cat([coord, depth.reshape(B, -1, 1)], dim=2)
        # import pdb; pdb.set_trace()
        
        ret = self.forward_img(lr_img, pred_feat.permute(0, 3, 1, 2), out_size=hr_size)
        
        ret = ret.flip(2).permute(0, 1, 3, 2) if self.rotate else ret
        pred_feat = pred_feat.flip(1).permute(0, 2 ,1 ,3) if self.rotate else pred_feat

        return ret, pred_feat.permute(0, 3, 1, 2)
    


