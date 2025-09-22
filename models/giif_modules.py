import torch
import torch.nn as nn
import torch.nn.functional as F


from inspect import isfunction
from torch import nn, einsum
from einops import rearrange, repeat

import math



def to_pixel_samples(b, h, w):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """

    coord, cell = make_coord_cell(b, h, w)
    return coord, cell

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


def make_linear_layers(feat_dims, relu_final=True, use_bn=False, bias=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1], bias=bias))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_conv1d_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True, groups=1):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv1d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                groups=groups
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def make_deconv_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.ConvTranspose2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)
class Swish(nn.Module):
    def forward(self,x):
        return  x * torch.sigmoid(x)


class MLP_Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden, out_dim))
            lastv = out_dim
        self.layers = layers  
        # self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        
        #  for layers in self.layers:
            
        
        # x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.reshape(-1, x.shape[-1]))
        return x.view(*shape, -1)
    
class MLP_cond(nn.Module):
    def __init__(self, in_dim, out_dim, cond_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        self.pre_layer = nn.Linear(in_dim, hidden_list[0])
        self.layers = nn.ModuleList([])
        for hidden in hidden_list:
            self.layers.append(
                nn.ModuleList([
                    nn.Linear(cond_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    # nn.Identity(),
                ])
            )
            lastv = hidden
        self.final = nn.Linear(lastv, out_dim)

    def forward(self, x, c):
        bs = c.shape[0]
        q = x.shape[0] // bs
        shape = x.shape[:-1]
        h = x.view(-1, x.shape[-1])
        h = self.pre_layer(h)
        for i, (lin_c, act1, lin2) in enumerate(self.layers):
            h = h.reshape(bs, q, -1)
            h = h + lin_c(c).unsqueeze(1)
            h = h.reshape(*shape, -1)
            h = act1(lin2(h))  
            
        return self.final(h)
class LMMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, hidden_depth,
                 mod_scale=True, mod_shift=True, mod_up_merge=False, use_conv=False):
        super().__init__()
        self.hidden_depth = hidden_depth
        self.hidden_dim = hidden_dim

        # Modulation configs
        self.mod_scale = mod_scale
        self.mod_shift = mod_shift
        self.mod_dim = 0
        # If we modulate both scale and shift, we have twice the number of modulations at every layer and feature
        self.mod_dim += hidden_dim if self.mod_scale else 0
        self.mod_dim += hidden_dim if self.mod_shift else 0

        # For faster inference, set to True if upsample scale mod and shift mod together.
        self.mod_up_merge = mod_up_merge and self.mod_scale and self.mod_shift

        layers = []
        lastv = in_dim
        for _ in range(hidden_depth):
            if use_conv:
                layers.append(nn.Conv2d(lastv, hidden_dim, 1))
            else:
                layers.append(nn.Linear(lastv, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            lastv = hidden_dim
        if use_conv:
            layers.append(nn.Conv2d(lastv, out_dim, 1))
        else:
            layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, mod=None, coord=None, only_layer0=False, skip_layer0=False):
        shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1])

        if only_layer0:
            return self.layers[0](x)

        if coord is None:
            mod = mod.view(-1, mod.shape[-1])

        # Split modulations into shifts and scales and apply them to hidden features
        mid_dim = (
            self.mod_dim * self.hidden_depth // 2 if (self.mod_scale and self.mod_shift) else 0
        )

        for idx, module in enumerate(self.layers):
            if not (skip_layer0 and idx == 0):
                x = module(x)

            if idx == self.hidden_depth * 2 or idx % 2 == 1:
                # skip output linear layer or hidden activation layer
                continue

            start, end = (idx // 2) * self.hidden_dim, ((idx // 2) + 1) * self.hidden_dim

            # Use modulations on hidden linear layer outputs
            if self.mod_up_merge and coord is not None:
                # Upsample scale mod and shift mod together when GPU memory is sufficient.
                bs, q = coord.shape[:2]
                q_mod = F.grid_sample(
                    torch.cat([mod[:, start: end, :, :], mod[:, mid_dim + start: mid_dim + end, :, :]], dim=1),
                    coord.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous().view(bs * q, -1)
                x *= q_mod[:, :self.hidden_dim]
                x += q_mod[:, self.hidden_dim:]
            else:
                if self.mod_scale:
                    # Shape (b * h * w, hidden_dim). Note that we add 1 so modulations remain zero centered
                    if coord is not None:
                        bs, q = coord.shape[:2]
                        x *= (F.grid_sample(
                            mod[:, start: end, :, :], coord.flip(-1).unsqueeze(1),
                            mode='nearest', align_corners=False)[:, :, 0, :]
                              .permute(0, 2, 1).contiguous().view(bs * q, -1) + 1.0)
                    else:
                        x *= (mod[:, start: end] + 1.0)

                if self.mod_shift:
                    # Shape (b * h * w, hidden_dim)
                    if coord is not None:
                        bs, q = coord.shape[:2]
                        x += F.grid_sample(
                            mod[:, mid_dim + start: mid_dim + end, :, :], coord.flip(-1).unsqueeze(1),
                            mode='nearest', align_corners=False)[:, :, 0, :] \
                            .permute(0, 2, 1).contiguous().view(bs * q, -1)
                    else:
                        x += mod[:, mid_dim + start: mid_dim + end]

            # Broadcast scale and shift across x
            # scale, shift = 1.0, 0.0
            #x = scale * x + shift

        return x.view(*shape, -1)
    
    
class ConvEncoder(nn.Module):
    r''' Simple convolutional encoder network.

    It consists of 5 convolutional layers, each downsampling the input by a
    factor of 2, and a final fully-connected layer projecting the output to
    c_dim dimenions.

    Args:
        c_dim (int): output dimension of latent embedding
    '''
    # [3, 32] --> 128
    # [32, 64] --> 64
    # [64, 128] --> 32
    # [128, 256] --> 16
    # [256, 512] --> 8
    def __init__(self, c_dims=[3, 32, 64, 128, 256, 512], kernel_size=3, stride=2, padding=1):
        super().__init__()
        
        self.c_dims = c_dims
        self.layers = nn.ModuleList([])
        for i in range(len(c_dims)-1):
            self.layers.append(nn.Sequential(
                nn.Conv2d(c_dims[i], c_dims[i+1], kernel_size=kernel_size, stride=stride, padding=padding)
            ))
        self.actvn = nn.ReLU()

    def forward(self, x):
        for l in self.layers:
            x = l(x)
            x = self.actvn(x)
        
        return x

class ConvEncoderMLP(nn.Module):
    r''' Simple convolutional encoder network.

    It consists of 5 convolutional layers, each downsampling the input by a
    factor of 2, and a final fully-connected layer projecting the output to
    c_dim dimenions.

    Args:
        c_dim (int): output dimension of latent embedding
    '''

    def __init__(self, c_dims=[3, 32, 64, 128, 256, 512], out_dim=42):
        super().__init__()
        self.encoder = ConvEncoder(c_dims)
        
        self.fc_out = nn.Linear(c_dims[-1], out_dim)
        self.actvn = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        net = self.encoder(x)
       
        net = net.view(batch_size, 512, -1).mean(2)
        out = self.fc_out(self.actvn(net))

        return out


class Attention(nn.Module):
    def __init__(self, in_ch, num_groups, D=3):
        super(Attention, self).__init__()
        assert in_ch % num_groups == 0
        if D == 3:
            self.q = nn.Conv3d(in_ch, in_ch, 1)
            self.k = nn.Conv3d(in_ch, in_ch, 1)
            self.v = nn.Conv3d(in_ch, in_ch, 1)

            self.out = nn.Conv3d(in_ch, in_ch, 1)
        elif D == 2:
            self.q = nn.Conv2d(in_ch, in_ch, 1)
            self.k = nn.Conv2d(in_ch, in_ch, 1)
            self.v = nn.Conv2d(in_ch, in_ch, 1)

            self.out = nn.Conv2d(in_ch, in_ch, 1)
        elif D == 1:
            self.q = nn.Conv1d(in_ch, in_ch, 1)
            self.k = nn.Conv1d(in_ch, in_ch, 1)
            self.v = nn.Conv1d(in_ch, in_ch, 1)

            self.out = nn.Conv1d(in_ch, in_ch, 1)

        self.norm = nn.GroupNorm(num_groups, in_ch)
        self.nonlin = Swish()

        self.sm = nn.Softmax(-1)


    def forward(self, query, key, value):
        B, C = value.shape[:2]
        x = value

        q = self.q(query).reshape(B,C,-1)
        k = self.k(key).reshape(B,C,-1)
        v = self.v(value).reshape(B,C,-1)
        qk = torch.matmul(q.permute(0, 2, 1), k) #* (int(C) ** (-0.5))

        w = self.sm(qk)

        h = torch.matmul(v, w.permute(0, 2, 1)).reshape(B,C,*value.shape[2:])

        h = self.out(h)

        x = h + x

        x = self.nonlin(self.norm(x))

        return x



class SelfAttention(nn.Module):
    def __init__(self, in_ch, num_groups, D=3):
        super(SelfAttention, self).__init__()
        assert in_ch % num_groups == 0
        if D == 3:
            self.q = nn.Conv3d(in_ch, in_ch, 1)
            self.k = nn.Conv3d(in_ch, in_ch, 1)
            self.v = nn.Conv3d(in_ch, in_ch, 1)

            self.out = nn.Conv3d(in_ch, in_ch, 1)
        elif D == 1:
            self.q = nn.Conv1d(in_ch, in_ch, 1)
            self.k = nn.Conv1d(in_ch, in_ch, 1)
            self.v = nn.Conv1d(in_ch, in_ch, 1)

            self.out = nn.Conv1d(in_ch, in_ch, 1)

        self.norm = nn.GroupNorm(num_groups, in_ch)
        self.nonlin = Swish()

        self.sm = nn.Softmax(-1)


    def forward(self, x):
        B, C = x.shape[:2]
        h = x

        q = self.q(h).reshape(B,C,-1)
        k = self.k(h).reshape(B,C,-1)
        v = self.v(h).reshape(B,C,-1)

        qk = torch.matmul(q.permute(0, 2, 1), k) #* (int(C) ** (-0.5))

        w = self.sm(qk)

        h = torch.matmul(v, w.permute(0, 2, 1)).reshape(B,C,*x.shape[2:])

        h = self.out(h)

        x = h + x

        x = self.nonlin(self.norm(x))

        return x


class IDNet(nn.Module):
    def __init__(self, id_code_dim):
        super(IDNet, self).__init__()
        self.encoder = ConvEncoder()
        self.id_out = make_linear_layers([self.encoder.c_dims[-1], 32, id_code_dim], relu_final=False)

    def forward(self, img,):
        batch_size = img.size(0)
        
        enc = self.encoder(img)
        
        enc = enc.view(batch_size, 512, -1).mean(2)
        out = self.id_out(enc)
        return out


class ImageJointEncoder(nn.Module):
    def __init__(self, in_ch, joint_ch, ch_list=[32, 64, 128], latent_size=32, num_joints=16):
        super(ImageJointEncoder, self).__init__()
        
        self.layers = []
        self.im_encoder = []
        self.latent_size = latent_size
        self.num_joints = num_joints
        
        layers = []
        ch_list.insert(0, in_ch)
        for i in range(len(ch_list) - 1):
            layers.append(nn.Conv2d(ch_list[i], ch_list[i+1], kernel_size=3, stride=2, padding=1))    
            if i == len(ch_list) - 2:
                layers.append(nn.ReLU())
        
        self.im_encoder = nn.Sequential(*layers)

        self.joint_encoder = make_linear_layers([ch_list[-1], 64, joint_ch], relu_final=False)


            
    def index(self, feat, joint):
        joint = joint.unsqueeze(2)  # [B, N, 1, 2]
        samples = torch.nn.functional.grid_sample(feat, joint, align_corners=True)  # [B, C, N, 1]
        return samples[..., 0]   # [B, C, N] [B, 128, 16]
        
        
    def forward(self, img, joint):
        B, H, W = img.shape[0], img.shape[2], img.shape[3]
        img_enc = self.im_encoder(img) # [B, C, 32, 32]
        # joint_enc = self.joint_encoder(joint).permute(0, 2, 1) # [B, C, J]
        joint = torch.clamp((joint - 0.5) * 2, -1, 1)
        im_joint_feat = self.index(img_enc, joint[..., :2]).permute(0, 2, 1)
        
        im_joint_feat = self.joint_encoder(im_joint_feat)
        
        
        # img_enc : [B, C, 32, 32]
        # im_joint_feat : [B, C, N]
        return img_enc, im_joint_feat
    
    
heatmap_tensor = torch.tensor([0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63], requires_grad=False)

class PoseNet(nn.Module):
    def __init__(self, joint_num):
        super(PoseNet, self).__init__()
        self.joint_num = joint_num

        if cfg.simple_meshnet:
            self.deconv = make_deconv_layers([2048,256,256,256])

            self.conv_x = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
            self.conv_y = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
            self.conv_z = make_conv1d_layers([2048, 256, self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
        else:
            self.deconv = make_deconv_layers([2048,256,256,256])
            self.conv_x = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
            self.conv_y = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)
            self.conv_z_1 = make_conv1d_layers([2048,256*cfg.output_hm_shape[0]], kernel=1, stride=1, padding=0)
            self.conv_z_2 = make_conv1d_layers([256,self.joint_num], kernel=1, stride=1, padding=0, bnrelu_final=False)

        self.conv = make_conv_layers([256, 64], kernel=3, stride=1, padding=1)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 2)
        heatmap_size = heatmap1d.shape[2]
        # coord = heatmap1d * torch.cuda.comm.broadcast(torch.arange(heatmap_size).type(torch.cuda.FloatTensor), devices=[heatmap1d.device.index])[0]
        coord = heatmap1d * torch.arange(heatmap_size).type(torch.cuda.FloatTensor).to(heatmap1d.device)
        coord = coord.sum(dim=2, keepdim=True)
        return coord

    def forward(self, img_feat):
        # Batchsize x 2048 x 8 x 8
        img_feat_xy = self.deconv(img_feat)
        # Batchsize x 256 x 64 x 64

        # x axis
        img_feat_x = img_feat_xy.mean((2))
        heatmap_x = self.conv_x(img_feat_x)
        coord_x = self.soft_argmax_1d(heatmap_x)
        # y axis
        img_feat_y = img_feat_xy.mean((3))
        heatmap_y = self.conv_y(img_feat_y)
        # (BatchSize x 21 x 64)
        coord_y = self.soft_argmax_1d(heatmap_y)
        
        # z axis
        # img_feat_z = img_feat.mean((2,3))[:,:,None]
        # # (BatchSize x 2048 x 1)
        # img_feat_z = self.conv_z_1(img_feat_z)
        # img_feat_z = img_feat_z.view(-1,256,cfg.output_hm_shape[0])
        # heatmap_z = self.conv_z_2(img_feat_z)
        # # (BatchSize x 21 x 64)
        # coord_z = self.soft_argmax_1d(heatmap_z)

        if cfg.simple_meshnet:
            img_feat_z = img_feat.view(-1, 2048, 64)
            heatmap_z = self.conv_z(img_feat_z)
            coord_z = self.soft_argmax_1d(heatmap_z)
        else:
            img_feat_z = img_feat.mean((2,3))[:,:,None]
            img_feat_z = self.conv_z_1(img_feat_z)
            img_feat_z = img_feat_z.view(-1,256,cfg.output_hm_shape[0])
            heatmap_z = self.conv_z_2(img_feat_z)
            coord_z = self.soft_argmax_1d(heatmap_z)
        
        joint_coord = torch.cat((coord_x, coord_y, coord_z),2)
        joint_coord_feat = torch.cat((heatmap_x, heatmap_y, heatmap_z),2)

        joint_feat = self.conv(img_feat_xy)
        # joint_feat = img_feat_xy

        return joint_coord, joint_coord_feat, joint_feat
    
    
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h
    
    





def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

