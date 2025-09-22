import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.pvcnn import SimplePointModel, PVCNN2


from models.utils import compute_laplacian ,get_adjacency
class MLP(torch.nn.Module):

    def __init__(self, input_c, output_c, num_h=256, depth=8):
        super(MLP, self).__init__()

        self.model = nn.Sequential()
        for i in range(depth):
            if i == 0:
                self.model.add_module('linear%d' % (i + 1), torch.nn.Linear(input_c, num_h))
                self.model.add_module('relu%d' % (i + 1), torch.nn.ReLU())
            elif i != depth - 1:
                self.model.add_module('linear%d' % (i + 1), torch.nn.Linear(num_h, num_h))
                self.model.add_module('relu%d' % (i + 1), torch.nn.ReLU())
            else:
                self.model.add_module('linear%d' % (i + 1), torch.nn.Linear(num_h, output_c))
        self.sig = nn.Sigmoid()
    def forward(self, x):
        return self.sig(self.model(x))

class MLP_res(torch.nn.Module):

    def __init__(self, input_c, output_c, num_h=256, depth=8, res_layer=[4]):
        super(MLP_res, self).__init__()

        self.model = nn.ModuleDict()
        self.res_layer = res_layer
        self.depth = depth
        for i in range(depth):
            if i == 0:
                self.model['linear%d' % (i + 1)] = torch.nn.Linear(input_c, num_h)
                self.model['relu%d' % (i + 1)]= torch.nn.ReLU()
            elif i != depth - 1:
                if i + 1 in self.res_layer:
                    self.model['linear%d' % (i + 1)] = torch.nn.Linear(num_h + input_c, num_h)
                    self.model['relu%d' % (i + 1)] = torch.nn.ReLU()
                else:
                    self.model['linear%d' % (i + 1)] = torch.nn.Linear(num_h, num_h)
                    self.model['relu%d' % (i + 1)] = torch.nn.ReLU()
            else:
                self.model['linear%d' % (i + 1)] =  torch.nn.Linear(num_h, output_c)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        tmp_x = x.clone()
        for i in range(self.depth):
            if i != self.depth - 1:
                if i + 1 in self.res_layer:
                    x = torch.cat([x, tmp_x], 1)
                x = self.model['relu%d' % (i + 1)](self.model['linear%d' % (i + 1)](x))
            else:
                x = self.model['linear%d' % (i + 1)](x)

        return self.sig(x)
    
    
class EmbedderNERF:
    def __init__(self, input_dims=3, include_input=True, max_freq_log2=10-1, num_freqs=10, log_sampling=True, periodic_fns=[torch.sin, torch.cos]):
        self.input_dims = input_dims
        self.include_input = include_input
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns

        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs
        
        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], 1)


def get_embedder_nerf(multires, input_dims=3, i=0):
    if i == -1:
        return nn.Identity(), input_dims
    
    embed_kwargs = {
                'input_dims' : input_dims,
                'include_input' : True,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = EmbedderNERF(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(PI * x)
    return embed, embedder_obj.out_dim

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                        freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim


""" MLP for neural implicit shapes. The code is based on https://github.com/lioryariv/idr with adaption. """


class ConditionNetwork(torch.nn.Module):
    def __init__(
            self,
            x_mean,
            d_in,
            d_out,
            d_k,
            width,
            depth,
            d_latent=10,
            geometric_init=True,
            bias=1.0,
            weight_norm=True,
            learnable_mean=False,
            multires=0,
            skip_layer=[],
            use_condition=False
    ):
        super().__init__()
        self.use_condition = use_condition
        
        dims = [d_in + d_latent + (3 if use_condition else 0)] + [width] * depth + [d_k]
        
        self.num_layers = len(dims)

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.skip_layer = skip_layer

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_layer:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = torch.nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(
                        lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001
                    )
                    torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_layer:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = torch.nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = torch.nn.Softplus(beta=100)
        self.softmax = torch.nn.Softmax(dim=0)

        if not learnable_mean:
            self.x_mean = nn.Parameter(x_mean, requires_grad=False)
        else:
            self.x_mean = nn.Parameter(x_mean, requires_grad=True)

        assert self.x_mean.shape[1] == d_out
        self.point_num = self.x_mean.shape[0]
        self.static_code = nn.Parameter(torch.zeros(d_k, d_out))
        self.latent_code = nn.Parameter(torch.zeros(self.point_num, d_latent))

    def forward(self, input, z_code=None, mask=None, condition=None):
        """MPL query.

        Tensor shape abbreviation:
            B: batch size
            N: number of points
            D: input dimension

        Args:
            input (tensor): network input. shape: [B, N, D]
            cond (dict): conditional input.
            mask (tensor, optional): only masked inputs are fed into the network. shape: [B, N]

        Returns:
            output (tensor): network output. Might contains placehold if mask!=None shape: [N, D, ?]
        """

        n_batch, n_dim = input.shape  # [1, 48]
        
        
        input = torch.cat([input.unsqueeze(1).expand(-1, self.point_num, -1), self.latent_code.unsqueeze(0).expand(n_batch, -1, -1)], 2)
        if condition is not None:
            input = torch.cat([input, condition], 2)
        if mask is not None:
            input = input[mask]

        input_embed = input if self.embed_fn is None else self.embed_fn(input)
        
        x = input_embed  # [1, 49281, 10]

        # z_sample = F.grid_sample(z_code, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_layer:
                x = torch.cat([x, input_embed], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        # add placeholder for masked prediction
        if mask is not None:
            x_full = torch.zeros(n_batch, self.point_num, x.shape[-1], device=x.device)
            x_full[mask] = x
        else:
            x_full = x

        out = self.x_mean + torch.matmul(x_full, self.softmax(self.static_code))

        return out, x_full


class SimpleNetwork(torch.nn.Module):
    def __init__(
            self,
            d_in,
            d_out,
            width,
            depth,
            d_latent=10,
            geometric_init=True,
            bias=1.0,
            weight_norm=True,
            multires=0,
            skip_layer=[],
    ):
        super().__init__()

        dims = [d_in + d_latent] + [width] * depth + [d_out]
        self.num_layers = len(dims)

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.skip_layer = skip_layer

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_layer:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = torch.nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(
                        lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001
                    )
                    torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_layer:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = torch.nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = torch.nn.Softplus(beta=100)

        self.point_num = 49281
        self.latent_code = nn.Parameter(torch.zeros(self.point_num, d_latent))

    def forward(self, input, mask=None):
        """MPL query.

        Tensor shape abbreviation:
            B: batch size
            N: number of points
            D: input dimension

        Args:
            input (tensor): network input. shape: [B, N, D]
            cond (dict): conditional input.
            mask (tensor, optional): only masked inputs are fed into the network. shape: [B, N]

        Returns:
            output (tensor): network output. Might contains placehold if mask!=None shape: [N, D, ?]
        """

        n_batch, n_dim = input.shape

        input = torch.cat([input.unsqueeze(1).expand(-1, self.point_num, -1), self.latent_code.unsqueeze(0).expand(n_batch, -1, -1)], 2)
        if mask is not None:
            input = input[mask]

        input_embed = input if self.embed_fn is None else self.embed_fn(input)

        x = input_embed

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_layer:
                x = torch.cat([x, input_embed], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        # add placeholder for masked prediction
        if mask is not None:
            x_full = torch.zeros(n_batch, self.point_num, x.shape[-1], device=x.device)
            x_full[mask] = x
        else:
            x_full = x

        return x_full, self.latent_code


class MLP_Detail(nn.Module):
    def __init__(self, manifold_pos_dim, pose_dim, hidden_dim = 800, num_hidden_layer = 3, output_dim = 3):
        super(MLP_Detail, self).__init__()

        self.pos_embedder, pos_embedder_out_dim = get_embedder_nerf(10, input_dims=manifold_pos_dim, i=0)
        # self.pos_embedder, pos_embedder_out_dim = get_embedder_nerf(10, input_dims=manifold_pos_dim, i=0)

        self.fc_input = nn.Linear(pos_embedder_out_dim + pose_dim, hidden_dim)
        self.fc_hidden = self.make_layer(nn.Linear(hidden_dim, hidden_dim), num_hidden_layer)
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def make_layer(self, layer, num):
        layers = []
        for _ in range(num):
            layers.append(nn.LeakyReLU())
            layers.append(layer)
        return nn.Sequential(*layers)

    def forward(self, input_manifold_pos, input_pose_code):
        input_pos_embed = self.pos_embedder(input_manifold_pos)
        x = self.fc_input(torch.cat((input_pos_embed, input_pose_code), dim=1))
        x = self.fc_hidden(x)
        return self.output_linear(x)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
               
class GCN_layer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN_layer, self).__init__()

        self.gc1 = GraphConvolution(in_features, hidden_features)
        self.gc2 = GraphConvolution(hidden_features, out_features)
        

    def forward(self, x, adj, act=None):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        
        if act == 'relu':
            return F.relu(x)
        elif act == 'sig':
            return F.sigmoid(x)
        elif act == 'tanh':
            return F.tanh(x)
        return x

class LaplacianNetwork(torch.nn.Module):
    def __init__(
            self,
            x_mean, x_faces,
            d_in,
            d_out,
            d_k,
            width,
            depth,
            d_latent=10,
            geometric_init=True,
            bias=1.0,
            weight_norm=True,
            learnable_mean=False,
            multires=0,
            skip_layer=[],
            mano_layer=None
    ):
        super().__init__()

        dims = [d_in + d_latent] + [width] * depth + [d_out]
        self.num_layers = len(dims)
        
        self.softplus = torch.nn.Softplus(beta=100)
        self.softmax = torch.nn.Softmax(dim=0)
        self.embed_fn = None
        self.d_latent = d_latent
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.x_faces = x_faces
        if not learnable_mean:
            self.x_mean = nn.Parameter(x_mean, requires_grad=False)
        else:
            self.x_mean = nn.Parameter(x_mean, requires_grad=True)
        
        self.joints_name = ['Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Index_1', 'Index_2', 'Index_3', 'Middle_1', 'Middle_2', 'Middle_3', 'Ring_1', 'Ring_2', 'Ring_3', 'Pinky_1', 'Pinky_2', 'Pinky_3']
        self.neighbor_joint_idxs = ([0,0,1], [0,1,2], [1,2,3], [2, 3, 0], [0,4,5], [4,5,6], [5,6,0], [0,7,8], [7,8,9], [8, 9, 0], [0,10,11], [10,11,12], [11,12,0], [0,13,14], [13,14,15], [14,15,0]) # without root joint
        self.neighbor_joint_mask = ([0,1,1], [1,1,1], [1,1,1], [0,0,0], [0,1,1], [1,1,1], [0,0,0], [0,1,1], [1,1,1], [0,0,0], [0,1,1], [1,1,1], [0,0,0], [0,1,1], [1,1,1], [0,0,0])
        self.joint_num = len(self.joints_name)
        
        self.gcn = GCN_layer(
            in_features=dims[0], 
            hidden_features=dims[1],
            out_features=dims[-1]
        )
        self.mano_layer = mano_layer
        self.connect = torch.tensor([
            [0, 1], [1, 2], [2, 3], 
            [0, 4], [4, 5], [5, 6],
            [0, 7], [7, 8], [8, 9],
            [0, 10], [10, 11], [11, 12],
            [0, 13], [13, 14], [14, 15]]).view(-1).long()

        assert self.x_mean.shape[1] == d_out
        self.adj = get_adjacency(self.x_mean.long(), self.x_faces.long())
        self.point_num = self.x_mean.shape[0]
        self.latent_code = nn.Parameter(torch.zeros(self.joint_num, d_latent))
        self.mask = nn.Parameter(torch.zeros(self.joint_num, self.point_num))
        self.vertex = nn.Sequential(nn.Conv1d((self.joint_num)*(27+self.d_latent), (self.joint_num)*64, kernel_size=1, groups=self.joint_num),
                                    nn.ReLU(inplace=True), 
                                    nn.Conv1d((self.joint_num)*64, (self.joint_num)*self.point_num*3, kernel_size=1, groups=self.joint_num))
        
    def add_neighbor_joints(self, hand_pose):
        hand_pose = torch.stack([hand_pose[:,self.neighbor_joint_idxs[j],:] for j in range(self.joint_num)],1)
        hand_pose = hand_pose.view(-1,self.joint_num,3,9)
        hand_pose = hand_pose * torch.FloatTensor(self.neighbor_joint_mask).cuda().view(1,self.joint_num,3,1)
        hand_pose = hand_pose.view(-1,self.joint_num,27)
        return hand_pose

    def forward(self, input, input_verts, mask=None):
        
        n_batch, n_dim = input.shape
        pose = input.reshape(n_batch, 16, -1)
        hand_pose = self.add_neighbor_joints(pose)  # [1, 16, 27]
        
        feat = torch.cat((hand_pose, self.latent_code[None,]),2)
        feat = feat.view(n_batch,(self.joint_num)*(27+self.d_latent),1)
        vertex_pose_corrective_per_joint = self.vertex(feat).view(n_batch,self.joint_num,self.point_num,3) / 1000
        
        zeros = torch.zeros((n_batch,self.joint_num,9)).float().cuda()
        zeros = self.add_neighbor_joints(zeros)
        feat = torch.cat((zeros, self.latent_code[None,]),2)
        feat = feat.view(n_batch,(self.joint_num)*(27+self.d_latent),1)
        vertex_pose_corrective_per_joint_zero_pose = self.vertex(feat).view(n_batch,self.joint_num,self.point_num,3) / 1000
        
        vertex_pose_corrective_per_joint = vertex_pose_corrective_per_joint - vertex_pose_corrective_per_joint_zero_pose
        
        mask = F.relu(self.mask).view(n_batch, self.joint_num, self.point_num, 1)
        x_full = input_verts + (vertex_pose_corrective_per_joint * mask).sum(1)
        # input = torch.cat([input.unsqueeze(1).expand(-1, self.point_num, -1), self.latent_code.unsqueeze(0).expand(n_batch, -1, -1)], 2)       
        # if mask is not None:
        #     input = input[mask]
        # input_embed = input if self.embed_fn is None else self.embed_fn(input)

        # x = input_embed[0]
        # weight = self.gcn(x, adj=self.adj, act='sig')
        # x = x.unsqueeze(0)
        
        # add placeholder for masked prediction
        
        return x_full
        

if __name__ == '__main__':
    # x_mean = torch.zeros(49281, 16)
    #
    # net = ConditionNetwork(x_mean, 58, 16, 10, 128, 5)
    # net2 = SimpleNetwork(58, 16,  128, 5)
    #
    # input = torch.zeros(1, 58)
    # out = net(input)
    # out2 = net2(input)
    # print(out.shape)
    # print(out2.shape)

    renderer = MLP_res(6, 3)
    print(renderer(torch.rand(100,6)).shape)