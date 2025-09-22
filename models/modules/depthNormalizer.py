import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthNormalizer(nn.Module):
    def __init__(self, loadSize=512, z_size=200):
        super(DepthNormalizer, self).__init__()
        self.loadSize = loadSize
        self.z_size = z_size

    def forward(self, z, calibs=None, index_feat=None):
        '''
        Normalize z_feature
        :param z_feat: [B, 1, N] depth value for z in the image coordinate system
        :return:
        '''
        z_feat = z * (self.loadSize // 2) / self.z_size
        return z_feat
