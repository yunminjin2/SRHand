import torch
import numpy as np


def rotate(points, cam):   
    no_b = False
    if len(points.shape) != 3:
        points = points.unsqueeze(0)
        cam = cam.unsqueeze(0)
        no_b = True
    R = cam[:, :3, :3]
    T = cam[:, :3, 3:4]
    
    pts = torch.baddbmm(T, R, points)

    return pts[0] if no_b else pts

def projection(points, cam_i=None):
    no_b = False
    if len(points.shape) != 3:
        points = points.unsqueeze(0)
        cam_i = cam_i.unsqueeze(0)
        no_b = True
    
    points = torch.bmm(cam_i, points)
    points[:, :2] = points[:, :2] / points[:, 2:]
    return points[0] if no_b else points
    
def rotate_n(points, w2c):   
    no_b = False
    if len(points.shape) != 3:
        points = points.unsqueeze(0)
        w2c = w2c.unsqueeze(0)
        no_b = True
    points = torch.einsum('ijk,ikl->ijl', points, w2c)
    return points[0] if no_b else points

def projection_n(points, proj=None):
    no_b = False
    if len(points.shape) != 3:
        points = points.unsqueeze(0)
        proj = proj.unsqueeze(0)
        no_b = True
    points = torch.einsum('ijk,ikl->ijl', points, proj)
    
    return points[0] if no_b else points
    