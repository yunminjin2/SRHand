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
import datetime
import smplx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import nvdiffrast.torch as dr
from models.utils import get_normals, laplacian_smoothing, compute_color
from get_data import mano_layer, get_interhand_seqdatabyframe

from repose import lbs, lbs_pose, pose2rot
from trainers.giif_trainer import GiifTrainer

mano_layer = {'right': smplx.create('./', 'mano', use_pca=False, is_rhand=True), 'left': smplx.create('./', 'mano', use_pca=False, is_rhand=False)}

if torch.sum(torch.abs(mano_layer['left'].shapedirs[:,0,:] - mano_layer['right'].shapedirs[:,0,:])) < 1:
    print('Fix shapedirs bug of MANO')
    mano_layer['left'].shapedirs[:,0,:] *= -1



def main(conf_path, data_path=None):
    conf = ConfigFactory.parse_file(conf_path)
    
    if data_path is not None:
        out_path = data_path.split('/')[-1].replace('data', 'out')
        os.makedirs(out_path, exist_ok=True)
        out_mesh_dire = out_path + '/' + conf.get_string('out_mesh_dire')
        input_mesh_dire = out_path + '/' + conf.get_string('input_mesh_dire')
        os.makedirs(out_mesh_dire, exist_ok=True)
    else:
        data_path = conf.get_string('data_path')
        data_name = conf.get_string('data_name')
        capture_name = conf.get_string('capture_name')
        out_dire = './%s_out/%s_%s' % ('interhand', capture_name, data_name)
        
        os.makedirs(out_dire, exist_ok=True)
        try:
            exp_name = conf.get_string('exp_name')
        except:
            exp_name = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        out_mesh_dire = out_dire + '/' + exp_name
        backup_dir = out_mesh_dire + '/backup'
        os.makedirs(out_mesh_dire, exist_ok=True)
        os.makedirs(backup_dir, exist_ok=True)
        # backup key codes
        os.system("cp %s %s/ih_sfsseq.conf &&  cp get_data.py %s" % (
        conf_path, backup_dir, backup_dir))


    trainer = GiifTrainer(       
        conf, 
        mano_layer=mano_layer
    )
    
    trainer.set_log_path(
        out_dire=out_dire,
        out_mesh_dire=out_mesh_dire,     
    )

    if args.eval:
        trainer.prepare_data(data_path, data_name=data_name, split='eval')
        trainer.initialize_model(args.model_path)
        trainer.eval()
        
    else:
        trainer.prepare_data(data_path, data_name=data_name, split='train')
        trainer.prepare_data(data_path, data_name=data_name, split='eval')
        
        trainer.initialize_model(args.model_path)
        trainer.pretrain()

    trainer.finish()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='confs/demo_sfs.conf')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    args = parser.parse_args()
    main(args.conf, args.data_path)
