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

from trainers.srhand_trainer import SRHandTrainer


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
        data_type = conf.get_string('data_type')
        data_name = conf.get_string('data_name')
        capture_name = conf.get_string('capture_name')
        out_dire = './%s_out/%s_%s' % (data_type, capture_name, data_name)
        
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
        os.system("cp %s %s/ih_sfsseq.conf && cp main.py %s && cp trainer.py %s && cp models/mlp.py %s" % (
        conf_path, backup_dir, backup_dir, backup_dir, backup_dir))

    with open('mano/mano_weight_sub3.pkl', 'rb') as f:
        pkl = pickle.load(f)

    model_path = None
    optimizer_path = None
    implicit_path = None
    implicit_optimizer_path = None
    
    data_type = conf.get_string('data_type')
    capture_name = conf.get_string('capture_name')
    data_name = conf.get_string('data_name')
    if args.r:
        exp_name = conf.get_string('exp_name')
        
        pretrain_path = args.model_path if args.model_path else './%s_out/%s_%s/%s/' % (data_type, capture_name, data_name, exp_name)
        
        model_path = join(pretrain_path, 'model.pth')
        optimizer_path = join(pretrain_path, 'optimizer.pth')
        print('Loading model from ', model_path)
        tmp = args.implicit_path if args.implicit_path else join(pretrain_path, 'implicit_model.pth')
        if os.path.exists(tmp):
            implicit_path = tmp
            print('Optimized Implicit Model from %s' % implicit_path)

            tmp = join(pretrain_path, 'implicit_optimizer.pth')
            if tmp: implicit_optimizer_path = tmp
    
    trainer = SRHandTrainer(
        conf, 
        pkl, 
        mano_layer=mano_layer,
        is_continue=args.r,
        model_path=model_path,
        optimizer_path=optimizer_path,
        implicit_path=implicit_path,
        implicit_optimizer_path=implicit_optimizer_path,
    )
    
    trainer.set_log_path(
        out_dire=out_dire,
        out_mesh_dire=out_mesh_dire,     
    )
    
    trainer.prepare_data(data_path, data_type=data_type, data_name=data_name, split='train')
    trainer.initialize_model() 
    
    if args.eval:
        trainer.prepare_data(data_path, data_type=data_type, data_name=data_name, split='test')
        trainer.eval(xhand_path=model_path, implicit_path=implicit_path, save_vis=args.save_vis)
   
    else:
        trainer.pretrain()
        trainer.train()
    trainer.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='confs/demo_sfs.conf')
    parser.add_argument('--eval', action='store_true') 
    parser.add_argument('--r', default=False, action='store_true')
    parser.add_argument('--use_liif', default=False, action='store_true')
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--implicit_path', default=None, type=str)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--save_vis', default=False, action='store_true')
    
    args = parser.parse_args()
    main(args.conf, args.data_path)
