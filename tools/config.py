# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
import torch
import argparse
import os
import sys
import cv2
import time

config_dict = {
    # 'EXP_NAME': 'VOCWeakCLIP',
    # 'TRAIN_LR': 0.001,
    # 'TRAIN_ITERATION': 40000,
    # 'DATA_NAME': 'VOCDataset',
    # 'DATA_PSEUDO_GT': 'VOCWeakCLIPPGT',
    # 'MODEL_NUM_CLASSES': 21,

    'TRAIN_LR': 0.0015,
    'TRAIN_ITERATION': 40000,
    'EXP_NAME': 'COCOWeakCLIP_plusitr40000_plusitr40000',
    'DATA_NAME': 'COCODataset',
    'DATA_PSEUDO_GT': 'COCOWeakCLIPPGT',
    'MODEL_NUM_CLASSES': 91,

    'GPUS': 2,



    'DATA_YEAR': 2012,
    'DATA_AUG': True,
    'DATA_WORKERS': 4,
    'DATA_MEAN': [0.485, 0.456, 0.406],
    'DATA_STD': [0.229, 0.224, 0.225],
    'DATA_RANDOMCROP': 448,
    'DATA_RANDOMSCALE': [0.5, 1.5],
    'DATA_RANDOM_H': 10,
    'DATA_RANDOM_S': 10,
    'DATA_RANDOM_V': 10,
    'DATA_RANDOMFLIP': 0.5,

    
    'MODEL_NAME': 'deeplabv1',
    'MODEL_BACKBONE': 'resnet38',
    'MODEL_BACKBONE_PRETRAIN': True,

    'MODEL_FREEZEBN': False,


    'TRAIN_MOMENTUM': 0.9,
    'TRAIN_WEIGHT_DECAY': 0.0005,
    'TRAIN_BN_MOM': 0.0003,
    'TRAIN_POWER': 0.9,
    'TRAIN_BATCHES': 10,
    'TRAIN_SHUFFLE': True,
    'TRAIN_MINEPOCH': 0,
    'TRAIN_MINBATCHES':10,
    'TRAIN_TBLOG': True,

    'TEST_MULTISCALE': [1.0],
    'TEST_FLIP': False,
    # 'TEST_MULTISCALE': [0.5, 0.75, 1.0, 1.25, 1.5],
    # 'TEST_FLIP': True,
    'TEST_CRF': False,
    'TEST_BATCHES': 1,
}

config_dict['ROOT_DIR'] = os.path.abspath(os.path.join(os.path.dirname("__file__")))
config_dict['MODEL_SAVE_DIR'] = os.path.join(config_dict['ROOT_DIR'], 'model', config_dict['EXP_NAME'])
config_dict['TRAIN_CKPT'] = os.path.join(config_dict['ROOT_DIR'], f'model/COCOWeakCLIP_plusitr40000'
                                                                 '/deeplabv1_resnet38_COCODataset_itr40000_all.pth')
config_dict['LOG_DIR'] = os.path.join(config_dict['ROOT_DIR'], 'log', config_dict['EXP_NAME'])
config_dict['TEST_CKPT'] = os.path.join(config_dict['ROOT_DIR'], f'model/{config_dict["EXP_NAME"]}'
                                                                 '/deeplabv1_resnet38_COCODataset_itr40000_all.pth')
config_dict['TEST_SAVE'] = None
# config_dict['TEST_SAVE'] = os.path.join(config_dict['ROOT_DIR'], f'prob_npy')
sys.path.insert(0, os.path.join(config_dict['ROOT_DIR'], 'lib'))
