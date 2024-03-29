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
    'EXP_NAME': 'VOCWeakCLIP_77p16',
    'TRAIN_LR': 0.001,
    'TRAIN_ITERATION': 40000,
    'DATA_NAME': 'VOCDataset',
    'DATA_PSEUDO_GT': 'VOCWeakCLIPPGT_77p16',
    'MODEL_NUM_CLASSES': 21,

    # 'TRAIN_LR': 0.0015,
    # 'TRAIN_ITERATION': 160000,
    # 'EXP_NAME': 'COCOWeakCLIPPGT_48p41',
    # 'DATA_NAME': 'COCODataset',
    # 'DATA_PSEUDO_GT': 'COCOWeakCLIPPGT_48p41',
    # 'MODEL_NUM_CLASSES': 91,

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
    'MODEL_BACKBONE_PRETRAIN_URL': 'ilsvrc-cls_rna-a1_cls1000_ep-0001.params',

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
    'TEST_CRF': False,
    'TEST_BATCHES': 1,
}

config_dict['ROOT_DIR'] = os.path.abspath(os.path.join(os.path.dirname("__file__")))
config_dict['MODEL_SAVE_DIR'] = os.path.join(config_dict['ROOT_DIR'], 'model', config_dict['EXP_NAME'])
config_dict['TRAIN_CKPT'] = None
config_dict['LOG_DIR'] = os.path.join(config_dict['ROOT_DIR'], 'log', config_dict['EXP_NAME'])
config_dict['TEST_CKPT'] = None
config_dict['TEST_SAVE'] = None
sys.path.insert(0, os.path.join(config_dict['ROOT_DIR'], 'lib'))
