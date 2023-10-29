# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import numpy as np
import random
torch.manual_seed(1) # cpu
torch.cuda.manual_seed_all(1) #gpu
np.random.seed(1) #numpy
random.seed(1) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import sys
import time

from config import config_dict
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
import torch.optim as optim
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from net.sync_batchnorm.replicate import patch_replication_callback
from utils.configuration import Configuration
from utils.finalprocess import writelog
from utils.imutils import img_denorm
from net.sync_batchnorm import SynchronizedBatchNorm2d
from utils.visualization import generate_vis, max_norm
from tqdm import tqdm

from utils.test_utils import single_gpu_test
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--period", default="val_5000", type=str)
parser.add_argument("--batchsize", default=16, type=int)
parser.add_argument("--lr", default=0.005, type=float)
parser.add_argument("--iter", default=40000, type=int)
parser.add_argument("--eval_step", default=None, type=int)
parser.add_argument("--local_rank", type=int)
parser.add_argument("--resume-from", default=None, type=str)
parser.add_argument("--model-url", default=None, type=str)

args = parser.parse_args()

# distributed
torch.distributed.init_process_group(backend='nccl',init_method='env://')
torch.cuda.set_device(args.local_rank) 
device = torch.device("cuda", args.local_rank)

cfg = Configuration(config_dict)

if args.batchsize is not None:
	cfg.TRAIN_BATCHES = args.batchsize
if args.lr is not None:
	cfg.TRAIN_LR = args.lr
if args.iter is not None:
	cfg.TRAIN_ITERATION = args.iter

cfg.EXP_NAME = cfg.EXP_NAME + '_bs' + str(cfg.TRAIN_BATCHES) + '_lr' + str(cfg.TRAIN_LR) + \
'_iter' + str(cfg.TRAIN_ITERATION)
if args.resume_from is not None:
	cfg.TRAIN_CKPT = args.resume_from
	cfg.EXP_NAME += '_resume' + args.resume_from.split("/")[-2]

if args.model_url is not None:
	cfg.MODEL_BACKBONE_PRETRAIN_URL = args.model_url
	cfg.EXP_NAME = cfg.EXP_NAME + '_pretrain' + args.model_url.split("/")[-1]


cfg.MODEL_SAVE_DIR = os.path.join(cfg.ROOT_DIR, 'model', cfg.EXP_NAME)
cfg.LOG_DIR = os.path.join(cfg.ROOT_DIR, 'log', cfg.EXP_NAME)

if torch.distributed.get_rank() == 0:
	if not os.path.isdir(cfg.LOG_DIR):
		os.makedirs(cfg.LOG_DIR)
	if not os.path.isdir(cfg.MODEL_SAVE_DIR):
		os.makedirs(cfg.MODEL_SAVE_DIR)

	with open(os.path.join(cfg.MODEL_SAVE_DIR, 'config.txt'), 'w') as f:
		for k, v in config_dict.items():
			f.write(k + ':' + str(v) + '\n')

def train_net():
	period = 'train'
	transform = 'weak'
	dataset = generate_dataset(cfg, period=period, transform=transform)
	def worker_init_fn(worker_id):
		np.random.seed(1 + worker_id)
	
	is_distributed = torch.distributed.get_world_size() > 1
	train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=cfg.TRAIN_SHUFFLE) if is_distributed else None
	dataloader = DataLoader(dataset, 
				batch_size=cfg.TRAIN_BATCHES // torch.distributed.get_world_size(), 
				shuffle=cfg.TRAIN_SHUFFLE if train_sampler is None else False,
				sampler=train_sampler,
				num_workers=cfg.DATA_WORKERS,
				pin_memory=True,
				drop_last=True,
				worker_init_fn=worker_init_fn)
	# eval dataset
	eval_dataset = generate_dataset(cfg, period=args.period, transform='none')
	eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False) if is_distributed else None

	eval_dataloader = DataLoader(eval_dataset, 
				batch_size=1, 
				shuffle=False, 
				sampler=eval_sampler,
				num_workers=cfg.DATA_WORKERS,
				worker_init_fn = worker_init_fn)
	
	net = generate_net(cfg, batchnorm=nn.BatchNorm2d)
	if cfg.TRAIN_CKPT:
		net.load_state_dict(torch.load(cfg.TRAIN_CKPT, map_location=device),strict=True)
		print('load pretrained model')
	if cfg.TRAIN_TBLOG:
		from tensorboardX import SummaryWriter
		# Set the Tensorboard logger
		tblogger = SummaryWriter(cfg.LOG_DIR)	


	net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
	net.to(device)		
	net = torch.nn.parallel.DistributedDataParallel(net,
                                                  device_ids=[args.local_rank],
                                                  output_device=args.local_rank, find_unused_parameters=True)
	parameter_source = net.module
	parameter_groups = parameter_source.get_parameter_groups()
	optimizer = optim.SGD(
		# backbone < last conv 
		# weight < bias
		params = [
			{'params': parameter_groups[0], 'lr': cfg.TRAIN_LR, 'weight_decay': cfg.TRAIN_WEIGHT_DECAY},
			{'params': parameter_groups[1], 'lr': args.bias_lr*cfg.TRAIN_LR, 'weight_decay': args.bias_wd * cfg.TRAIN_WEIGHT_DECAY},
			{'params': parameter_groups[2], 'lr': 10*cfg.TRAIN_LR, 'weight_decay': cfg.TRAIN_WEIGHT_DECAY},
			{'params': parameter_groups[3], 'lr': args.bias_lr*10*cfg.TRAIN_LR, 'weight_decay': args.bias_wd * cfg.TRAIN_WEIGHT_DECAY},
		],
		momentum=cfg.TRAIN_MOMENTUM,
		weight_decay=cfg.TRAIN_WEIGHT_DECAY
	)
	itr = cfg.TRAIN_MINEPOCH * len(dataset)//(cfg.TRAIN_BATCHES)
	max_itr = cfg.TRAIN_ITERATION
	max_epoch = max_itr*(cfg.TRAIN_BATCHES)//len(dataset)+1
	tblogger = SummaryWriter(cfg.LOG_DIR)
	criterion = nn.CrossEntropyLoss(ignore_index=255)
	def prepare_func(sample):	
		image_msf = []
		for rate in cfg.TEST_MULTISCALE:
			inputs_batched = sample['image_%f'%rate]
			image_msf.append(inputs_batched)
			if cfg.TEST_FLIP:
				image_msf.append(torch.flip(inputs_batched,[3]))
		return image_msf

	def inference_func(model, img):
		seg = model(img)
		return seg

	def collect_func(result_list, sample):
		[batch, channel, height, width] = sample['image'].size()
		for i in range(len(result_list)):
			result_seg = F.interpolate(result_list[i], (height, width), mode='bilinear', align_corners=True)	
			if cfg.TEST_FLIP and i % 2 == 1:
				result_seg = torch.flip(result_seg, [3])
			result_list[i] = result_seg
		prob_seg = torch.cat(result_list, dim=0)
		prob_seg = F.softmax(torch.mean(prob_seg, dim=0, keepdim=True),dim=1)[0]
		
		result = torch.argmax(prob_seg, dim=0, keepdim=False).cpu().numpy()
		return result

	def save_step_func(result_sample):
		eval_dataset.save_result([result_sample], cfg.EXP_NAME)

	with tqdm(total=max_itr) as pbar:
		for epoch in range(cfg.TRAIN_MINEPOCH, max_epoch):
			if is_distributed:
				train_sampler.set_epoch(epoch)
			for i_batch, sample in enumerate(dataloader):					
				now_lr = adjust_lr(optimizer, itr, max_itr, cfg.TRAIN_LR, cfg.TRAIN_POWER)
				optimizer.zero_grad()

				inputs, seg_label = sample['image'], sample['segmentation']
				n,c,h,w = inputs.size()

				pred1 = net(inputs.to(device))
				loss = criterion(pred1, seg_label.to(device))
				loss.backward()
				optimizer.step()

				torch.cuda.synchronize()

				pbar.set_description("loss=%g " % (loss.item()))
				pbar.update(1)
				time.sleep(0.001)
				#print('epoch:%d/%d\tbatch:%d/%d\titr:%d\tlr:%g\tloss:%g' % 
				#	(epoch, max_epoch, i_batch, len(dataset)//(cfg.TRAIN_BATCHES),
				#	itr+1, now_lr, loss.item()))
				if cfg.TRAIN_TBLOG and itr%100 == 0 and torch.distributed.get_rank() == 0:
					inputs1 = img_denorm(inputs[-1].cpu().numpy()).astype(np.uint8)
					label1 = sample['segmentation'][-1].cpu().numpy()
					label_color1 = dataset.label2colormap(label1).transpose((2,0,1))

					n,c,h,w = inputs.size()
					seg_vis1 = torch.argmax(pred1[-1], dim=0).detach().cpu().numpy()
					seg_color1 = dataset.label2colormap(seg_vis1).transpose((2,0,1))

					tblogger.add_scalar('loss', loss.item(), itr)
					tblogger.add_scalar('lr', now_lr, itr)
					tblogger.add_image('Input', inputs1, itr)
					tblogger.add_image('Label', label_color1, itr)
					tblogger.add_image('SEG1', seg_color1, itr)
				itr += 1
				if itr>=max_itr:
					break
				if args.eval_step is not None and itr % args.eval_step == 0:
					single_gpu_test(net, device, eval_dataloader, prepare_func=prepare_func, inference_func=inference_func, collect_func=collect_func, save_step_func=save_step_func, save_path=cfg.TEST_SAVE)
					if is_distributed:
						torch.distributed.barrier()

					if not is_distributed or torch.distributed.get_rank() == 0:
						resultlog = eval_dataset.do_python_eval(cfg.EXP_NAME)
						for k,v in resultlog.items():
							with open(os.path.join(cfg.MODEL_SAVE_DIR, f"resultlog_{args.period}.txt"), 'a') as f:
								f.write('epoch:%d\titer:%d\t%s:%g\n'%(epoch,itr,k,v))
							tblogger.add_scalar(f'{args.period}_{k}', v, itr)
						print('Test finished')

					net.train()
			if torch.distributed.get_rank() == 0:
				save_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_epoch%d.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,epoch))
				torch.save(parameter_source.state_dict(), save_path)
				print('%s has been saved'%save_path)
			# remove_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_epoch%d.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,epoch-1))
			# if os.path.exists(remove_path):
			# 	os.remove(remove_path)
			single_gpu_test(net, device, eval_dataloader, prepare_func=prepare_func, inference_func=inference_func,collect_func=collect_func, save_step_func=save_step_func, save_path=cfg.TEST_SAVE)
			if is_distributed:
				torch.distributed.barrier()
			if not is_distributed or torch.distributed.get_rank() == 0:
				resultlog = eval_dataset.do_python_eval(cfg.EXP_NAME)
				#eva
				for k,v in resultlog.items():
					with open(os.path.join(cfg.MODEL_SAVE_DIR, f"resultlog_{args.period}.txt"), 'a') as f:
						f.write('epoch:%d\titer:%d\t%s:%g\n'%(epoch,itr,k,v))
					tblogger.add_scalar(f'{args.period}_{k}', v, itr)
				print('Test finished')
			if is_distributed:
				torch.distributed.barrier()
			net.train()
	if not is_distributed or torch.distributed.get_rank() == 0:
		save_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_itr%d_all.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,cfg.TRAIN_ITERATION))
		torch.save(parameter_source.state_dict(),save_path)
		if cfg.TRAIN_TBLOG:
			tblogger.close()
		print('%s has been saved'%save_path)
		writelog(cfg, period)

def adjust_lr(optimizer, itr, max_itr, lr_init, power):
	now_lr = lr_init * (1 - itr/(max_itr+1)) ** power
	optimizer.param_groups[0]['lr'] = now_lr
	optimizer.param_groups[1]['lr'] = 2*now_lr
	optimizer.param_groups[2]['lr'] = 10*now_lr
	optimizer.param_groups[3]['lr'] = 20*now_lr
	return now_lr

def get_params(model, key):
	for m in model.named_modules():
		if key == 'backbone':
			if ('backbone' in m[0]) and isinstance(m[1], (nn.Conv2d, SynchronizedBatchNorm2d, nn.BatchNorm2d, nn.InstanceNorm2d)):
				for p in m[1].parameters():
					yield p
		elif key == 'cls':
			if ('cls_conv' in m[0]) and isinstance(m[1], (nn.Conv2d, SynchronizedBatchNorm2d, nn.BatchNorm2d, nn.InstanceNorm2d)):
				for p in m[1].parameters():
					yield p
		elif key == 'others':
			if ('backbone' not in m[0] and 'cls_conv' not in m[0]) and isinstance(m[1], (nn.Conv2d, SynchronizedBatchNorm2d, nn.BatchNorm2d, nn.InstanceNorm2d)):
				for p in m[1].parameters():
					yield p
if __name__ == '__main__':
	train_net()


