---
title: WeakCLIP230404
date: 2023-04-04 22:09:42
excerpt: 为啥要做？做完后有何收获感想体会？
tags: 
rating: ⭐
status: inprogress
destination: 
share: false
obsidianUIMode: source
---

# code
## VOC12

| Work_dirs                                                    | train mIoU | val  mIoU | +CRF        | +deeplab  |
| ------------------------------------------------------------ | ---------- | --------- | ----------- | --------- |
| voc_denseclip_fpn_vit-b_512x512_20k_pseudo_mask_new_pmm_fix_bn_fix_backbone_cosine_1e4_unfix_layernorm_pyramid_lr_adj/71p56_iter_8000 | 73.84      |           |             |           |
| voc_denseclip_fpn_vit-b_512x512_20k_pseudo_mask_new_pmm_fix_bn_fix_backbone_cosine_1e4_unfix_layernorm_pyramid_64/iter_8000 | 74.47      | 72.82     | 74.93/73.40 | 71.2/72.3 |

>- pseudo_mask: 2022 /data/yingyueli/WeakTr/OnlineRetraining/seg_base_mask_VOC2012_PseudoMaskRefine1CiVitB020 
>- iter:20000


| lr     | val mIoU/+CRF | WeakTr    | +multiscale1.75 |
| ------ | ------------- | --------- | --------------- |
| 0.0006 | 70.5          | 71.6/72.9 |                 |
| 0.0008 | 71.2/72.3     | 72        |                 |
| 0.001  | 71.5/72.4     | 72.5/73.5 |                 |
| 0.001_plus20000  |      |  |                 |
| 0.001_plus40000  |      |  |                 |
| 0.0015 | **71.6**/72.2 | 72.9/74   | 71.3            |
| 0.002  | 71.2          | 72.8/73.6 |                 |
| 0.003  | 68.9          | 71.4/71.8 |                 |
| 0.004  | 66.24         |           |                 |

lr0.001: 15:69.040%
16: 69.230%
17:69.363%
18: 69.375%
last:69.375%


val:76.0

test:76.6

## COCO14

### 生成对应 argmax 之后的 npy

- [x] +CRF / 改 code 到保存对应类别并通过 CRF 生成 PGT

```bash
**PGT generation ViT for COCO**
python val_tmp.py --predict-dir tmp --list /data/zhulianghui/data/COCO14/voc_format/train_part1.txt --out-list /data/zhulianghui/data/COCO14/voc_format/train_part1_tmp.txt
CUDA_VISIBLE_DEVICES=0, PORT=29500 bash dist_test_denseclip.sh configs/fpn_denseclip_voc_dgcn/coco_denseclip_fpn_vit-b_512x512_40k_pseudo_mask_new_pmm_fix_bn_fix_backbone_cosine_1e4_unfix_layernorm_pyramid_64_test_480_part1.py /data/zhulianghui/ClipAbout/DenseCLIP/segmentation/work_dirs/coco_denseclip_fpn_vit-b_512x512_40k_pseudo_mask_new_pmm_fix_bn_fix_backbone_cosine_1e4_unfix_layernorm_pyramid_64/42p58_iter_35500.pth 1 --aug-test --eval "mIoU"

python val_tmp.py --predict-dir tmp --list /data/zhulianghui/data/COCO14/voc_format/train_part5.txt --out-list /data/zhulianghui/data/COCO14/voc_format/train_part5_tmp.txt
CUDA_VISIBLE_DEVICES=0, PORT=29504 bash dist_test_denseclip.sh configs/fpn_denseclip_voc_dgcn/coco_denseclip_fpn_vit-b_512x512_40k_pseudo_mask_new_pmm_fix_bn_fix_backbone_cosine_1e4_unfix_layernorm_pyramid_64_test_480_part5.py /data/zhulianghui/ClipAbout/DenseCLIP/segmentation/work_dirs/coco_denseclip_fpn_vit-b_512x512_40k_pseudo_mask_new_pmm_fix_bn_fix_backbone_cosine_1e4_unfix_layernorm_pyramid_64/42p58_iter_35500.pth 1 --aug-test --eval "mIoU"
python make_crf.py \
--list voc_format/train_100.txt \
--data-path /data/zhulianghui/data/COCO14 \
--predict-dir tmp \
--img-path images \
--gt-folder voc_format/class_labels \
--num-cls 91 --dataset coco \
--type npypng

python make_crf.py \
--list voc_format/train.txt \
--data-path /data/zhulianghui/data/COCO14 \
--predict-dir tmp \
--predict-png-dir tmppng \
--img-path images \
--gt-folder voc_format/class_labels \
--num-cls 91 --dataset coco

python make_crf.py \
--list voc_format/train_part1.txt \
--data-path /data/zhulianghui/data/COCO14 \
--predict-dir tmppng \
--img-path images \
--gt-folder voc_format/class_labels \
--num-cls 91 --dataset coco \
--type png

python val_tmp.py --predict-dir tmp --list /data/zhulianghui/data/COCO14/voc_format/train_part2.txt --out-list /data/zhulianghui/data/COCO14/voc_format/train_part2_tmp.txt
CUDA_VISIBLE_DEVICES=1, PORT=29501 bash dist_test_denseclip.sh configs/fpn_denseclip_voc_dgcn/coco_denseclip_fpn_vit-b_512x512_40k_pseudo_mask_new_pmm_fix_bn_fix_backbone_cosine_1e4_unfix_layernorm_pyramid_64_test_480_part2.py /data/zhulianghui/ClipAbout/DenseCLIP/segmentation/work_dirs/coco_denseclip_fpn_vit-b_512x512_40k_pseudo_mask_new_pmm_fix_bn_fix_backbone_cosine_1e4_unfix_layernorm_pyramid_64/42p58_iter_35500.pth 1 --aug-test --eval "mIoU"

python val_tmp.py --predict-dir tmp --list /data/zhulianghui/data/COCO14/voc_format/train_part6.txt --out-list /data/zhulianghui/data/COCO14/voc_format/train_part6_tmp.txt
CUDA_VISIBLE_DEVICES=1, PORT=29505 bash dist_test_denseclip.sh configs/fpn_denseclip_voc_dgcn/coco_denseclip_fpn_vit-b_512x512_40k_pseudo_mask_new_pmm_fix_bn_fix_backbone_cosine_1e4_unfix_layernorm_pyramid_64_test_480_part6.py /data/zhulianghui/ClipAbout/DenseCLIP/segmentation/work_dirs/coco_denseclip_fpn_vit-b_512x512_40k_pseudo_mask_new_pmm_fix_bn_fix_backbone_cosine_1e4_unfix_layernorm_pyramid_64/42p58_iter_35500.pth 1 --aug-test --eval "mIoU"
  
python val_tmp.py --predict-dir tmp --list /data/zhulianghui/data/COCO14/voc_format/train_part3.txt --out-list /data/zhulianghui/data/COCO14/voc_format/train_part3_tmp.txt
CUDA_VISIBLE_DEVICES=2, PORT=29502 bash dist_test_denseclip.sh configs/fpn_denseclip_voc_dgcn/coco_denseclip_fpn_vit-b_512x512_40k_pseudo_mask_new_pmm_fix_bn_fix_backbone_cosine_1e4_unfix_layernorm_pyramid_64_test_480_part3.py /data/zhulianghui/ClipAbout/DenseCLIP/segmentation/work_dirs/coco_denseclip_fpn_vit-b_512x512_40k_pseudo_mask_new_pmm_fix_bn_fix_backbone_cosine_1e4_unfix_layernorm_pyramid_64/42p58_iter_35500.pth 1 --aug-test --eval "mIoU"

python val_tmp.py --predict-dir tmp --list /data/zhulianghui/data/COCO14/voc_format/train_part7.txt --out-list /data/zhulianghui/data/COCO14/voc_format/train_part7_tmp.txt
CUDA_VISIBLE_DEVICES=2, PORT=29506 bash dist_test_denseclip.sh configs/fpn_denseclip_voc_dgcn/coco_denseclip_fpn_vit-b_512x512_40k_pseudo_mask_new_pmm_fix_bn_fix_backbone_cosine_1e4_unfix_layernorm_pyramid_64_test_480_part7.py /data/zhulianghui/ClipAbout/DenseCLIP/segmentation/work_dirs/coco_denseclip_fpn_vit-b_512x512_40k_pseudo_mask_new_pmm_fix_bn_fix_backbone_cosine_1e4_unfix_layernorm_pyramid_64/42p58_iter_35500.pth 1 --aug-test --eval "mIoU"
  
python val_tmp.py --predict-dir tmp --list /data/zhulianghui/data/COCO14/voc_format/train_part4.txt --out-list /data/zhulianghui/data/COCO14/voc_format/train_part4_tmp.txt
CUDA_VISIBLE_DEVICES=3, PORT=29503 bash dist_test_denseclip.sh configs/fpn_denseclip_voc_dgcn/coco_denseclip_fpn_vit-b_512x512_40k_pseudo_mask_new_pmm_fix_bn_fix_backbone_cosine_1e4_unfix_layernorm_pyramid_64_test_480_part4.py /data/zhulianghui/ClipAbout/DenseCLIP/segmentation/work_dirs/coco_denseclip_fpn_vit-b_512x512_40k_pseudo_mask_new_pmm_fix_bn_fix_backbone_cosine_1e4_unfix_layernorm_pyramid_64/42p58_iter_35500.pth 1 --aug-test --eval "mIoU"

python val_tmp.py --predict-dir tmp --list /data/zhulianghui/data/COCO14/voc_format/train_part8.txt --out-list /data/zhulianghui/data/COCO14/voc_format/train_part8_tmp.txt
CUDA_VISIBLE_DEVICES=3, PORT=29507 bash dist_test_denseclip.sh configs/fpn_denseclip_voc_dgcn/coco_denseclip_fpn_vit-b_512x512_40k_pseudo_mask_new_pmm_fix_bn_fix_backbone_cosine_1e4_unfix_layernorm_pyramid_64_test_480_part8.py /data/zhulianghui/ClipAbout/DenseCLIP/segmentation/work_dirs/coco_denseclip_fpn_vit-b_512x512_40k_pseudo_mask_new_pmm_fix_bn_fix_backbone_cosine_1e4_unfix_layernorm_pyramid_64/42p58_iter_35500.pth 1 --aug-test --eval "mIoU"
```

|      | +MS  | +crf  |
| ---- | ---- | ----- |
| mIoU | 43.8 | 43.8% |

![image-20230406135922108](/Users/liyingyue/Library/Application Support/typora-user-images/image-20230406135922108.png)



### Segmenter Retrain

- [ ] ViT-B Segmenter COCO retrain

```bash
CUDA_VISIBLE_DEVICES=1 MASTER_PORT=12131 DATASET=/data/zhulianghui/data WORK=./ \
python -m segm.train --log-dir seg_vit_base_patch16_384_mask_COCO_weakclip \
--dataset coco --backbone vit_base_patch16_384 --decoder mask_transformer \
--batch-size 4 --epochs 100 -lr 1e-4 --eval-freq 1 \
--ann-dir /data/zhulianghui/ClipAbout/DenceCLIP-copy/segmentation/tmppng \
```

- [ ] ViT-B Segmenter COCO eval

```bash
python val_tmp.py --predict-dir seg_vit_base_patch16_384_mask_COCO_weakclip_ddp/seg_prob_ms \
--list /data/zhulianghui/data/COCO14/voc_format/val.txt --out-list /data/zhulianghui/data/COCO14/voc_format/val_tmp.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. DATASET=/data/zhulianghui/data WORK=. \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=10201 \
segm/eval/miou.py --window-batch-size 1 --multiscale --eval-split voc_format/val_tmp.txt \
--predict-dir seg_vit_base_patch16_384_mask_COCO_weakclip_ddp/seg_prob_ms \
seg_vit_base_patch16_384_mask_COCO_weakclip_ddp/checkpoint_best.pth \
coco

```

|          | Val_5000 | Val   | +crf |
| -------- | -------- | ----- | ---- |
| **mIoU** | 45.0%    | 45.0% |      |
|          |          |       |      |
|          |          |       |      |



### 写作

https://www.overleaf.com/7946248541vtwkhkpktjbh