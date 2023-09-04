# DeeplabV1 ResNet38 Retrain

First download the ResNet38 pretrain from [here](https://drive.google.com/file/d/16Ij5lqBExoZT7ijERiBBJYZh5qMDT2Z1/view?usp=share_link).

## Training
Please modify the configration in [config.py](tools/config.py) according to your device firstly.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 tools/train.py 
```

## Test

Don't forget to check test configration in [config.py](tools/config.py) first.

### VOC12

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 tools/test.py --period val --test_ckpt [checkpoint path] --test_flip --test_multiscale --test_save prob_npy
```

### COCO14
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 tools/test.py --period val --test_ckpt [checkpoint path] --test_flip --test_multiscale --test_save prob_npy
```

## CRF Postprocess
### VOC12
```bash
python tools/make_crf.py \
--list val.txt \
--data-path data \
--predict-dir prob_npy \
--predict-png-dir pred_png \
--num-cls 21 \
--dataset voc12
```

### COCO14
```bash
python tools/make_crf.py \
--list val.txt \
--data-path data \
--predict-dir prob_npy \
--predict-png-dir pred_png \
--num-cls 91 \
--dataset coco
```