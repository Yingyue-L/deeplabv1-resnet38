python val_tmp.py --list ../../data/COCO14/voc_format/val_mini.txt --predict-dir ../../model/WeakTrCOCOMask_lr0.0015/val
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=10191 \
train.py --period val_mini
python make_crf.py --predict-dir ../../model/WeakTrCOCOMask/val --list ../../data/COCO14/voc_format/val_100.txt \
--img-path ../../data/COCO14/images --gt-folder ../../data/COCO14/voc_format/class_labels --num-cls 91 --type png