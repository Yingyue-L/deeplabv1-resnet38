import os
import pandas as pd
import numpy as np
from PIL import Image
import argparse

import torch
import torch.nn.functional as F

import joblib
from pathlib import Path

categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']
categories_coco = ['background',
                   'person',
                   'bicycle',
                   'car',
                   'motorcycle',
                   'airplane',
                   'bus',
                   'train',
                   'truck',
                   'boat',
                   'traffic light',
                   'fire hydrant',
                   'street sign',
                   'stop sign',
                   'parking meter',
                   'bench',
                   'bird',
                   'cat',
                   'dog',
                   'horse',
                   'sheep',
                   'cow',
                   'elephant',
                   'bear',
                   'zebra',
                   'giraffe',
                   'hat',
                   'backpack',
                   'umbrella',
                   'shoe',
                   'eye glasses',
                   'handbag',
                   'tie',
                   'suitcase',
                   'frisbee',
                   'skis',
                   'snowboard',
                   'sports ball',
                   'kite',
                   'baseball bat',
                   'baseball glove',
                   'skateboard',
                   'surfboard',
                   'tennis racket',
                   'bottle',
                   'plate',
                   'wine glass',
                   'cup',
                   'fork',
                   'knife',
                   'spoon',
                   'bowl',
                   'banana',
                   'apple',
                   'sandwich',
                   'orange',
                   'broccoli',
                   'carrot',
                   'hot dog',
                   'pizza',
                   'donut',
                   'cake',
                   'chair',
                   'couch',
                   'potted plant',
                   'bed',
                   'mirror',
                   'dining table',
                   'window',
                   'desk',
                   'toilet',
                   'door',
                   'tv',
                   'laptop',
                   'mouse',
                   'remote',
                   'keyboard',
                   'cell phone',
                   'microwave',
                   'oven',
                   'toaster',
                   'sink',
                   'refrigerator',
                   'blender',
                   'book',
                   'clock',
                   'vase',
                   'scissors',
                   'teddy bear',
                   'hair drier',
                   'toothbrush']


def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, n_jobs=10):
    def compare(idx):
        name = name_list[idx]
        predict_file = os.path.join(predict_folder, '%s.png' % name)
        predict = np.array(Image.open(predict_file))  # cv2.imread(predict_file)
        if num_cls == 81:
            predict = predict - 91

        gt_file = os.path.join(gt_folder, '%s.png' % name)
        gt = np.array(Image.open(gt_file))
        cal = gt < 255
        mask = (predict == gt) * cal

        p_list, t_list, tp_list = [0] * num_cls, [0] * num_cls, [0] * num_cls
        for i in range(num_cls):
            p_list[i] += np.sum((predict == i) * cal)
            t_list[i] += np.sum((gt == i) * cal)
            tp_list[i] += np.sum((gt == i) * mask)

        return p_list, t_list, tp_list

    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
        [joblib.delayed(compare)(j) for j in range(len(name_list))]
    )
    p_lists, t_lists, tp_lists = zip(*results)
    TP = [0] * num_cls
    P = [0] * num_cls
    T = [0] * num_cls
    for idx in range(len(name_list)):
        p_list = p_lists[idx]
        t_list = t_lists[idx]
        tp_list = tp_lists[idx]
        for i in range(num_cls):
            TP[i] += tp_list[i]
            P[i] += p_list[i]
            T[i] += t_list[i]

    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = []
    Pred = []
    Recall = []
    for i in range(num_cls):
        IoU.append(TP[i] / (T[i] + P[i] - TP[i]))
        T_TP.append(T[i] / (TP[i]))
        P_TP.append(P[i] / (TP[i]))
        FP_ALL.append((P[i] - TP[i]) / (T[i] + P[i] - TP[i]))
        FN_ALL.append((T[i] - TP[i]) / (T[i] + P[i] - TP[i]))
        Pred.append(TP[i]/P[i])
        Recall.append(TP[i]/T[i])
    loglist = {}
    for i in range(num_cls):
        if num_cls == 21:
            loglist[categories[i]] = IoU[i] * 100
        else:
            loglist[categories_coco[i]] = IoU[i] * 100
    miou = np.nanmean(np.array(IoU))
    loglist['mIoU'] = miou * 100
    fp = np.nanmean(np.array(FP_ALL))
    loglist['FP'] = fp * 100
    fn = np.nanmean(np.array(FN_ALL))
    loglist['FN'] = fn * 100
    prediction = np.nanmean(np.array(Pred))
    recall = np.nanmean(np.array(Recall))
    for i in range(num_cls):
        if num_cls == 21:
            if i % 2 != 1:
                print('%11s:%7.3f%%' % (categories[i], IoU[i] * 100), end='\t')
            else:
                print('%11s:%7.3f%%' % (categories[i], IoU[i] * 100))
        else:
            if i % 2 != 1:
                print('%11s:%7.3f%%' % (categories_coco[i], IoU[i] * 100), end='\t')
            else:
                print('%11s:%7.3f%%' % (categories_coco[i], IoU[i] * 100))
    print('\n======================================================')
    print('%11s:%7.3f%%' % ('mIoU', miou * 100))
    print('\n')
    print(f'FP = {fp * 100}, FN = {fn * 100}')
    print(f'Prediction = {prediction * 100}%, Recall = {recall * 100}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--list",
                        # default='../../data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt',
                        default='../../data/COCO14/voc_format/val.txt',
                        type=str)
    parser.add_argument("--predict-dir",
                        default="WeakTrMask",
                        type=str)
    parser.add_argument("--n-jobs", default=10, type=int)
    parser.add_argument("--gt-dir",
                        # default=None
                        # default="../../data/VOCdevkit/VOC2012/SegmentationClassAug"
                        default="../../data/COCO14/voc_format/class_labels"
                        )
    parser.add_argument("--num-classes", default=91, type=int)

    args = parser.parse_args()

    df = pd.read_csv(args.list, names=['filename'], converters={"filename": str})
    name_list = df['filename'].values
    do_python_eval(args.predict_dir, args.gt_dir, name_list, args.num_classes)
