import os
import argparse
import pandas as pd
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--list",
                        default='../../data/COCO14/voc_format/val.txt',
                        type=str)
    parser.add_argument("--out-list",
                        default='../../data/COCO14/voc_format/val_tmp.txt',
                        type=str)
    parser.add_argument("--predict-dir", default="ep50_seg_deit_small_distilled_patch16_224_mask_lr1e-4_WeakTrCOCOPseudoMask/seg_prob_ms", type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.list, names=['filename'], converters={"filename": str})
    name_list = df['filename'].values
    f = open(args.out_list, "w")
    for name in name_list:
        # print(name)
        path = os.path.join(args.predict_dir, name + ".npy")
        if not os.path.isfile(path):
            f.write(name + "\n")