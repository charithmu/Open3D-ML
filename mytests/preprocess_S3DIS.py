from sklearn.neighbors import KDTree
import sys
from os.path import join, exists, dirname, abspath
import numpy as np
import pandas as pd
import os, sys, glob, pickle
import argparse

from helper_ply import write_ply

def convert_pc2ply(anno_path, save_path):
    """
    Convert original dataset files to ply file (each line is XYZRGBL).
    We aggregated all the points from each instance in the room.
    :param anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
    :param save_path: path to save original point clouds (each line is XYZRGBL)
    :return: None
    """
    data_list = []

    for f in glob.glob(join(anno_path, '*.txt')):
        class_name = os.path.basename(f).split('_')[0]
        if class_name not in gt_class:  # note: in some room there is 'staris' class..
            class_name = 'clutter'
        pc = pd.read_csv(f, header=None, delim_whitespace=True).values
        labels = np.ones((pc.shape[0], 1)) * gt_class2label[class_name]
        data_list.append(np.concatenate([pc, labels], 1))  # Nx7

    pc_label = np.concatenate(data_list, 0)
    xyz_min = np.amin(pc_label, axis=0)[0:3]
    pc_label[:, 0:3] -= xyz_min

    xyz = pc_label[:, :3].astype(np.float32)
    colors = pc_label[:, 3:6].astype(np.uint8)
    labels = pc_label[:, 6].astype(np.uint8)
    write_ply(save_path, (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])


basedir = sys.argv[1]
print(sys.argv)


dataset_path = basedir + '/S3DIS/'
anno_paths = [line.rstrip() for line in open(dataset_path + 'anno_paths.txt')]
anno_paths = [join(dataset_path, p) for p in anno_paths]

gt_class = [x.rstrip() for x in open(dataset_path + 'class_names.txt')]
gt_class2label = {cls: i for i, cls in enumerate(gt_class)}

original_pc_folder = join(dirname(dataset_path), 'original_ply')
os.mkdir(original_pc_folder) if not exists(original_pc_folder) else None

out_format = '.ply'

for annotation_path in anno_paths:
        print(annotation_path)
        elements = str(annotation_path).split('/')
        out_file_name = elements[-3] + '_' + elements[-2] + out_format
        convert_pc2ply(annotation_path, join(original_pc_folder, out_file_name))