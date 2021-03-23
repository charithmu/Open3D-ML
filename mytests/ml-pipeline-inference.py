
import argparse
import copy
import os
import os.path as osp
import pprint
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import yaml

os.environ["OPEN3D_ML_ROOT"] = "/home/threedee/repos/Open3D-ML"
import open3d.ml as _ml3d

basedir = "/home/threedee/repos/Open3D-ML"
datesetbase = "/home/threedee/datasets"

randlanet_smartlab_cfg = basedir + "/ml3d/configs/randlanet_smartlab.yml"
randlanet_semantickitti_cfg = basedir + \
    "/ml3d/configs/randlanet_semantickitti.yml"
randlanet_s3dis_cfg = basedir + "/ml3d/configs/randlanet_s3dis.yml"

ckpt_path = basedir + "/logs/RandLANet_SmartLab_tf/checkpoint/ckpt-11"

kwargs = {
    "framework": "tf",
    "device": "cuda",
    "dataset_path": datesetbase + "/SmartLab",
    "split": "training",
    "ckpt_path": ckpt_path,
    "cfg_file": randlanet_smartlab_cfg,
}

args = type("args", (object,), kwargs)()

pprint.pprint(kwargs)

if args.framework == "torch":
    import open3d.ml.torch as ml3d
else:
    import open3d.ml.tf as ml3d
    import tensorflow as tf


def merge_cfg_file(cfg, args, extra_dict):
    if args.device is not None:
        cfg.pipeline.device = args.device
        cfg.model.device = args.device
    if args.split is not None:
        cfg.pipeline.split = args.split
    if args.dataset_path is not None:
        cfg.dataset.dataset_path = args.dataset_path
    if args.ckpt_path is not None:
        cfg.model.ckpt_path = args.ckpt_path

    return cfg.dataset, cfg.pipeline, cfg.model


device = args.device
gpus = tf.config.experimental.list_physical_devices("GPU")
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if device == "cpu":
            tf.config.set_visible_devices([], "GPU")
        elif device == "cuda":
            tf.config.set_visible_devices(gpus[0], "GPU")
        else:
            idx = device.split(":")[1]
            tf.config.set_visible_devices(gpus[int(idx)], "GPU")
    except RuntimeError as e:
        print(e)

cfg = _ml3d.utils.Config.load_from_file(args.cfg_file)

Pipeline = _ml3d.utils.get_module(
    "pipeline", cfg.pipeline.name, args.framework)
Model = _ml3d.utils.get_module("model", cfg.model.name, args.framework)
Dataset = _ml3d.utils.get_module("dataset", cfg.dataset.name)

cfg_dataset, cfg_pipeline, cfg_model = merge_cfg_file(cfg, args, None)

dataset = Dataset(**cfg_dataset)
model = Model(**cfg_model)
pipeline = Pipeline(model, dataset, **cfg_pipeline)

test_split = dataset.get_split("test")

vis_points = []
times = []

for idx in range(len(test_split)):

    st = time.perf_counter()
    attr = test_split.get_attr(idx)
    data = test_split.get_data(idx)

    # print(attr)
    results = pipeline.run_inference(data)

    pred = (results['predict_labels']).astype(np.int32)

    label = data['label']
    pts = data['point']

    vis_d = {
        "name": attr['name'],
        "points": pts,
        "labels": label,
        "pred": pred,
    }

    vis_points.append(vis_d)
    et = time.perf_counter()
    times.append(et-st)

print("\n")
print(times)
print(f"Average time {np.mean(times):0.4f} seconds")
