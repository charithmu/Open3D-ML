import argparse
import copy
import os
import os.path as osp
import pprint
import sys
import time
from pathlib import Path

import open3d.ml as _ml3d
import yaml


framework = "tf"
device = "gpu"

randlanet_smartlab_cfg = "/home/charith/repos/Open3D-ML/ml3d/configs/randlanet_smartlab.yml"
randlanet_semantickitti_cfg = "/home/charith/repos/Open3D-ML/ml3d/configs/randlanet_semantickitti.yml"
randlanet_s3dis_cfg = "/home/charith/repos/Open3D-ML/ml3d/configs/randlanet_s3dis.yml"

cfg_file = randlanet_s3dis_cfg


if framework == "torch":
    import open3d.ml.torch as ml3d
else:
    import open3d.ml.tf as ml3d
    import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices("GPU")
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


cfg = _ml3d.utils.Config.load_from_file(cfg_file)

Pipeline = _ml3d.utils.get_module("pipeline", cfg.pipeline.name, framework)
Model = _ml3d.utils.get_module("model", cfg.model.name, framework)
Dataset = _ml3d.utils.get_module("dataset", cfg.dataset.name)

cfg_dict_dataset, cfg_dict_pipeline, cfg_dict_model = _ml3d.utils.Config.merge_cfg_file(
    cfg, args, extra_dict)

dataset = Dataset(**cfg_dict_dataset)

model = Model(**cfg_dict_model)

pipeline = Pipeline(model, dataset, **cfg_dict_pipeline)


pipeline.cfg_tb = {
    "readme": "readme",
    "cmd_line": "cmd_line",
    "dataset": pprint.pformat(cfg_dict_dataset, indent=2),
    "model": pprint.pformat(cfg_dict_model, indent=2),
    "pipeline": pprint.pformat(cfg_dict_pipeline, indent=2),
}

if args.split == "test":
    pipeline.run_test()
else:
    pipeline.run_train()
