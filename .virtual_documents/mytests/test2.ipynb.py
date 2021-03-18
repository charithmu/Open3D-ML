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
from open3d.ml.datasets import (
    S3DIS,
    ParisLille3D,
    Semantic3D,
    SemanticKITTI,
    SmartLab,
    Toronto3D,
)
from open3d.ml.tf.models import RandLANet
from open3d.ml.tf.pipelines import SemanticSegmentation
from open3d.ml.utils import Config, get_module


import open3d.ml.tf as ml3d  # or open3d.ml.tf as ml3d

randlanet_smartlab_cfg = (
    "/home/charith/repos/Open3D-ML/ml3d/configs/randlanet_smartlab.yml"
)
randlanet_semantickitti_cfg = (
    "/home/charith/repos/Open3D-ML/ml3d/configs/randlanet_semantickitti.yml"
)
randlanet_s3dis_cfg = "/home/charith/repos/Open3D-ML/ml3d/configs/randlanet_s3dis.yml"

cfg = _ml3d.utils.Config.load_from_file(randlanet_smartlab_cfg)

# construct a dataset by specifying dataset_path
dataset = ml3d.datasets.SmartLab(**cfg.dataset)

# get the 'all' split that combines training, validation and test set
all_split = dataset.get_split("training")

# print the attributes of the first datum
print(all_split.get_attr(0))

# print the shape of the first point cloud
print(all_split.get_data(0)["point"].shape)

# show the first 100 frames using the visualizer
vis = ml3d.vis.Visualizer()
vis.visualize_dataset(dataset, "training")  # , indices=range(100)


all_split.get_data(0)

for val in dataset.label_to_names.values():
    print(val)


dataset = S3DIS("/home/charith/datasets/S3DIS/", use_cache=True)

model = RandLANet(dim_input=6, dim_feature=8)

pipeline = SemanticSegmentation(model=model, dataset=dataset, max_epoch=100)


with open("scripts/README.md", "r") as f:
    readme = f.read()

pipeline.cfg_tb = {
    "readme": readme,
    "cmd_line": " ".join(sys.argv[:]),
    "dataset": pprint.pformat("S3DIS", indent=2),
    "model": pprint.pformat("RandLANet", indent=2),
    "pipeline": pprint.pformat("SemanticSegmentation", indent=2),
}

pipeline.run_train()





    # Inference and test example
    from open3d.ml.tf.models import RandLANet
    from open3d.ml.tf.pipelines import SemanticSegmentation

    Pipeline = get_module("pipeline", "SemanticSegmentation", "tf")
    Model = get_module("model", "RandLANet", "tf")
    Dataset = get_module("dataset", "SemanticKITTI")

    RandLANet = Model(ckpt_path=args.path_ckpt_randlanet)

    # Initialize by specifying config file path
    SemanticKITTI = Dataset(args.path_semantickitti, use_cache=False)

    pipeline = Pipeline(model=RandLANet, dataset=SemanticKITTI)

    # inference
    # get data
    train_split = SemanticKITTI.get_split("train")
    data = train_split.get_data(0)
    # restore weights

    # run inference
    results = pipeline.run_inference(data)
    print(results)

    # test
    pipeline.run_test()
