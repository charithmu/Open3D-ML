import argparse
import copy
import os
import os.path as osp
import pprint
import sys
import time
from pathlib import Path

import numpy as np
import yaml


class SemSegPredictor:
    def __init__(self):

        self.home_path = str(Path.home())
        self.base_path = self.home_path + "/dev/Open3D-ML"
        self.dateset_path = self.home_path + "/datasets/SmartLab"
        self.ckpt_path = self.base_path + "/mytests/logs/RandLANet_SmartLab_tf/checkpoint/ckpt-6"
        self.randlanet_smartlab_cfg = self.base_path + "/ml3d/configs/randlanet_smartlab.yml"
        self.kpconv_smartlab_cfg = self.base_path + "/ml3d/configs/kpconv_smartlab.yml"

        os.environ["OPEN3D_ML_ROOT"] = base_path
        import open3d.ml as _ml3d

        self._ml3d = _ml3d

        kwargs = {
            "framework": "tf",
            "device": "cuda",
            "dataset_path": datesetbase,
            "split": "test",
            "ckpt_path": ckpt_path,
            "cfg_file": randlanet_smartlab_cfg,
        }

        self.args = type("args", (object,), kwargs)()

        pprint.pprint(kwargs)

        if args.framework == "torch":
            import open3d.ml.torch as ml3d

            self.ml3d = ml3d
        else:
            import open3d.ml.tf as ml3d
            import tensorflow as tf

            self.ml3d = ml3d
            self.tf = tf

    def config_gpu(self):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        print(gpus)

        if gpus:
            try:
                for gpu in gpus:
                    self.tf.config.experimental.set_memory_growth(gpu, True)
                if self.args.device == "cpu":
                    self.tf.config.set_visible_devices([], "GPU")
                elif self.args.device == "cuda":
                    self.tf.config.set_visible_devices(gpus[0], "GPU")
                else:
                    idx = self.args.device.split(":")[1]
                    self.tf.config.set_visible_devices(gpus[int(idx)], "GPU")
            except RuntimeError as e:
                print(e)

    @staticmethod
    def merge_cfg_file(cfg, args):
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

    def setup_lut_colors(self):
        Unlabeled = [231, 87, 36]
        Floor = [188, 169, 26]
        Wall = [100, 244, 245]
        Robot = [150, 30, 140]
        Human = [0, 248, 26]
        AGV = [18, 35, 243]

        colors = {
            "Unlabeled": [x / 255.0 for x in Unlabeled],
            "Floor": [x / 255.0 for x in Floor],
            "Wall": [x / 255.0 for x in Wall],
            "Robot": [x / 255.0 for x in Robot],
            "Human": [x / 255.0 for x in Human],
            "AGV": [x / 255.0 for x in AGV],
        }

        self.colors = colors

    def config_pipeline(self):

        cfg = self._ml3d.utils.Config.load_from_file(self.args.cfg_file)

        Pipeline = self._ml3d.utils.get_module("pipeline", cfg.pipeline.name, self.args.framework)
        Model = self._ml3d.utils.get_module("model", cfg.model.name, self.args.framework)
        Dataset = self._ml3d.utils.get_module("dataset", cfg.dataset.name)

        cfg_dataset, cfg_pipeline, cfg_model = merge_cfg_file(cfg, self.args)

        self.dataset = Dataset(**cfg_dataset)
        self.model = Model(**cfg_model)
        self.pipeline = Pipeline(model, dataset, **cfg_pipeline)

        self.pipeline.cfg_tb = {
            "readme": "readme",
            "cmd_line": "cmd_line",
            "dataset": pprint.pformat(cfg_dataset, indent=2),
            "model": pprint.pformat(cfg_model, indent=2),
            "pipeline": pprint.pformat(cfg_pipeline, indent=2),
        }

    def run_pipeline_inference(dataframe):
        print(dataframe.name)
        results = self.pipeline.run_inference(dataframe.data)
        pred = (results["predict_labels"]).astype(np.int32)
        return pred

    def run_pipeline_traintest():
        if self.args.split == "test":
            self.pipeline.run_test()
        else:
            self.pipeline.run_train()

    def run_pipeline_test():
        self.pipeline.load_ckpt(ckpt_path=self.args.ckpt_path)
        test_split = self.dataset.get_split("test")

        vis_points = []
        times = []
        acc = []

        for idx in range(len(test_split)):
            # for idx in range(5):

            st = time.perf_counter()
            attr = test_split.get_attr(idx)
            data = test_split.get_data(idx)

            print(attr)
            results = self.pipeline.run_inference(data)

            pred = (results["predict_labels"]).astype(np.int32)
            # scores = results['predict_scores']

            label = data["label"]
            pts = data["point"]

            vis_d = {
                "name": attr["name"],
                "points": pts,
                "labels": label,
                "pred": pred,
            }

            vis_points.append(vis_d)
            et = time.perf_counter()
            times.append(et - st)

            correct = (pred == label).sum()
            print(f"Correct: " + str(correct) + " out of " + str(len(label)))
            accuracy = correct / len(label)
            acc.append(accuracy)
            print(f"Accuracy: " + str(accuracy))

        print("\n")
        print(times)
        print(f"Average time {np.mean(times):0.4f} seconds")
        overall_acc = np.asarray(acc).sum() / len(acc)
        print("Overall Accuracy: " + str(overall_acc))

        v = self.ml3d.vis.Visualizer()
        lut = self.ml3d.vis.LabelLUT()

        label_names = self.dataset.get_label_to_names()
        pprint.pprint(label_names)

        for (c, cv), (l, lv) in zip(colors.items(), label_names.items()):
            lut.add_label(lv, l, cv)

        v.set_lut("labels", lut)
        v.set_lut("pred", lut)

        v.visualize(vis_points, width=2600, height=2000)


def main(args):
    ps = pointcloud_segmenter()


if __name__ == "__main__":
    main(sys.argv)
