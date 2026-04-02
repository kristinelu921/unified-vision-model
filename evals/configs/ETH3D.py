# --------------------------------------------------------
# What Matters When Repurposing Diffusion Models for General Dense Perception Tasks? (https://arxiv.org/abs/2403.06090)
# Github source: https://github.com/aim-uofa/GenPercept
# Copyright (c) 2024, Advanced Intelligent Machines (AIM)
# Licensed under The BSD 2-Clause License [see LICENSE for details]
# Author: Guangkai Xu (https://github.com/guangkaixu/)
# --------------------------------------------------------------------------
# This code is based on Marigold and diffusers codebases
# https://github.com/prs-eth/marigold
# https://github.com/huggingface/diffusers
# --------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/aim-uofa/GenPercept#%EF%B8%8F-citation
# More information about the method can be found at https://github.com/aim-uofa/GenPercept
# --------------------------------------------------------------------------

import torch
import tarfile
import os
import numpy as np

from .base_dataset import BaseDataset
from .benchmark_dataset_configs import ETH3D_CFG, resolve_name_mode


class ETH3DDataset(BaseDataset):
    HEIGHT: int
    WIDTH: int

    def __init__(
        self,
        **kwargs,
    ) -> None:
        cfg = {**ETH3D_CFG, **kwargs}
        self.HEIGHT = int(cfg.pop("depth_binary_height"))
        self.WIDTH = int(cfg.pop("depth_binary_width"))
        cfg.pop("max_depth", None)
        name_mode = resolve_name_mode(cfg.pop("name_mode"))
        super().__init__(
            name_mode=name_mode,
            max_depth=torch.inf,
            **cfg,
        )

    def _read_depth_file(self, rel_path):
        # Read special binary data: https://www.eth3d.net/documentation#format-of-multi-view-data-image-formats
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            binary_data = self.tar_obj.extractfile("./" + rel_path)
            binary_data = binary_data.read()

        else:
            depth_path = os.path.join(self.dataset_dir, rel_path)
            with open(depth_path, "rb") as file:
                binary_data = file.read()
        # Convert the binary data to a numpy array of 32-bit floats
        depth_decoded = np.frombuffer(binary_data, dtype=np.float32).copy()

        depth_decoded[depth_decoded == torch.inf] = 0.0

        depth_decoded = depth_decoded.reshape((self.HEIGHT, self.WIDTH))
        return depth_decoded
