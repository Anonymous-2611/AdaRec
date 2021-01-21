import logging
import os
import random
from pathlib import Path
from time import strftime, localtime

import numpy as np
import torch


def setup_seed(SEED, setup_pytorch=True):
    logging.info(f"Using seed: {SEED}")

    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    if setup_pytorch:
        torch.random.manual_seed(SEED)


def setup_gpu(gpu_s):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_s)


def setup_env(gpu_s, seed):
    setup_gpu(gpu_s)
    setup_seed(seed)


def ensure_folder(folder):
    folder = Path(folder)
    if not folder.is_dir():
        folder.mkdir(parents=True)


def setup_folder(store_root, project_name):
    root = Path(store_root)

    if not root.is_dir():
        print("Root folder of store is created: {}".format(root))
        root.mkdir(parents=True)

    folder_name = project_name + strftime("-%m.%d-%H.%M.%S", localtime())
    full_path = root.joinpath(folder_name)

    if full_path.is_dir():
        raise ValueError("Folder with name `{}` already exists.".format(full_path))
    full_path.mkdir()
    return full_path
