from pathlib import Path

import torch
import numpy as np


def load_state_dict(path, device=None):
    if device is None:
        map_fn = torch.device("cpu")
    else:
        map_fn = lambda storage, loc: storage.cuda(device)
    dumped_state_dict = torch.load(Path(path), map_location=map_fn)
    return dumped_state_dict


def np_softmax(x):
    e_x = np.exp(x - np.max(x))
    p = e_x / e_x.sum(axis=0)
    return p


def flatten_2d(list_2d):
    return [item for sublist in list_2d for item in sublist]

