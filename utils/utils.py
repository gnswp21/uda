import random
import torch
import numpy as np

import logging
from typing import Dict

def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_fig(logs : Dict, path='figure/train_00_only_sum/', save : bool= True):
    import matplotlib.pyplot as plt
    import os

    logging.basicConfig(level=logging.INFO)
    logging.info(f'Save figures at {path}')
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print ('Error: Creating directory. ' +  path)

    for key in logs.keys():
        plt.figure()
        plt.plot(logs[key])
        if save:
            plt.savefig(path + key + '.png', dpi=3000)
        else:
            plt.show()

def save_config(cfg, model_cfg):
    pass
