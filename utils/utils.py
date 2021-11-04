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


def save_fig(logs : Dict, path='results/figure/train_00/', save : bool= True):
    import matplotlib.pyplot as plt
    import os

    logging.basicConfig(level=logging.INFO)

    for key in logs.keys():
        plt.figure()
        plt.plot(logs[key])
        if save:
            logging.info(f'Save figures at {path}')
            try:
                if not os.path.exists(path):
                    os.makedirs(path)
            except OSError:
                print ('Error: Creating directory. ' +  path)
            plt.savefig(path + key + '.png', dpi=3000)
            
        else:
            plt.show()

def load_model(model, cfg, path='results/case00/model/'):
    model.load_state_dict(torch.load(path+cfg.case+'.pt'))
        
def save_model(model, cfg, path='model/'):
    import os
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Save model at {path}')
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print ('Error: Creating directory. ' +  path)
    torch.save(model.state_dict(), path+cfg.case+'.pt')


def save_cfg(path, cfgs):
    import os
    from shutil import copyfile

    logging.basicConfig(level=logging.INFO)
    logging.info(f'Save configs at {path}')
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print ('Error: Creating directory. ' +  path)

    for cfg in cfgs:
        copyfile(cfg, path + cfg[7:])

