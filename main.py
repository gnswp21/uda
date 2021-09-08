from utils import configuration
from utils.utils import set_seeds

from load_data import load_data
import torch
import torch.nn as nn



def main(cfg, model_cfg):    
    cfg = configuration.params.from_json(cfg)
    model_cfg = configuration.params.from_json(model_cfg)
    set_seeds(cfg.seed)

    # Load Data & Create Criterion
    data = load_data(cfg)
    if cfg.uda_mode:
        unsup_criterion = nn.KLDivLoss(reduction='none')
        data_iter = [data.sup_data_iter(), data.unsup_data_iter()] if cfg.mode=='train' \
            else [data.sup_data_iter(), data.unsup_data_iter(), data.eval_data_iter()]  # train_eval
    else:
        raise NotImplemented
    


if __name__ == '__main__':
    main('config/uda.json', 'config/bert_base.json')
