import logging

from utils import configuration
from utils.utils import set_seeds
from trainer import trainer
from load_data import load_data

import torch.optim as optim
import torch.nn as nn
import torch

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)


def main(cfg, model_cfg):    
    logging.basicConfig(level=logging.INFO)
    logging.info('RUN main.py')
    cfg = configuration.params.from_json(cfg)
    model_cfg = configuration.model.from_json(model_cfg)
    set_seeds(cfg.seed)

    # Load Data & Create Criterion
    logging.info('Load IMDB data')
    data = load_data(cfg)
    if cfg.uda_mode:
        unsup_criterion = nn.KLDivLoss(reduction='none')
        data_iter = [data.sup_data_iter(), data.unsup_data_iter()] if cfg.mode=='train' \
            else [data.sup_data_iter(), data.unsup_data_iter(), data.eval_data_iter()]  # train_eval
    else:
        raise NotImplemented
    
    
    # Load model
    logging.info(f'Load {model_cfg.model_name_or_path} model')
    config = AutoConfig.from_pretrained(model_cfg.model_name_or_path, num_labels=model_cfg.num_labels)
    model = AutoModelForSequenceClassification.from_config(config=config)

    ## train train.py 
    UDA_trainer = trainer(model, cfg)
    # could change optimzer
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    # train cpu mode
    logging.info('Train start')
    losses = UDA_trainer.train(data_iter, optimizer, device=torch.device('cpu'))
    logging.info('well done')

# 

    ## eval
    


if __name__ == '__main__':
    main('config/uda.json', 'config/bert_base.json')
