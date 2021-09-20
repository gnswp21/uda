import logging
import matplotlib.pyplot as plt

from utils import configuration
from utils.utils import set_seeds, save_fig
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

    # Logging configs
    logging.info(f'CASE : {cfg.case} ')
    logging.info(f'MODE : {cfg.mode}')
    logging.info(f'RATIO : {cfg.ratio}')

    # Load Data & Create Criterion
    logging.info('Load IMDB data')
    data = load_data(cfg)
    if cfg.uda_mode:
        data_iter = {}
        if 'train' in cfg.mode:
            data_iter['sup_train'] = data.sup_data_iter()
            data_iter['unsup_train'] = data.unsup_data_iter()
        if 'test' in cfg.mode:
            data_iter['sup_test'] = data.test_data_iter()
    else:
        raise NotImplemented
    
    
    # Load model set device
    logging.info(f'Load {model_cfg.model_name_or_path} model')
    config = AutoConfig.from_pretrained(model_cfg.model_name_or_path, num_labels=model_cfg.num_labels)
    model = AutoModelForSequenceClassification.from_config(config=config)
    device = torch.device(cfg.device)
    model.to(device)

    if cfg.mode =='train':
        ## Set Trainer
        UDA_trainer = trainer(model, cfg)
        # could change optimzer
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        # Train using cuda
        losses = UDA_trainer.train(data_iter, optimizer)

        # Save model
        model_path = 'model/'
        logging.info(f'Save model at {model_path}{cfg.case}.pt')
        UDA_trainer.save(path=model_path)
        
        # Save losses figure
        figure_path = 'figure/' + cfg.case + '/'
        save_fig(losses, path=figure_path, save=True)

    elif cfg.mode =='train_test':
        pass
    elif cfg.mode == 'test':
        model_name = cfg.case
        UDA_trainer = trainer(model, cfg)
        UDA_trainer.load(path='model/'+ model_name +'.pt')
        accuracy = UDA_trainer.test(data_iter)

    # END
    logging.info('Well done')


if __name__ == '__main__':
    main('config/uda_re.json', 'config/bert_base.json')
