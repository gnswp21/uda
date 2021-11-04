import logging

from utils import configuration
from utils.utils import *
from trainer import trainer
from load_data import load_data

import torch.optim as optim
import torch

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)


def main(cfg, model_cfg):    
    logging.basicConfig(level=logging.INFO)
    logging.info('RUN main.py')
    
    cfgs = (cfg, model_cfg)
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
            data_iter['sup_valid'] = data.valid_data_iter()
        if 'test' in cfg.mode:
            data_iter['sup_test'] = data.test_data_iter()
    else:
        raise NotImplemented
    
    
    # Load model set device
    logging.info(f'Load {model_cfg.model_name_or_path} model')
    device = torch.device(cfg.device)
    config = AutoConfig.from_pretrained(model_cfg.model_name_or_path, num_labels=model_cfg.num_labels)
    model = AutoModelForSequenceClassification.from_config(config=config)
    model.to(device)
    
    if 'train' in cfg.mode:
        ## Set Trainer
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        UDA_trainer = trainer(model, cfg, optimizer)
        # Train using cuda
        losses = UDA_trainer.train(data_iter)

        # Save model
        model_path = 'results/'+cfg.case+'/model/'
        logging.info(f'Save model at {model_path}')
        save_model(model, cfg=cfg, path=model_path)

        ## Save losses figure
        figure_path = f'results/{cfg.case}/figure/'
        logging.info(f'Save model at {figure_path}') 
        save_fig(losses, path=figure_path, save=True)


    if 'test' in cfg.mode:
        model_path = 'results/'+cfg.case+'/model/'
        load_model(model, cfg, path=model_path)
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        UDA_trainer = trainer(model, cfg, optimizer)
        accuracy = UDA_trainer.test(data_iter)
        file = 'results/'+cfg.case+'accuracy.txt'
        with open(file, 'w', encoding='utf-8') as f:
            f.write(str(accuracy))

    # Save cfgs
    cfg_path = 'results/'+cfg.case+'/cfg/'
    logging.info(f'Save model at {cfg_path}')
    save_cfg(cfg_path, cfgs)
    
    # END    
    logging.info('Well done')

if __name__ == '__main__':
    main('config/uda_re.json', 'config/bert_base.json')
