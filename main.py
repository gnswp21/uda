from utils import configuration
from utils.utils import set_seeds
import trainer
from load_data import load_data

import torch.optim as optim
import torch.nn as nn
import torch

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)


def main(cfg, model_cfg):    
    cfg = configuration.params.from_json(cfg)
    model_cfg = configuration.model.from_json(model_cfg)
    set_seeds(cfg.seed)

    # Load Data & Create Criterion
    data = load_data(cfg)
    if cfg.uda_mode:
        unsup_criterion = nn.KLDivLoss(reduction='none')
        data_iter = [data.sup_data_iter(), data.unsup_data_iter()] if cfg.mode=='train' \
            else [data.sup_data_iter(), data.unsup_data_iter(), data.eval_data_iter()]  # train_eval
    else:
        raise NotImplemented
    
    
    # Load model
    config = AutoConfig.from_pretrained(
        model_cfg.model_name_or_path,
        num_labels=model_cfg.num_labels
    )

    model = AutoModelForSequenceClassification.from_config(config=config)
      
    ## train train.py 
    #trainer = trainer.trainer(model, cfg, model_cfg)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda')
    
    for step, batch in enumerate(data_iter[0]):
        if step == 1:
            break
        input_ids, input_mask, input_type_ids,	label_ids = batch
        logits = model(input_ids, input_mask, input_type_ids)
       
    ## eval
    


if __name__ == '__main__':
    main('config/uda.json', 'config/bert_base.json')
