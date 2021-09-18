import json
from typing import NamedTuple


class params(NamedTuple):
    
    ############################ guide #############################
    # lr(learning rate) : fine_tune(2e-5), futher-train(1.5e-4~2e-5)
    # mode : train, eval, test
    # uda_mode : True, False
    # total_steps : n_epochs * n_examples / 3
    # max_seq_length : 128, 256, 512
    # unsup_ratio : more than 3
    # uda_softmax_temp : more than 0.5
    # uda_confidence_temp : ??
    # tsa : linear_schedule
    ################################################################

    # train    
    seed: int = 1421
    
    lr: int = 2e-5                      # lr_scheduled = lr * factor
   
    mode: str = None                    # train, eval, test
    uda_mode: bool = False              # True, False

    max_seq_length: int = 128
    train_batch_size: int = 32
    eval_batch_size: int = 8

    # unsup
    unsup_ratio: int = 0                # unsup_batch_size = unsup_ratio * sup_batch_size

    # data
    need_prepro: bool = False           # is data already preprocessed?
    sup_data_dir: str = None
    unsup_data_dir: str = None
    eval_data_dir: str = None


    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file,'r')))


class model(NamedTuple):
    # model_name_or_path  : model path
    # num_labels : number of labels
    
    model_name_or_path: str  = None
    num_labels: int = 0
    
    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, 'r')))
