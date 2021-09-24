import json
from random import shuffle
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

    # for train
    case :str = "train_00"
    ratio : float = 1.0
    device:'str'='cuda'
    
    # train    
    seed: int = 1421
    
    lr: int = 2e-5                      # lr_scheduled = lr * factor

    mode: str = None                    # train, eval, test
    uda_mode: bool = False
    cosistency_coeff : float = 1.0              # True, False
    temperature : float = 0.4
    beta : float =  0.8
    masking:bool = False
    prediction:bool = False
    

    max_seq_length: int = 128
    train_batch_size: int = 32
    test_batch_size: int = 8
    unsup_batch_size: int = 96
    total_steps: int = 10


    # unsup
    unsup_ratio: int = 0                # unsup_batch_size = unsup_ratio * sup_batch_size

    # data
    need_prepro: bool = False           # is data already preprocessed?
    sup_data_dir: str = None
    unsup_data_dir: str = None
    test_data_dir: str = None


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
