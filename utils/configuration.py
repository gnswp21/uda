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
    #lr: int = 2e-5                      # lr_scheduled = lr * factor
    # n_epochs: int = 3
    #warmup: float = 0.1                 # warmup steps = total_steps * warmup
    #do_lower_case: bool = True
    mode: str = None                    # train, eval, test
    uda_mode: bool = False              # True, False
    
   #total_steps: int = 100000           # total_steps >= n_epcohs * n_examples / 3
    max_seq_length: int = 128
    train_batch_size: int = 32
    eval_batch_size: int = 8

    # unsup
    unsup_ratio: int = 0                # unsup_batch_size = unsup_ratio * sup_batch_size
    #uda_coeff: int = 1                  # total_loss = sup_loss + uda_coeff*unsup_loss
    #tsa: str = 'linear_schedule'           # log, linear, exp
    #uda_softmax_temp: float = -1        # 0 ~ 1
    #uda_confidence_thresh: float = -1   # 0 ~ 1

    # data
    #data_parallel: bool = True
    need_prepro: bool = False           # is data already preprocessed?
    sup_data_dir: str = None
    unsup_data_dir: str = None
    eval_data_dir: str = None
    #n_sup: int = None
    #n_unsup: int = None

    #model_file: str = None              # fine-tuned
    #pretrain_file: str = None           # pre-trained
    #vocab: str = None
    #task: str = None

    # results
    #save_steps: int = 100
    #check_steps: int = 10
    #results_dir: str = None

    # appendix
    #is_position: bool = False           # appendix not used
    
    @classmethod
    def from_json(cls, file):
         return cls(**json.load(open(file, 'r')))