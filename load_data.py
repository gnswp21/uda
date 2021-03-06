import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import ast

class IMDB(Dataset):
    labels = ('0', '1')
    def __init__(self, file, need_prepro, max_len, mode, d_type):
        super().__init__()

        if need_prepro:
            raise NotImplemented
        else:
            f = open(file, 'r', encoding='utf-8')
            data = pd.read_csv(f, sep='\t')

            if d_type == 'sup':
                columns = ['input_ids', 'input_mask', 'input_type_ids', 'label_ids']
                self.tensors = [torch.tensor(data[c].apply(lambda x: ast.literal_eval(x)), dtype=torch.long) \
                                    for c in columns[:-1]]
                self.tensors.append(torch.tensor(data[columns[-1]], dtype=torch.long))

            elif d_type =='unsup':
                columns = ['ori_input_ids', 'ori_input_type_ids', 'ori_input_mask',
                             'aug_input_ids', 'aug_input_type_ids', 'aug_input_mask']
                self.tensors = [torch.tensor(data[c].apply(lambda x: ast.literal_eval(x)), dtype=torch.long) \
                                    for c in columns]
            else:
                raise "d_type error"

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)



class load_data:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = IMDB

        if cfg.need_prepro:
            raise NotImplemented

        if 'train' in cfg.mode:
            self.sup_data_dir = cfg.sup_data_dir
            self.sup_batch_size = cfg.train_batch_size
            self.shuffle = True

            self.valid_data_dir = cfg.valid_data_dir            
            self.valid_batch_size = cfg.test_batch_size
        
        if 'test' in cfg.mode:
            self.test_data_dir = cfg.test_data_dir
            self.test_batch_size = cfg.test_batch_size

        if 'eval' in cfg.mode:
            raise NotImplemented
        
        if cfg.uda_mode:
            self.unsup_data_dir = cfg.unsup_data_dir
            self.unsup_batch_size =cfg.unsup_batch_size
    
    def sup_data_iter(self):
        sup_dataset = self.dataset(self.sup_data_dir, self.cfg.need_prepro, self.cfg.max_seq_length, self.cfg.mode, 'sup')
        sup_data_iter = DataLoader(sup_dataset, batch_size = self.sup_batch_size, shuffle= self.shuffle)
        
        return sup_data_iter

    def unsup_data_iter(self):
        unsup_dataset = self.dataset(self.unsup_data_dir, self.cfg.need_prepro, self.cfg.max_seq_length, self.cfg.mode, 'unsup')
        unsup_data_iter = DataLoader(unsup_dataset, batch_size=self.unsup_batch_size, shuffle=self.shuffle)
        
        return unsup_data_iter
    
    def valid_data_iter(self):
        valid_dataset = self.dataset(self.valid_data_dir, self.cfg.need_prepro, self.cfg.max_seq_length, self.cfg.mode, 'sup')
        valid_data_iter = DataLoader(valid_dataset, batch_size = self.valid_batch_size, shuffle=self.shuffle)
        
        return valid_data_iter

    def test_data_iter(self):
        test_dataset = self.dataset(self.test_data_dir, self.cfg.need_prepro, self.cfg.max_seq_length, 'test', 'sup')
        test_data_iter = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=True)

        return test_data_iter
