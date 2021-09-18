import torch.nn as nn

class trainer:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg

    def train(self, data_iter, optimizer, device):
        if self.cfg.uda_mode:
            unsup_train_iter, sup_train_iter, sup_test_iter = data_iter
            losses = []
            criterion = nn.KLDivLoss(reduction='batchmean')
            for step, batch in enumerate(data_iter[1]):
                # 텐서로 바꿔준다. 데이터 종류에 따른  dtype을 다르게 한다
                if step == 2:
                    break
                ori_input_ids, ori_input_mask, ori_input_type_ids, \
                aug_input_ids, aug_input_mask, aug_input_type_ids = [t.to(device) for t in batch]

                ori_outputs = self.model(ori_input_ids, ori_input_mask, ori_input_type_ids)
                aug_outputs = self.model(aug_input_ids, aug_input_mask, aug_input_type_ids)   

                loss = criterion(ori_outputs.logits, aug_outputs.logits)
                losses.append(loss)

                # backpropagation
                loss.backward()
                optimizer.step()

                # logging
        
            
            return losses

        else:
            raise NotImplementedError

    
    

    def load(self):
        pass
        
    def save(self):
        pass

