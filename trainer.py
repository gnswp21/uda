from numpy.lib.function_base import percentile
import torch.nn as nn
import itertools
import logging
import torch



class trainer:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        logging.basicConfig(level=logging.INFO)
        if self.cfg.uda_mode:
            self.prev_model = model


    def train(self, data_iter, optimizer):
        if self.cfg.uda_mode:
            device = self.cfg.device
            logging.info('Train start')

            sup_train_iter, unsup_train_iter = data_iter['sup_train'], data_iter['unsup_train']
            
            losses = {'total' : [], 'sup' : [], 'unsup' : []}
            
            # setting criterion & LogSoftMax for KLDivLoss
            unsup_criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)
            LSM = nn.LogSoftmax(dim=1)
            sup_criterion = nn.CrossEntropyLoss(reduction='mean')

            # make iter infinite(cycle)
            sup_iter = self.make_cycle_iterator(sup_train_iter)
            
            # 1 epochs
            epochs = 1
            self.model.train()
            for n in range(epochs):
                for step, batch in enumerate(unsup_train_iter):
                    # end 조건
                    if step > self.cfg.ratio * len(unsup_train_iter):
                        break

                    # unsup data를 device에 담는다
                    ori_input_ids, ori_input_mask, ori_input_type_ids, \
                    aug_input_ids, aug_input_mask, aug_input_type_ids = (t.to(device) for t in batch)

                    # inputs에 따른 outputs을 낸다
                    # ori_outputs은 고정된 모델로 할 수 있게끔 수정할 것
                    with torch.no_grad():
                        ori_outputs = self.model(ori_input_ids, ori_input_mask, ori_input_type_ids)
                    
                    
                    aug_outputs = self.model(aug_input_ids, aug_input_mask, aug_input_type_ids)
                    
                    ## Will be implemented
                    ## case 수정해야함
                    if self.cfg.masking and self.cfg.sharpening:
                        # confidence-based masking
                        use_ids, maxPs = self.confidence_based_masking(ori_outputs.logits, beta=self.cfg.beta)
                        
                        if use_ids.nelement() == 0:
                            # 사용할 unsup data가 없는 경우
                            unsup_loss = 0
                            logging.info(f'[NO USING UNSUP DATA] Currnent train step: {step}/{int(len(unsup_train_iter) * self.cfg.ratio)}')
                            logging.info(f'[NO USING UNSUP DATA] ori max Probs: {torch.max(maxPs)}')
                        else:
                            # make logits LogProbability
                            # sharpening prediction and masking
                            ori_logP = self.sharpening_prediction(ori_outputs.logits, temperature=self.cfg.temperature)
                            aug_logP = LSM(aug_outputs.logits)
                            ori_logP, aug_logP = ori_logP[use_ids], aug_logP[use_ids]                    
                            unsup_loss = unsup_criterion(ori_logP, aug_logP)
                        losses['unsup'].append(unsup_loss)
                    else:
                        ori_logP = LSM(ori_outputs.logits)
                        aug_logP = LSM(aug_outputs.logits)
                        unsup_loss = unsup_criterion(ori_logP, aug_logP)
                        losses['unsup'].append(unsup_loss)

                    #  sup data를 device에 담는다.
                    sup_input_ids, sup_input_mask, sup_input_type_ids, label_ids = (t.to(device) for t in next(sup_iter))

                    # inputs에 따른 outputs을 낸다
                    sup_outputs = self.model(sup_input_ids, sup_input_mask, sup_input_type_ids)

                    sup_loss = sup_criterion(sup_outputs.logits, label_ids)
                    losses['sup'].append(sup_loss)

                    # 두 종류의 loss를 더 한다.
                    # cositency_coeff에 따라 조합의 정도가 달라진다.
                    loss = sup_loss + self.cfg.cosistency_coeff* unsup_loss 
                    losses['total'].append(loss)
                    # backpropagation
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # logging
                    if step % 10 == 0:
                        logging.info(f'Currnent train step: {step}/{int(len(unsup_train_iter) * self.cfg.ratio)}')
                        logging.info(f'Currnent total loss :{loss}, sup loss : {sup_loss}, unsup_loss : {unsup_loss}')

            logging.info('Train end')
            
            return losses
        else:
            raise NotImplementedError

    
    
    def eval(self, data_iter):
        ## define metric
        ## Do eval
        pass

    def test(self, data_iter, show_loss:bool = False):
        ## define metric
        sup_test_iter = data_iter['sup_test']
        
        # setting metric
        '''
        metric을 acc, f1-score ,etc...  
        다양하게 정의할 수 있게끔 수정
        '''
        metric = 'acc'
        device = self.cfg.device
        logging.info('Test start')
        total_acc = 0
        total_test_num = 0

        self.model.eval()
        with torch.no_grad():
            
            for step, batch in enumerate(sup_test_iter):
                # end 조건
                if step > self.cfg.ratio * len(sup_test_iter):
                    break


                #  sup data를 device에 담는다.
                sup_input_ids, sup_input_mask, sup_input_type_ids, label_ids = (t.to(device) for t in batch)

                # inputs에 따른 outputs을 낸다
                sup_outputs = self.model(sup_input_ids, sup_input_mask, sup_input_type_ids)
                predictions = torch.argmax(sup_outputs.logits, dim = 1)
                
                # for logging
                current_acc = torch.sum(predictions == label_ids, dim = 0)
                test_num = len(label_ids)                        
                total_acc += current_acc
                total_test_num += test_num

                # logging
                if step % 10 == 0:
                    logging.info(f'pred : {predictions} labels : {label_ids}')
                    logging.info(f'Currnent test step: {step}/{int(len(sup_test_iter) * self.cfg.ratio)}')
                    logging.info(f'Currnent accuracy : {current_acc}/{test_num}:{current_acc/test_num : 6.2f}')
                    logging.info(f'Total accuracy : {total_acc}/{total_test_num}:{total_acc/total_test_num : 6.2f}')


            logging.info('Test end')
            logging.info(f'Total accuracy : {total_acc/total_test_num : 6.2f}')
        return total_acc

    def sharpening_prediction(self, i : torch.Tensor, temperature:float = 0.4, use_log:bool = True):
        i = i / temperature
        if use_log:
            f = torch.nn.LogSoftmax(dim=1)
        else:
            f = torch.nn.Softmax(dim=1)
        return f(i)

    def confidence_based_masking(self, x: torch.Tensor, beta:float = 0.8, use_log=False):
        import math

        if use_log:
            f = torch.nn.LogSoftmax(dim=1)
            y = f(x)
            maxPs, _ = torch.max(y, dim=1)
            use_id = torch.nonzero(maxPs > math.log(beta))
        else:
            f = torch.nn.Softmax(dim=1)
            y = f(x)
            maxPs, _ = torch.max(y, dim=1)
            use_id = torch.nonzero(maxPs > beta)
        return use_id, maxPs
        
    
    def make_cycle_iterator(self, iterator):
        cycle_iter = itertools.cycle(iterator)
        return  cycle_iter
    
    def load(self, path) :
        return self.model.load_state_dict(torch.load(path))
        
    def save(self, path='model/'):
        torch.save(self.model.state_dict(), path + self.cfg.case + '.pt')

    
        

    

