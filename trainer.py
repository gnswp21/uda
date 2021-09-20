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
            #unsup_iter = self.make_cycle_iterator(unsup_train_iter)
            
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
                    ori_outputs = self.model(ori_input_ids, ori_input_mask, ori_input_type_ids)
                    aug_outputs = self.model(aug_input_ids, aug_input_mask, aug_input_type_ids)

                    # make logits LogProbability
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
                    total_loss = unsup_loss + sup_loss
                    losses['total'].append(total_loss)
                    # backpropagation
                    total_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # logging
                    if step % 10 == 0:
                        logging.info(f'Currnent train step: {step}/{int(len(unsup_train_iter) * self.cfg.ratio)}')
                        logging.info(f'Currnent total loss :{total_loss}, sup loss : {sup_loss}, unsup_loss : {unsup_loss}')

            logging.info('Train end')
            
            return losses
        else:
            raise NotImplementedError

    
    
    def eval(self, data_iter):
        ## define metric
        ## Do eval
        pass

    def test(self, data_iter):
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
                logging.info(f'Currnent accuracy : {current_acc/test_num : 6.2f}')
                logging.info(f'Total accuracy : {total_acc/total_test_num : 6.2f}')


        logging.info('Test end')
        logging.info(f'Total accuracy : {total_acc/total_test_num : 6.2f}')
        return total_acc


    def make_cycle_iterator(self, iterator):
        cycle_iter = itertools.cycle(iterator)
        return  cycle_iter
    
    def load(self, path) :
        return self.model.load_state_dict(torch.load(path))
        
    def save(self, path='model/'):
        torch.save(self.model.state_dict(), path + self.cfg.case + '.pt')
        

    

