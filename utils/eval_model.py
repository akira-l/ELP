#coding=utf8
from __future__ import print_function, division
import os,time,datetime
import numpy as np
import datetime
from math import ceil

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils.utils import LossRecord

import pdb

class AccRecord(): 
    def __init__(self, ): 
        self.val_cor1 = 0 
        self.val_cor2 = 0 
        self.val_cor3 = 0 

    def update(self, pred, label): 
        top3_val, top3_pos = torch.topk(pred, 3) 
        b_cor1 = torch.sum((top3_pos[:, 0] == label)).data.item() 
        self.val_cor1 += b_cor1
        b_cor2 = torch.sum((top3_pos[:, 1] == label)).data.item() 
        self.val_cor2 += (b_cor1 + b_cor2) 
        b_cor3 = torch.sum((top3_pos[:, 2] == label)).data.item() 
        self.val_cor3 += (b_cor1 + b_cor2 + b_cor3) 

    def return_acc(self, count): 
        return self.val_cor1 / count, self.val_cor2 / count, self.val_cor3 / count
 
def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def eval_turn(model_recieve, data_loader, val_version, epoch_num, config):

    model = model_recieve['base']
    as_model = model_recieve['bank']

    model.train(False)
    as_model.train(False)

    val_size = data_loader.__len__()
    item_count = data_loader.total_item_len
    t0 = time.time()

    get_ce_loss = nn.CrossEntropyLoss()

    val_batch_size = data_loader.batch_size
    val_epoch_step = data_loader.__len__()

    val_loss_recorder = LossRecord(val_batch_size)
    val_celoss_recorder = LossRecord(val_batch_size)

    base_acc = AccRecord()
    as_acc = AccRecord() 

    print('evaluating %s ...'%val_version)
    with torch.no_grad():
        for batch_cnt_val, data_val in enumerate(data_loader):
            # print data
            #inputs,  labels, labels_swap, law_swap, img_name = data_val
            inputs = Variable(data_val[0].cuda())
            labels = Variable(torch.from_numpy(np.array(data_val[1])).long().cuda())
            img_names = data_val[-1]
            # forward
            outputs, cls_feat, avg_feat = model(inputs)
            pred_val, pred_ind = outputs.max(1)

            #tmp_out = as_model.module.sp_Acls(avg_feat)
            #tmp_score = F.softmax(tmp_out.detach(), 1)

            ce_loss = get_ce_loss(outputs, labels).item()

            val_loss_recorder.update(ce_loss)
            val_celoss_recorder.update(ce_loss)

            print('{:s} eval_batch: {:-6d} / {:d} loss: {:8.4f}'.format(val_version, batch_cnt_val, val_epoch_step, ce_loss))

            base_acc.update(outputs, labels)

        val_acc1, val_acc2, val_acc3 = base_acc.return_acc(item_count) 

        t1 = time.time()
        since = t1-t0
        print('\n')
        print('--'*30)
        print('|| eval: % 3d %s %s %s-loss: %.4f ||%s-acc@1: %.4f %s-acc@2: %.4f %s-acc@3: %.4f ||time: %d' % (epoch_num, val_version, dt(), val_version, val_loss_recorder.get_val(init=False), val_version, val_acc1, val_version, val_acc2, val_version, val_acc3, since))
        print('--' * 30)
        print('\n')

    return val_acc1, val_acc2, val_acc3


