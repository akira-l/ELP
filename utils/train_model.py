#coding=utf8
from __future__ import print_function, division

import os,time,datetime
import numpy as np
from math import ceil
import datetime
import random
import gc

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
#from torchvision.utils import make_grid, save_image

import torch.distributed as dist
from utils.utils import LossRecord, clip_gradient, weights_normal_init
from utils.eval_model import eval_turn
from utils.adjust_lr import adjust_learning_rate
#from utils.logger import Logger

import pdb

def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")


def train(Config,
          model_recieve,
          epoch_num,
          start_epoch,
          optimizer_recieve,
          scheduler_recieve,
          data_loader,
          save_dir,
          data_ver='all',
          data_size=448,
          savepoint=500,
          checkpoint=1000
          ):


    if isinstance(model_recieve, dict):
        model = model_recieve['base']
        elp_model = model_recieve['elp']

    if isinstance(optimizer_recieve, dict):
        optimizer = optimizer_recieve['common']
        filter_optim = optimizer_recieve['filter']

    if isinstance(scheduler_recieve, dict):
        exp_lr_scheduler = scheduler_recieve['common']

    step = 0
    eval_train_flag = False
    rec_loss = []
    checkpoint_list = []
    max_record = []

    train_batch_size = data_loader['train'].batch_size
    train_epoch_step = data_loader['train'].__len__()
    train_loss_recorder = LossRecord(train_batch_size)

    #logger = Logger('./tb_logs')

    if savepoint > train_epoch_step:
        savepoint = 1*train_epoch_step
        checkpoint = savepoint

    get_ce_loss = nn.CrossEntropyLoss()

    for epoch in range(1,epoch_num+1):

        #optimizer = adjust_learning_rate(optimizer, epoch)
        model.train(True)
        elp_model.train(True)

        for batch_cnt, data in enumerate(data_loader['train']):
            step += 1
            loss = 0
            model.train(True)
            elp_model.train(True)

            inputs, labels, img_names = data
            inputs = Variable(inputs.cuda())
            labels = Variable(torch.from_numpy(np.array(labels)).cuda())

            optimizer.zero_grad()
            filter_optim.zero_grad()

            outputs, cls_feat, avg_feat = model(inputs, labels, img_names)

            ce_loss = get_ce_loss(outputs, labels)
            mem_out = elp_model(avg_feat.detach()) 
            mem_loss = get_ce_loss(mem_out, labels)

            tmp_out = elp_model.module.sp_Acls(avg_feat.detach())
            tmp_score = F.softmax(tmp_out.detach(), 1)
            main_score = F.softmax(outputs, 1)

            if Config.train_ver == 'sum': 
                alpha = Config.alpha  
                sum_score = alpha * main_score + (1 - alpha) * tmp_score
                div_score = (main_score - 1 + 2*tmp_score) / sum_score 
                div_score = div_score.detach() 

            if Config.train_ver == 'mul': 
                mul_score = main_score * tmp_score
                tmp_sub_score = mul_score - main_score + tmp_score   
                div_score = (mul_score - 1 + tmp_score) / mul_score 
                div_score = div_score.detach() 

            sel_mask = torch.FloatTensor(len(tmp_score), Config.numcls).zero_().cuda()
            sel_mask.scatter_(1, labels.unsqueeze(1), 1.0)
            sel_mask.cuda()

            sel_prob = (div_score * sel_mask).sum(1).view(-1, 1)
            sel_prob = torch.clamp(sel_prob, 1e-8, 1 - 1e-8)

            gamma = Config.gamma 
            mem_focal = - torch.pow(1 - sel_prob, gamma) * main_score.log()
            mem_focal = mem_focal.mean()

            loss = ce_loss + mem_loss + mem_focal 

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(elp_model.module.parameters(), 0.4)
            #torch.nn.utils.clip_grad_norm_(model.module.parameters(), 0.4)

            optimizer.step()
            filter_optim.step()

            print('step: {:-8d} / {:d} loss=ce + elp + elp-sr: {:6.4f} = {:6.4f} + {:6.4f} + {:6.4f}  '.format(step, train_epoch_step,
                                                                                                                   loss.detach().item(),
                                                                                                                   ce_loss.detach().item(),
                                                                                                                   mem_loss.detach().item(),
                                                                                                                   mem_focal.detach().item(),
                                                                                                                   ))
            train_loss_recorder.update(loss.detach().item())

            torch.cuda.synchronize()
            #torch.cuda.empty_cache()

            # evaluation & save
            if step % checkpoint == 0:
                model_dict = {}
                model_dict['base'] = model
                model_dict['bank'] = elp_model
                rec_loss = []
                print(32*'-')
                print('step: {:d} / {:d} global_step: {:8.2f} train_epoch: {:04d} rec_train_loss: {:6.4f}'.format(step, train_epoch_step, 1.0*step/train_epoch_step, epoch, train_loss_recorder.get_val()))
                print('current lr:%s' % exp_lr_scheduler.get_lr())
                if eval_train_flag:
                    trainval_acc1, trainval_acc2, trainval_acc3 = eval_turn(model_dict, data_loader['trainval'], 'trainval', epoch, Config)
                    if abs(trainval_acc1 - trainval_acc3) < 0.01:
                        eval_train_flag = False

                val_acc1, val_acc2, val_acc3 = eval_turn(model_dict, data_loader['val'], 'val', epoch, Config)
                #train_val_acc1, train_val_acc2, train_val_acc3 = eval_turn(model_dict, data_loader['train'], 'train', epoch, Config)


                save_path = os.path.join(save_dir, 'weights__base__%d_%d_%.4f_%.4f.pth'%(epoch, batch_cnt, val_acc1, val_acc3))
                as_save_path = os.path.join(save_dir, 'weights_elp_model_base-pick50__%d_%d_%.4f_%.4f.pth'%(epoch, batch_cnt, val_acc1, val_acc3))
                torch.cuda.synchronize()
                
                #torch.save(model.state_dict(), save_path)
                #if epoch %10 == 0:# and val_acc1 > max(max_record):
                #    torch.save(model.state_dict(), save_path)
                #    torch.save(elp_model.state_dict(), as_save_path)
                
                print('saved model to %s' % (save_path))
                max_record.append(val_acc1)
                torch.cuda.empty_cache()

            # save only
            elif step % savepoint == 0:
                train_loss_recorder.update(rec_loss)
                rec_loss = []
                save_path = os.path.join(save_dir, 'savepoint__base__weights-%d-%s.pth'%(step, dt()))

                checkpoint_list.append(save_path)
                if len(checkpoint_list) == 6:
                    os.remove(checkpoint_list[0])
                    del checkpoint_list[0]
                #torch.save(model.state_dict(), save_path)

                torch.cuda.empty_cache()

        exp_lr_scheduler.step(epoch)

        if epoch % Config.init == 0:
            weights_normal_init(elp_model)

        gc.collect()






