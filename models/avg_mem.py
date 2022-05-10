import os
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torchvision import models, transforms, datasets
#import pretrainedmodels

from utils.rhm_map import rhm, rhm_single

import pdb


class AvgMem(nn.Module):
    def __init__(self, config, momentum, choose):
        super(AvgMem, self).__init__()

        self.num_classes = config.numcls
        self.dim = config.bank_dim
        self.choose = choose

        self.momentum = momentum
        self.feat_bank = torch.zeros(self.num_classes, self.dim)
        self.bank_confidence = torch.zeros(self.num_classes)

        self.update_feat_bank = torch.zeros(self.num_classes, self.dim)
        self.update_times = torch.zeros(self.num_classes)

        self.softmax = nn.Softmax(dim=1)



    def update_bank(self):
        self.feat_bank = torch.pow(self.momentum, self.update_times).unsqueeze(1) * self.feat_bank + (1 - self.momentum) * self.update_feat_bank

        #self.update_feat_bank = torch.zeros(self.num_classes, self.dim)
        nn.init.constant_(self.update_feat_bank, 0)
        #self.update_times = torch.zeros(self.num_classes)
        nn.init.constant_(self.update_times, 0)

    def return_bank_feat(self):
        return torch.FloatTensor(self.feat_bank)


    def forward(self):
        pass

    def proc(self, scores, labels, feat):

        #device = torch.get_device(self.update_feat_bank)
        labels = labels.detach()
        scores = self.softmax(scores.detach())
        pred_val, pred_pos = torch.max(scores, 1)
        #pred_val = pred_val.to(device)
        feat = feat.detach()
        #feat = feat.to(device)
        bs, dim = feat.size()

        correct_judge = (pred_pos == labels)
        error_judge = (pred_pos != labels)
        correct_ind = torch.nonzero(correct_judge).squeeze(1)
        error_ind = torch.nonzero(error_judge).squeeze(1)
        if self.choose == 'err':
            choosen_range = error_ind
        if self.choose == 'all':
            choosen_range = list(range(len(labels)))
        if self.choose == 'correct':
            choosen_range = correct_ind
        #for ind in correct_ind:
        #for ind in range(len(labels)):
        #for ind in error_ind:

        #print( '\nchoose range', choosen_range, 'update_feat_bank', self.update_feat_bank.shape, 'labels', labels, '\n')

        for ind in choosen_range:
            sub_label = labels[ind]
            tmp_clone = self.update_feat_bank[sub_label].clone()
            self.update_feat_bank[sub_label] = self.momentum * tmp_clone + (1 - self.momentum) * feat[ind]

            self.update_times[sub_label] += 1



