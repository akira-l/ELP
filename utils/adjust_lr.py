import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import pdb

def adjust_learning_rate(optimizer, epoch):
    
    lr = 8e-6
    if epoch < 10:
        lr = 1e-4
    elif epoch >= 10 and epoch <= 60:
        lr = 8e-4 
    elif epoch > 60 and epoch <=120:
        lr = 8e-5
    elif epoch > 120 and epoch <=180:
        lr = 8e-6
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
