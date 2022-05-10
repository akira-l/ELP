import torch
from torch import nn
from torchvision import models, transforms, datasets
import torch.nn.functional as F

import pdb

class ELPModule(nn.Module):
    def __init__(self, config):
        super(ELPModule, self).__init__()
        self.num_classes = config.numcls
        self.sp_Acls = nn.Sequential(
                                 nn.Linear(2048, 512), 
                                 #nn.ReLU(), 
                                 nn.Linear(512, self.num_classes)
                                 ) 


    def forward(self, mem_feat):

        mem_out = self.sp_Acls(mem_feat)

        return mem_out





