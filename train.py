#coding=utf-8
import os
import datetime
import argparse
import logging
import pandas as pd

import torch
import torch.nn as nn
from  torch.nn import CrossEntropyLoss
import torch.utils.data as torchdata
from torchvision import datasets, models
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from transforms import transforms
from utils.train_model import train
from utils.utils import setup_seed
from utils.warmup_scheduler import GradualWarmupScheduler

from models.LoadModel import MainModel
from models.elp_model import ELPModule

from config import LoadConfig, load_data_transformers
from dataset.dataset import collate_fn4train, collate_fn4val, collate_fn4test, collate_fn4backbone, dataset

print(torch.__version__)

import pdb


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1234'
    #dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--data', dest='dataset',
                        default='imagenet', type=str, help='imagenet | cub | stcar | aircraft ')
    parser.add_argument('--save', dest='resume',
                        default=None, #'../pretrained/resnet50-19c8e357.pth', 
                        type=str)
    parser.add_argument('--backbone', dest='backbone',
                        default='resnet50', type=str)
    parser.add_argument('--train_ver', dest='train_ver',
                        default='sum', type=str)
    parser.add_argument('--auto_resume', dest='auto_resume',
                        action='store_true')
    parser.add_argument('--gpu', dest='gpu',
                        default=0, type=int)

    parser.add_argument('--epoch', dest='epoch',
                        default=100, type=int)
    parser.add_argument('--tb', dest='train_batch',
                        default=1024, type=int)
    parser.add_argument('--vb', dest='val_batch',
                        default=2048, type=int)

    parser.add_argument('--sp', dest='save_point',
                        default=5000, type=int)
    parser.add_argument('--cp', dest='check_point',
                        default=5000, type=int)
    parser.add_argument('--lr', dest='base_lr',
                        default=0.2, type=float)
    parser.add_argument('--lr_step', dest='decay_step',
                        default=60, type=int)
    parser.add_argument('--cls_lr_ratio', dest='cls_lr_ratio',
                        default=10.0, type=float)

    parser.add_argument('--alpha', dest='alpha',
                        default=0.5, type=float)
    parser.add_argument('--gamma', dest='gamma',
                        default=2, type=float)
    parser.add_argument('--init', dest='init',
                        default=2, type=float)

    parser.add_argument('--start_epoch', dest='start_epoch',
                        default=0,  type=int)
    parser.add_argument('--tnw', dest='train_num_workers',
                        default=8, type=int)
    parser.add_argument('--vnw', dest='val_num_workers',
                        default=8, type=int)
    parser.add_argument('--detail', dest='discribe',
                        default='', type=str)
    parser.add_argument('--size', dest='resize_resolution',
                        default=256, type=int)
    parser.add_argument('--crop', dest='crop_resolution',
                        default=224, type=int)

    parser.add_argument('--weighted', dest='weighted_sample',
                        action='store_true')
    parser.add_argument('--local_rank', default=-1, type=int)
    args = parser.parse_args()
    return args


def auto_load_resume(load_dir):
    folders = os.listdir(load_dir)
    date_list = [int(x.split('_')[1].replace(' ',0)) for x in folders]
    choosed = folders[date_list.index(max(date_list))]
    weight_list = os.listdir(os.path.join(load_dir, choosed))
    acc_list = [x[:-4].split('_')[-1] if x[:7]=='weights' else 0 for x in weight_list]
    acc_list = [float(x) for x in acc_list]
    choosed_w = weight_list[acc_list.index(max(acc_list))]
    return os.path.join(load_dir, choosed, choosed_w)


if __name__ == '__main__':
    args = parse_args()
    setup_seed(42)
    print('args: \n', args, '\n\n', '========'*6)
    Config = LoadConfig(args, 'train')
    print('Config: \n', vars(Config), '\n\n', '========'*6)
    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution)

    os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
    if args.dataset == 'imagenet': 
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    else: 
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu) 

    #dist.init_process_group(backend="nccl")
    #local_rank = torch.distributed.get_rank()
    #torch.cuda.set_device(local_rank)
    #device = torch.device("cuda", local_rank)

    train_set = dataset(Config = Config,\
                        anno = Config.train_anno,\
                        unswap = transformers["unswap"],\
                        swap = transformers["swap"],\
                        totensor = transformers["train_totensor"],\
                        train = True)

    trainval_set = dataset(Config = Config,\
                        anno = Config.train_anno,\
                        unswap = transformers["None"],\
                        swap = transformers["None"],\
                        totensor = transformers["val_totensor"],\
                        train = False,
                        train_val = True)

    val_set = dataset(Config = Config,\
                      anno = Config.val_anno,\
                      unswap = transformers["None"],\
                      swap = transformers["None"],\
                      totensor = transformers["val_totensor"],\
                      train=False)

    dataloader = {}
    dataloader['train'] = torch.utils.data.DataLoader(train_set,\
                                                batch_size=args.train_batch,\
                                                shuffle=True,\
                                                #False if args.weighted_sample else True,\
                                                #sampler=DistributedSampler(train_set), #train_set.get_weighted_sampler() if args.weighted_sample else None,\
                                                num_workers=args.train_num_workers,\
                                                collate_fn=collate_fn4backbone,
                                                drop_last=False,
                                                pin_memory=True)

    setattr(dataloader['train'], 'total_item_len', len(train_set))

    dataloader['trainval'] = torch.utils.data.DataLoader(trainval_set,\
                                                batch_size=args.val_batch,\
                                                shuffle=False,\
                                                num_workers=args.val_num_workers,\
                                                collate_fn=collate_fn4backbone,
                                                drop_last=False,
                                                pin_memory=True)

    setattr(dataloader['trainval'], 'total_item_len', len(trainval_set))

    dataloader['val'] = torch.utils.data.DataLoader(val_set,\
                                                batch_size=args.val_batch,\
                                                shuffle=False,\
                                                num_workers=args.val_num_workers,\
                                                collate_fn=collate_fn4val if not Config.use_backbone else collate_fn4backbone,
                                                drop_last=False,
                                                pin_memory=True)

    setattr(dataloader['val'], 'total_item_len', len(val_set))
    cudnn.benchmark = True

    print('Choose model and train set')
    #fpn not implemented
    model = MainModel(Config)
    print('-'*10, 'main model', '-'*10)
    print(model)
    elp_model = ELPModule(Config)
    print('-'*10, 'as model', '-'*10)
    print(elp_model)
    print('-'*25)

    if (args.resume is None) and (not args.auto_resume):
        print('train from imagenet pretrained models ...')
    else:
        if not args.resume is None:
            resume = args.resume
            #print('load from pretrained checkpoint %s ...'% resume)
        elif args.auto_resume:
            resume = auto_load_resume(Config.save_dir)
            print('load from %s ...'%resume)
        else:
            raise Exception("no checkpoints to load")

        model_dict = model.state_dict()
        pretrained_dict = torch.load(resume)
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    print('Set cache dir')
    time = datetime.datetime.now()
    filename = '%s_%d%d%d_%s'%(args.discribe, time.month, time.day, time.hour, Config.dataset)
    save_dir = os.path.join(Config.save_dir, filename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    world_size = 1
    rank = 0
    n = torch.cuda.device_count() // world_size
    device_ids = list(range(rank * n, (rank + 1) * n))
    model.cuda()
    ddp_model = nn.DataParallel(model)
    #ddp_model = DDP(model, device_ids=device_ids)
    elp_model.cuda()
    ddp_elp_model = nn.DataParallel(elp_model)
    #ddp_elp_model = DDP(elp_model, device_ids=device_ids)

    # set new layer's lr
    ignored_cls = list(map(id, ddp_model.module.classifier.parameters()))
    #ignored_cls = list(map(id, model.classifier.parameters()))
    ignored_params = ignored_cls

    print('the num of new layers:', len(ignored_params))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    lr_ratio = args.cls_lr_ratio
    base_lr = args.base_lr
    simple_cls_ratio = 10
    optimizer = optim.SGD([{'params': base_params},
                               {'params': ddp_model.module.classifier.parameters(), 'lr': lr_ratio*base_lr},
                               #{'params': ddp_model.classifier.parameters(), 'lr': lr_ratio*base_lr},

                              ], lr = base_lr, momentum=0.9)
    filter_optim = optim.SGD([
                               {'params': ddp_elp_model.module.parameters()},
                               #{'params': ddp_elp_model.parameters(), 'lr': 0.01},

                              ], lr = 0.01, momentum=0.9)


    if args.dataset == 'imagenet': 
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else: 
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[120, 180, 240, 300], gamma=0.1)
    #filter_exp_lr_scheduler = lr_scheduler.MultiStepLR(filter_optim, milestones=[100], gamma=0.1)

    optim_dict = {}
    optim_dict['common'] = optimizer
    optim_dict['filter'] = filter_optim

    sch_dict = {}
    sch_dict['common'] = exp_lr_scheduler
    #sch_dict['filter'] = filter_exp_lr_scheduler

    models_transfer_dict = {}
    models_transfer_dict['base'] = ddp_model
    models_transfer_dict['elp'] = ddp_elp_model

    train(Config,
          models_transfer_dict,
          epoch_num=args.epoch,
          start_epoch=args.start_epoch,
          optimizer_recieve=optim_dict,
          scheduler_recieve=sch_dict,#warmup_scheduler,
          data_loader=dataloader,
          save_dir=save_dir,
          data_ver='allw',
          data_size=args.crop_resolution,
          savepoint=args.save_point,
          checkpoint=args.check_point)


