import os
import pandas as pd
import torch

from transforms import transforms
from utils.autoaugment import ImageNetPolicy

if os.path.exists('../pretrained'):
    pretrained_model = {'resnet50' : './../pretrained/resnet50-19c8e357.pth',
                        'resnet101': './models/pretrained/se_resnet101-7e38fcc6.pth',
                        'senet154':'./models/pretrained/checkpoint_epoch_017_prec3_93.918_pth.tar'}
else:
    pretrained_model = {'resnet50' : './pretrained/resnet50-19c8e357.pth',
                        'resnet101': './models/pretrained/se_resnet101-7e38fcc6.pth',
                        'senet154':'./models/pretrained/checkpoint_epoch_017_prec3_93.918_pth.tar'}


def load_data_transformers(resize_reso=512, crop_reso=448, swap_num=[7, 7]):
    center_resize = 600
    Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data_transforms = {
       	'swap': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((crop_reso, crop_reso)),
            transforms.RandomHorizontalFlip(),
            transforms.Randomswap((swap_num[0], swap_num[1])),
        ]),
       	'food_swap': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(degrees=90),
            #transforms.RandomCrop((crop_reso, crop_reso)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=crop_reso, scale=(0.75, 1)),
            transforms.Randomswap((swap_num[0], swap_num[1])),
        ]),
       	'food_unswap': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(degrees=90),
            #transforms.RandomCrop((crop_reso, crop_reso)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(size=crop_reso, scale=(0.75, 1)),
        ]),

        'unswap': transforms.Compose([
            #transforms.RandomResizedCrop(size=crop_reso,),
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(size=crop_reso, scale=(0.4, 1)),
            transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            #transforms.RandomCrop((crop_reso,crop_reso), padding=8),
            #transforms.RandomCrop((crop_reso,crop_reso)),
        ]),

        'imagenet_unswap': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((crop_reso,crop_reso)),
            transforms.RandomHorizontalFlip(),
        ]),

        'train_totensor': transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            #ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),

        'imagenet_train_totensor': transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),

        'val_totensor': transforms.Compose([
            #transforms.Resize((crop_reso, crop_reso)),
            transforms.Resize((resize_reso, resize_reso)),
            transforms.CenterCrop((crop_reso, crop_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'test_totensor': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.CenterCrop((crop_reso, crop_reso)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'None': None,
        'Centered_swap': transforms.Compose([
            transforms.CenterCrop((center_resize, center_resize)),
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((crop_reso, crop_reso)),
            transforms.RandomHorizontalFlip(),
            transforms.Randomswap((swap_num[0], swap_num[1])),
       ]),
        'Centered_unswap': transforms.Compose([
            transforms.CenterCrop((center_resize, center_resize)),
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((crop_reso, crop_reso)),
            transforms.RandomHorizontalFlip(),
       ]),
        'Tencrop': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.TenCrop((crop_reso, crop_reso)),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
      ])
    }

    return data_transforms




class LoadConfig(object):
    def __init__(self, args, version):
        if version == 'train':
            get_list = ['train', 'val']
        elif version == 'val':
            get_list = ['val']
        elif version == 'test':
            get_list = ['test']
        elif version == 'ensamble':
            get_list = []
        else:
            raise Exception("train/val/test ???\n")

        ###############################
        #### add dataset info here ####
        ###############################

        if args.dataset == 'product':
            self.dataset = args.dataset
            self.rawdata_root = './../FGVC_product/data'
            self.anno_root = './../FGVC_product/anno'
            self.numcls = 2019

        if args.dataset == 'CUB':
            self.dataset = args.dataset
            self.rawdata_root = '../../data/CUB_200_2011/all'
            self.anno_root = '../../data/CUB_200_2011'
            self.numcls = 200

        if args.dataset == 'imagenet':
            self.dataset = args.dataset
            self.rawdata_root = './'
            self.anno_root = './imagenet_anno'
            self.numcls = 1000


        if args.dataset == 'inat18':
            self.dataset = args.dataset
            self.rawdata_root = '/data03/liangyzh/iNaturelist18'
            if not os.path.exists(self.rawdata_root):
                self.rawdata_root = './'
            self.anno_root = '/data03/liangyzh/iNaturelist18/anno'
            if not os.path.exists(self.anno_root):
                self.anno_root = './anno'
            self.numcls = 8142


        if args.dataset == 'stcar':
            self.dataset = args.dataset
            self.rawdata_root = '../../data'
            self.anno_root = '../../data/fgvc_anno/car_anno'
            self.numcls = 196


        if args.dataset == 'aircraft':
            self.dataset = args.dataset
            self.rawdata_root = '../../data'
            self.anno_root = '../../data/fgvc_anno/air_anno'
            self.numcls = 100


        if args.dataset == 'herb':
            self.dataset = args.dataset
            self.rawdata_root = './../FGVC/herb/data'
            self.anno_root = './../FGVC/herb/anno'
            self.numcls = 683

        if args.dataset == 'food':
            self.dataset = args.dataset
            self.rawdata_root = './../FGVC_food/data'
            self.anno_root = './../FGVC_food/anno'
            self.numcls = 251

        if 'train' in get_list:
            if args.dataset == 'inat18':
                self.train_anno = pd.read_csv(os.path.join(self.anno_root, 'train2018.txt'),\
                                           sep=" ",\
                                           header=None,\
                                           names=['ImageName', 'label'])
            elif args.dataset == 'CUB':
                self.train_anno = pd.read_csv(os.path.join(self.anno_root, 'train.txt'),\
                                           sep=" ",\
                                           header=None,\
                                           names=['ImageName', 'label'])
            else:
                self.train_anno = pd.read_csv(os.path.join(self.anno_root, 're_train.txt'),\
                                           sep=" ",\
                                           header=None,\
                                           names=['ImageName', 'label'])

        if args.dataset == 'inat18':
            self.val_anno = pd.read_csv(os.path.join(self.anno_root, 'val2018.txt'),\
                                           sep=" ",\
                                           header=None,\
                                           names=['ImageName', 'label'])
        elif 'val' in get_list:
            if args.dataset == 'CUB':
                self.val_anno = pd.read_csv(os.path.join(self.anno_root, 'val.txt'),\
                                           sep=" ",\
                                           header=None,\
                                           names=['ImageName', 'label'])
            else: 
                self.val_anno = pd.read_csv(os.path.join(self.anno_root, 're_val.txt'),\
                                           sep=" ",\
                                           header=None,\
                                           names=['ImageName', 'label'])

        if 'test' in get_list:
            self.test_anno = pd.read_csv(os.path.join(self.anno_root, 'test_labeled.txt'),\
                                           sep=" ",\
                                           header=None,\
                                           names=['ImageName', 'label'])

        self.swap_num = args.swap_num

        #self.save_dir = './net_model'
        #self.save_dir = 'afs/output/'
        self.save_dir = './net_model'
        self.backbone = args.backbone
        self.train_bs = args.train_batch

        self.use_dcl = False
        self.use_backbone = False if self.use_dcl else True
        self.use_Asoftmax = False
        self.use_focal_loss = False
        self.use_fpn = True

        self.weighted_sample = False

        self.log_folder = './logs'
        if not os.path.exists(self.log_folder):
            os.mkdir(self.log_folder)

        self.bank_pick_num = 5
        self.otmap_thresh = 0#0.3
        self.otmap_struct_max = 50

        self.bank_dim = 2048

        self.st_downsample = 64
        self.st_down_size = self.bank_dim // 64#self.bank_dim // 64
        self.st_bank_dim = self.bank_dim #self.bank_dim // 64

        self.st_map_size = 14
        self.train_ver = args.train_ver 

        self.mem_m = 0.5

        self.alpha = args.alpha 
        self.gamma = args.gamma 
        self.init = args.init 

