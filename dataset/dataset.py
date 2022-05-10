# coding=utf8
from __future__ import division
import os
import torch
import torch.utils.data as data
import pandas
import random
import PIL.Image as Image
from PIL import ImageStat

import pdb

def random_sample(img_names, labels):
    anno_dict = {}
    img_list = []
    anno_list = []
    for img, anno in zip(img_names, labels):
        if not anno in anno_dict:
            anno_dict[anno] = [img]
        else:
            anno_dict[anno].append(img)

    for anno in anno_dict.keys():
        anno_len = len(anno_dict[anno])
        fetch_keys = random.sample(list(range(anno_len)), anno_len//10)
        img_list.extend([anno_dict[anno][x] for x in fetch_keys])
        anno_list.extend([anno for x in fetch_keys])
    return img_list, anno_list


class dataset(data.Dataset):
    def __init__(self, Config, anno, swap_size=[7,7], unswap=None, swap=None, totensor=None, train=False, train_val=False, test=False):
        self.root_path = Config.rawdata_root
        self.numcls = Config.numcls
        self.dataset = Config.dataset
        if isinstance(anno, pandas.core.frame.DataFrame):
            self.paths = anno['ImageName'].tolist()
            self.labels = anno['label'].tolist()
        elif isinstance(anno, dict):
            self.paths = anno['img_name']
            self.labels = anno['label']

        if train_val:
            self.paths, self.labels = random_sample(self.paths, self.labels)
        self.unswap = unswap
        self.swap = swap
        self.totensor = totensor
        self.cfg = Config
        self.train = train
        self.swap_size = swap_size
        self.test = test

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_path, self.paths[item])
        img = self.pil_loader(img_path)
        if self.test:
            img = self.totensor(img)
            if self.dataset == 'CUB': 
                label = self.labels[item] 
            else: 
                label = self.labels[item] 
            return img, label, self.paths[item]
        img_unswap = self.unswap(img) if not self.unswap is None else img

        img_unswap = self.totensor(img_unswap)
        if self.dataset == 'CUB': 
            label = self.labels[item] 
        else: 
            label = self.labels[item] 
        return img_unswap, label, self.paths[item]


    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def crop_image(self, image, cropnum):
        width, high = image.size
        crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        im_list = []
        for j in range(len(crop_y) - 1):
            for i in range(len(crop_x) - 1):
                im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
        return im_list


    def get_weighted_sampler(self):
        img_nums = len(self.labels)
        weights = [self.labels.count(x) for x in range(self.numcls)]
        return torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=img_nums)


def collate_fn4train(batch):
    imgs = []
    label = []
    label_swap = []
    law_swap = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        imgs.append(sample[1])
        label.append(sample[2])
        label.append(sample[2])
        label_swap.append(0)
        label_swap.append(1)
        law_swap.append(sample[3])
        law_swap.append(sample[4])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, label_swap, law_swap, img_name

def collate_fn4val(batch):
    imgs = []
    label = []
    label_swap = []
    law_swap = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
        label_swap.append(0)
        law_swap.append(sample[2])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, label_swap, law_swap, img_name

def collate_fn4backbone(batch):
    imgs = []
    label = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, img_name


def collate_fn4test(batch):
    imgs = []
    label = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, img_name
