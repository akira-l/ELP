# coding=utf8
from __future__ import division
import os
import json
import csv
import random
import pandas as pd

import torch
import torch.utils.data as data
import PIL.Image as Image
from PIL import ImageStat

import pdb


def save_herb(result_gather=None, save_suffix=''):
    if result_gather is None:
        result_gather = torch.load('./result/result_gather.pt')
    result_folder = './result/'
    anno_file = open('./../FGVC_herb/herb/anno/test_labeled.txt')
    test_name = anno_file.readlines()
    img_names = [x.split(' ')[0] for x in test_name]

    #torch.save(result_gather, result_folder + 'result_save_%s.pt'%args.save_suffix)
    result_file=open(result_folder+'result_%s.csv'%save_suffix,'w+')
    csv_writer = csv.writer(result_file, lineterminator='\n')
    wdata_ = [['Id', 'Category']]
    wdata = [[x.split('/')[1], result_gather[x]] for x in img_names]
    wdata_.extend(wdata)
    csv_writer.writerows(wdata_)
    print('saved file for submitting ...')


def save_product(result_gather=None, save_suffix=''):
    result_folder = './result/'
    if result_gather is None:
        result_gather_ = torch.load('./result_gather.pt')
    sample = open('./../FGVC_product/anno/sample_submission.csv')
    sample = sample.readlines()
    sample = sample[1:]
    sample = [x.split(',')[0] for x in sample]
    result_gather = {}
    for item_keys in result_gather_:
        result_gather[item_keys.split('/')[1]] = result_gather_[item_keys]
    test_json = json.load(open('./../FGVC_product/anno/test.json'))
    result_file=open(result_folder + 'result%s.csv'%save_suffix,'w+')
    #csv_writer = csv.writer(result_file, delimiter =',',quotechar =' ',quoting=csv.QUOTE_MINIMAL)
    csv_writer = csv.writer(result_file, lineterminator='\n')
    wdata_ = [['id', 'predicted']]
    #wdata = [[x['id'], result_gather[x['id']]] for x in test_json['images']]
    wdata = [[x, result_gather[x]] for x in sample]
    wdata_.extend(wdata)
    csv_writer.writerows(wdata_)
    print('saved file for submitting ...')


def save_butterfly(result_gather=None, save_suffix=''):
    if result_gather is None:
        result_gather = torch.load('./result/focal_turned_result_gather.pt')
    result_folder = './../result/'
    anno_file = './../../butterfly/anno/fgvc_fg_testing.json'
    test_json = json.load(open(anno_file))
    result_file=open(result_folder+'result_%s.csv'%save_suffix,'w+')
    csv_writer = csv.writer(result_file, lineterminator='\n')
    wdata_ = [['id', 'predicted']]
    wdata = [[x['id'], result_gather['test_data/'+x['file_name']]] for x in test_json['images']]
    wdata_.extend(wdata)
    csv_writer.writerows(wdata_)
    print('saved file for submitting ...')


class Submit_result(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, results, save_suffix):
        if self.dataset == 'product':
            save_product(results, save_suffix)
        if self.dataset == 'herb':
            save_herb(results, save_suffix)
        if self.dataset == 'butterfly':
            save_butterfly(results, save_suffix)


def save_result(result_gather, save_suffix):
    result_folder = './result/'
    anno_file = './../butterfly/anno/fgvc_fg_testing.json'

    #torch.save(result_gather, result_folder + 'result_save_%s.pt'%args.save_suffix)
    try:
        test_json = json.load(open(anno_file))
        result_file=open(result_folder+'result_%s.csv'%save_suffix,'w+')
        csv_writer = csv.writer(result_file, lineterminator='\n')
        wdata_ = [['id', 'predicted']]
        wdata = [[x['id'], result_gather['test_data/'+x['file_name']]] for x in test_json['images']]
        wdata_.extend(wdata)
        csv_writer.writerows(wdata_)
        print('saved file for submitting ...')
    except:
        #torch.save(result_gather, './result/%s_result_gather.pt'%args.save_suffix)
        pdb.set_trace()
        test_json = json.load(open(anno_file))
        result_file=open(result_folder+'result_%s.csv'%save_suffix,'w+')
        csv_writer = csv.writer(result_file, lineterminator='\n')
        wdata_ = [['id', 'predicted']]
        wdata = [[x, result_gather[x]] for x in test_json['images']]
        wdata_.extend(wdata)
        csv_writer.writerows(wdata_)
        return False
    return True



if __name__ == '__main__':
    save_butterfly(None, 'buff_focalturned_submit')

