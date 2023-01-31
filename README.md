# ELP 

## Introduction

This project is an implementation of [*A Simple Episodic Linear Probe Improves Visual Recognition in the Wild*]

## Insight
* A simple online linear probe can boost recognition performances. The simple regularization leads to better performances without complex network designs or additional data.

## Requirements

Python 3 & Pytorch >= 0.4.0 

## Datasets Orgnization 

Similar to [DCL](https://github.com/JDAI-CV/DCL). 

## Training

Run `train.py` to train ELP.

For CUB / STCAR / AIR 

```shell
python train.py --data $DATASET --epoch 360 --backbone resnet50 \
                    --tb 16 --tnw 16 --vb 512 --vnw 16 \
                    --lr 0.0008 --lr_step 60 \
                    --cls_lr_ratio 10 --start_epoch 0 \
                    --detail training_descibe --size 512 \
                    --crop 448 
```

For ImageNet

```shell
python train.py --data CUB --epoch 100 --backbone resnet50 \
                    --tb 1024 --tnw 16 --vb 2048 --vnw 16 \
                    --lr 0.01 --lr_step 10 \
                    --cls_lr_ratio 10 --start_epoch $LAST_EPOCH \
                    --detail training_descibe4checkpoint --size 256 \
                    --crop 224 
```

You can rewrite line 98-125 in utils/train_model.py for your own codebase. 

## Citation
Please cite ELP paper if you find ELP is helpful in your work:
```
@InProceedings{liang2022elp,
title={A Simple Episodic Linear Probe Improves Visual Recognition in the Wild},
author={Liang, Yuanzhi and Zhu, Linchao and Wang, Xiaohan and Yang, Yi},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2022}
}
```
