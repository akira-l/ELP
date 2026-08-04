

# ELP 

## Introducción

Este proyecto es una implementación de [*A Simple Episodic Linear Probe Improves Visual Recognition in the Wild*]

## Idea principal
* Una sonda lineal en línea simple puede mejorar el rendimiento del reconocimiento. La simple regularización conduce a mejores resultados sin diseños de red complejos o datos adicionales.

## Requisitos

Python 3 y Pytorch >= 0.4.0 

## Organización de conjuntos de datos 

Similar a [DCL](https://github.com/JDAI-CV/DCL). 

## Entrenamiento

Ejecute `train.py` para entrenar ELP.

Para CUB / STCAR / AIR 

```shell
python train.py --data $DATASET --epoch 360 --backbone resnet50 \
                    --tb 16 --tnw 16 --vb 512 --vnw 16 \
                    --lr 0.0008 --lr_step 60 \
                    --cls_lr_ratio 10 --start_epoch 0 \
                    --detail training_descibe --size 512 \
                    --crop 448 
```

Para ImageNet

```shell
python train.py --data CUB --epoch 100 --backbone resnet50 \
                    --tb 1024 --tnw 16 --vb 2048 --vnw 16 \
                    --lr 0.01 --lr_step 10 \
                    --cls_lr_ratio 10 --start_epoch $LAST_EPOCH \
                    --detail training_descibe4checkpoint --size 256 \
                    --crop 224 
```

Puede reescribir las líneas 98-125 en utils/train_model.py para su propia base de código. 

## Citación
Por favor, cite el artículo de ELP si encuentra que ELP es útil en su trabajo:
```
@InProceedings{liang2022elp,
title={A Simple Episodic Linear Probe Improves Visual Recognition in the Wild},
author={Liang, Yuanzhi and Zhu, Linchao and Wang, Xiaohan and Yang, Yi},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2022}
}
```
