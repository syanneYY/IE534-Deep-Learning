# Human Action Recognition

This assignment will work with the UCF-101 human action recognition dataset. The original technical report published in 2012 discussing the dataset can be found here. The dataset consists of 13,320 videos between ~2-10 seconds long of humans performing one of 101 possible actions. The dimensions of each frame are 320 by 240.


This homework will compare a single frame model (spatial information only) with a 3D convolution based model (2 spatial dimensions + 1 temporal dimension).

- Part one: Fine-tune a 50-layer ResNet model (pretrained on ImageNet) on single UCF-101 video frames
- Part two: Fine-tune a 50-layer 3D ResNet model (pretrained on Kinetics) on UCF-101 video sequences


## Introduction

CNNs have been shown to work well in nearly all tasks related to images. A video can be thought of simply as a sequence of images. That is, a network designed for learning from videos must be able to handle the spatial information as well as the temporal information (typically referred to as learning spatiotemporal features). There are many modeling choices to deal with this. Below are a list of papers mentioned in the action recognition presentation given in class. This list is in no way comprehensive but definitely gives a good idea of the general progress.

Large-scale Video Classification with Convolutional Neural Networks (2014)

Two-Stream Convolutional Networks for Action Recognition in Videos (2014)

Beyond Short Snippets: Deep Networks for Video Classification (2015)

Learning Spatiotemporal Features with 3D Convolutional Networks (2015)

The Kinetics Human Action Video Dataset (2017)

Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset (2017)

Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet? (2017)
