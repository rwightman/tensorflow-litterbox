# Tensorflow Litterbox

## Overview
This repository started from the Inception-V3 codebase in Google's tensorflow models repository: https://github.com/tensorflow/models/tree/master/inception/inception. 

My initial motivation for creating this codebase was to compete in the Kaggle State Farm Distracted Driver competition. I ended up switching to Torch and Facebook's fb.resnet.torch implementation for the competition. However, since then I've been slowly working on this code base with the goal of learning the ins and outs of Tensorflow and developing a train/eval/prediction framework and models to use for another project.

## Models
The codebase includes several models:
 - VGG 16/19
 - ResNet (Standard Imagenet depths 18-200, width parameterization) 
 - Inception V3 (based on Google's original Inception V3 in the models repo)
 - Inception V4 + Inception-Resnet-V1 and Inception-Resnet-V2
 
I have used the tf.contrib layers that the latest TF-Slim is based on. As a result, an up to date version of Tensorflow is required (0.10 or newer, 0.9 may work).

I've managed to get all of the models training to the point where they're converging but have yet to run any long enough on an Imagenet scale dataset to the point where results are competitive.

If you are looking for the latest models in Tensorflow with competitive pre-trained weights, Google released some last week. I would recommending checking them out here: https://research.googleblog.com/2016/08/improving-inception-and-image.html

## TODO
 - Add to this document
 - Show usage examples for various components
 - Add/update code comments and docstrings
 - Verify that multi-tower training still works and fix if not
 - Add support for smaller image datasets (ie Cifar/Tiny Imagenet)
 - Add bounding-box / segmentation mask support with related models
 - Add MS Coco dataset support for above
 