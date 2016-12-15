# Tensorflow Litterbox

## Overview
This repository started from the Inception-V3 codebase in Google's tensorflow models repository: https://github.com/tensorflow/models/tree/master/inception/inception. 

I created this code with the goal of learning the ins and outs of Tensorflow and developing a train/eval/prediction/export framework and models to use for other projects. All development has been done in Python 3. Python 2.7 compatibility has not been tested or kept in mind.

A recent version of Tensorflow is required. 0.12 or newer recommended. 

## Models

The codebase includes several models descried in the following sections.

### My Models (models/my_slim)
I built several image classification models from the ground up in TF-Slim style as a learning process. At the point there were built, Inception V4 and Inception-Resnet models were not yet released by Google, this has since changed and I'd recommend using theirs. Each of the models was trained to the point where convergence was verified. Only Inception-Resnet-V2 was trained to the point where it was competitive on Imagenet.
 
Included models:

 - VGG 16/19
 - ResNet (Standard Imagenet depths 18-200, pre-activation, width parameterization) 
 - Inception V4, Inception-Resnet-V1, and Inception-Resnet-V2
 
### Google Models (models/google)
 Google released a variety of image classification models using the latest iteration of TF-Slim in tf.contrib. Those models exist in the tensorflow/models repository and the tensorflow repository itself. A full copy of those models are included here as a reference and starting port for experimentation. 
 
The ResNet and Inception models have been verified to work within this train/eval/predict framework with the pretrained weights that google has released (https://github.com/tensorflow/models/tree/master/slim#pre-trained-models )
 
### Self-Driving Car Models (models/sdc)
To explore the Udacity Self-Driving Car datasets and take a crack at Challenge 2/3 I built a variety of models to perform steering angle estimation and position estimation (image based localization). 

The steering results were decent for a fully feed-forward approach. I initially started with a model inspired by Nvidia's paper but moved on to better models with a convolutional root built from Google's Inception and ResNet network code and pretrained weights.

My attempt at PoseNet inspired location regression from the same Inception/ResNet based networks did not provide satisfactory results.

## Application Examples
There are several applications included in the /litterbox folder that utilize the modules in this code base for training, evaluation, and prediction tasks. Examples of their use follows.

### Image Classification
There are three sample apps for image classification using my models or the Google based models. `imagenet_train.py` can be used to train the models from an imagenet records based dataset and similarly `imagenet_eval.py` can be used to perform validation of the models on an imagenet records dataset. `example_predict.py` can be used to run inference on any folder of jpeg images with the trained models.

#### Train
Train my variant of Inception-Resnet-V2 from scratch using 2 gpus:

`python imagenet_train.py --data_dir /data/imagenet/tensorflow/ --train_dir /train/inception-resnet-a --batch_size 32 --num_gpus 2 --my --network inception_resnet_v2 --image_size 299 --image_norm global`

Fine-tune Google's Inception-V4 network from pretrained weights on a different dataset with possibly a different number of classes from Imagenet (assuming `imagenet_data.py` was altered to match new dataset parameters):

`python imagenet_train.py --data_dir /data/records-catsdogs/ --train_dir /train/inception-catsdogs-a --pretrained_model_path /pretrained/inception_v4.ckpt --network inception_v4 --fine_tune --batch_size 64 --image_size 299 --image_norm default`

#### Evaluate
Evaluate Google's pretrained Resnet V1 101 model on imagenet validation set in TF Records format. `--label_offset 1` is required because Google's pretrained resnet models have 1000 classes while the dataset being used has 1000 + 1 background class as per the default for training Google's inception models. Additionally the ResNet and VGG pretrained weights require image normalization of 0-255 RGB with ImageNet mean subtracted so `--image_norm caffe_rgb` is required.

` python imagenet_eval.py --data_dir /data/imagenet/records/ --checkpoint_path /pretrained/resnet_v1_101.ckpt --network resnet_v1_101 --label_offset 1 --image_norm caffe_rgb --image_size 224` 

#### Predict
Run inference on a folder of png images using an Inception V4 network with Google's pretrained weights. Output offset of 1 is specified so that the background class is not factored into the inferred class index.

`python example_predict.py --data_dir /images/ --checkpoint_path /pretrained/inception_v4.ckpt --image_fmt png --image_size 299 --image_norm default --network inception_v4 --output_offset 1`

Run inference on a folder of jpeg images using a Resnet 152 network with Google's pretrained weights:

`python example_predict.py --data_dir /images/ --checkpoint_path /pretrained/resnet_v1_152.ckpt --image_size 224 --image_norm caffe_rgb --network resnet_v1_152 --num_classes 1000`

### Self-Driving Car
As with image classification, there are a number of 'sdc' applications for train, evaluation, prediction, and additionally graph export.

The training/validation data for the `sdc_train.py` and `sdc_eval.py` scripts are expected to be in TF Records format. The `bag2tf.py` script in my https://github.com/rwightman/udacity-driving-reader repository converts the Udacity bag datasets into that format. For best results on steering prediction, `bag2tf` was typically run with `-s 0.1 -c` arguments for a 10% validation split and center images only.

#### Train
Begin training steering model from Google pretrained weights for the root convnet. In this case training a model with input resolution 128*96 with nesterov momentum optimizer having a learning rate of .01. Root network is a Resnet v1 50 and the top level output is specified as the version 5 variant. 'fine_tune' specified so that loading of the output layer weights is not attempted. Per-frame standardization is used for image input normalization.

`python sdc_train.py --data_dir /data/records-elCamino/ --train_dir /train/resnet-small-1 --pretrained_model_path /pretrained/resnet_v1_50.ckpt --root_network resnet_v1_50 --top_versin 5 --fine_tune --batch_size 64 --opt momentum --lr 0.01 --image_size 128 --image_aspect 1.333 --image_norm frame`

Resume training steering model from partially trained model with same parameters as above but with a different dataset. The optimizer is changed to ADAM.

`python sdc_train.py --data_dir /data/records-hbm-2 --train_dir /train/resnet-small-1a --pretrained_model_path /train/resnet-small-1/model.ckpt-10000 --root_network resnet_v1_50 --top_version 5 --batch_size 64 --opt adam --lr 0.0001 --image_size 128 --image_aspect 1.333 --image_norm frame`

#### Evaluate
Evaluate training model on validation portion of the specified records dataset. Specifying just the path of the training output folder will cause the evaluation script to loop, continuing to evaluate the latest checkoint files created during training. A specific model checkpoint file can be specified instead and only that will be evaluated.

`python sdc_eval.py --data_dir /data/records-hbm-1/ --checkpoint_path /train/resnet-small-1a/ --root_network resnet_v1_50 --top_version 5 --image_size 128 --image_aspect 1.333 --batch_size 64 --image_norm frame`

#### Predict
Run inference on all jpg (or png if '--image_fmt png' is specified) in the data dir folder. Results are output into './output_angle.csv' file.

`python sdc_pred.py --data_dir /data/Ch2-final-test/ --checkpoint_path /train/resnet-small-1a/model.ckpt-25000 --root_network resnet_v1_50 --top_version 5 --image_size 128 --image_aspect 1.333 --batch_size 64 --image_norm frame`

#### Export
Exporting a model from a python representation of the graph with weights located in various checkpoint files into a self contained graph with weights included is a three step process that involves two scripts I've written to do some graph surgery and verification and a script that Google has provided to freeze variable weights in a graph as constants. 

##### 1. Export Graph
`python sdc_export_graph.py --image_size 128 --image_aspect 1.333 --image_norm frame --checkpoint_path /train/resnet-small-1a/model.ckpt-25000 --root_network resnet_v1_50 --top_version 5`

Instead of specifying a single model on the command line, a CSV file can be passed using the `--ensemble_path ensemble.csv` argument. An example ensemble CSV:
```
root_network,top_version,image_norm,image_size,image_aspect,checkpoint_path,weight
resnet_v1_50,5,frame,256,1.333,/train/resnet-1b/model.ckpt-12000,.75
resnet_v1_50,3,global,192,1.333,/train/resnet-2bn/model.ckpt-58000,.75
resnet_v1_50,6,frame,128,1.333,/train/resnet-small-2c/model.ckpt-15000,1.5
```

Each model in the ensemble will be instantiated in the graph in its own variable scope. The corresponding weights will be loaded with scope adjusted and the outputs of the models will be combined as a weighted mean with the weights specified in the CSV file.

##### 2. Freeze Weights
`python /tensorflow/python/tools/freeze_graph.py --input_graph ./model-graph_def.pb.txt --input_checkpoint ./model-checkpoint --output_graph resnet50-small.model --output_node_names 'output_steer'`

##### 3. Run Graph (Test)
`python sdc_run_graph.py --graph_path resnet50-small.model  --data_dir /data/Ch2-final-test/ --alpha 0.1 --target_csv /data/Ch2-final.csv`

## TODO
 - Add/update code comments and docstrings
 - Move applications into a /bin folder as proper scripts with corresponding setuptools/setup.py module install
 - Verify that multi-GPU training still works and fix if not
 - Importing framework for pulling in pretrained weights from Torch/Caffe and other TF models
 - Add support for smaller image datasets (ie Cifar/Tiny Imagenet)
 - Add bounding-box / segmentation mask support with related models
 - Add MS Coco dataset support for above
 
