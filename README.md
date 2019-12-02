# EAST: An Efficient and Accurate Scene Text Detector

### Introduction
This is a tensorflow re-implementation of [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155v2) based off https://github.com/argman/EAST.
The features are summarized blow:
+ Only **RBOX** part is implemented.
+ A fast Locality-Aware NMS in C++ provided by the original author.

### Contents
1. [Installation](#installation)
2. [Download](#download)
3. [Main Structure of Repo]
4. [Train](#train)
5. [Test](#test)

### Installation
1. tensorflow==1.12.0
2. numpy==1.14.5
3. Shapely==1.6.4
4. scipy==1.1.0
5. opencv-python==3.4.3.18
Above are modules that need to be installed and example of versions that work. 

### Downloaded models
1. Models trained on ICDAR 2013 (training set) + ICDAR 2015 (training set): [BaiduYun link](http://pan.baidu.com/s/1jHWDrYQ) [GoogleDrive](https://drive.google.com/open?id=0B3APw5BZJ67ETHNPaU9xUkVoV0U)
2. Resnet V1 50 provided by tensorflow slim: [slim resnet v1 50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)

### Main Structure of Repository
- Data/
     Folder contains:
	* All the cropped_annotation files
	* All of the cropped image pieces in cropped_img/
        * All of the original map images and their annotations in maps/
 
- evalEAST.py
     Main code for evaluating EAST model text detection performances.
- icdar.py
     Contains code that help transform input data into a form that can be feed into the EAST model and transform output from the EAST model back into rectangle detections.
- lamns
     Folder contains code for performing locality-aware mns merging. Did not modify.
- model.py
     Contains code for how the RESNET feature extractor ties in with the rest of the neural network architecture in EAST.
- multigpu_train.py
     Main code for the training iterations in EAST.
- nets
     Folder contains code for the RESNET feature extractor in EAST.
- tmp/
     Folder contains the models and checkpoints for Tensorflow training

- genCharData.py
     Contains extension code for drawing annotated boxes on one single channel of the cropped images.

### Train
If you want to train the model, you should provide info such as dataset path, model path, batch size per gpu, etc. 

```
python multigpu_train.py --gpu_list=0 --input_size=512 --batch_size_per_gpu=14 --checkpoint_path=/tmp/east_icdar2015_resnet_v1_50_rbox/ \
--text_scale=512 --training_data_path=/data/ocr/icdar2015/ --geometry=RBOX --learning_rate=0.0001 --num_readers=24 \
--pretrained_model_path=/tmp/resnet_v1_50.ckpt
```

If you have more than one gpu, you can pass gpu ids to gpu_list(like --gpu_list=0,1,2,3)

>> run above with multigpu_train_39channels.py to train EAST with 39 channel inputs, containing alphanumeric info in addition to rgb info of the data

~~~~~ The above commands are encapsulated into the run.sh file for executing in Gyspum ~~~~~

### Evaluate
run
```
python evalEAST.py
```

a text file will be then written to the output path.

~~~~~ The above commands are encapsulated into the run2.sh file for execution in Gypsum ~~~~~
