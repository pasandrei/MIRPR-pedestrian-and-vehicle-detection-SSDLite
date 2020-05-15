# Introduction

  - This repo provides training, validation and inference support for three detection models: SSD + ResNet [1][2], SSDLite [3] and a modified version of SSDLite used for detecting the person class only.
  - Training and validation is done on the COCO [4] dataset, pycocotools being used for evaluation.
  
# The dataset
This repo also uses a subset of COCO that contains images with only the annotations of the person class. To obtain this dataset, run `coco_subset_getter.py` from /utils. Make sure to first create an empty directory structure similar to the original, in the same directory.

# Configuration
Before training or inference, there are several settings one can configure, all of which are set from the /general_config directory:
  - The model to use: change `model_id` in `general_config.py` to one of the 3 available in `constants.py`
  - To use the full COCO dataset or use only images that contain annotations of people, change `dataset_root` path in `constants.py`. The dataset should be up one directory of the project. When using the dataset containing only person annotations, the model should be set up accordingly. This is done in `classes_config.py`, by default, only the modified SSDLite is trained only on people. Change `model_to_ids` if some other setup is desired.
  - To play around with the anchor configuration of the models, change `model_to_anchors` in `anchor_config.py`
  - The device stuff is run on is also set from `general_config.py`, can be "cpu" or "cuda:0"

# Models description
- The SSD with ResNet backbone is taken from [2], and is trained and used only for benchmarking. The trainig setup is also similar to the one described in [2]
- The SSDLite is obtained by replacing regular convolutions from the original with depth wise separable ones and replacing the ResNet backbone with MobileNetV2, as explained in [3].
- The modified SSDLite uses fewer channels than the regular one, and is placed on top of the second to last MobileNetV2 layer (with 320 filters), instead of the last one (with 1280 filters), this idea is inspired from a semantic segmentation setup described in [3]. Also, only vertical and square anchor boxes are used. These modifications were made to optimise speed and performance when detecting only people. The resulting model has only 2M parameters.
    
# Training and evaluation
  - Data augmentation: all models are trained with data augmentation, specifically the following are employed: random crops, rotations, photometric distorsions and horizontal flipping.
  - Other training details: Warm up [5] and zero weight decay on batch norm and bias layers are used [6]. Complete details and hyperparameter settings are in the params.json file of each model experiment. Directory structure: /misc/experiments/model_id/params.json.
  - Evaluation is done on the COCO validation set, with pycocotools.
**Tutorial notebook** - [`tutorial_notebook.ipynb`](https://github.com/pasandrei/MIRPR-pedestrian-and-vehicle-detection-SSDLite/blob/develop/tutorial_notebook.ipynb)

# Inference
- Inference can be done on images or .mp4 videos following the example in the [`tutorial_notebook.ipynb`](https://github.com/pasandrei/MIRPR-pedestrian-and-vehicle-detection-SSDLite/blob/develop/tutorial_notebook.ipynb)
- Speed benchmarks are also available, which can be run on cpu or gpu.

# Results
- The following two results represent the performance of ResNet SSD and SSDLite on the COCO validation set:

ResNet SSD
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.237
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.400
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.245
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.064
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.254
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.389
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.220
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.319
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.335
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.103
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.361
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.529
```
SSDLite
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.177
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.315
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.177
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.017
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.150
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.351
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.180
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.254
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.265
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.039
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.252
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.502
```

- The following three results represent the performance of the three models the COCO person validation set:
It needs to be mentioned that ResNet SSD and SSDLite were trained on the entire COCO dataset (80 classes), and these evaluation results are obtained by only considering the person class. The modified SSDLite was trained directly and only on the person class.

ResNet SSD - person mAP
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.333
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.627
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.311
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.110
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.400
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.587
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.152
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.368
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.419
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.187
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.498
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.676
 ```

SSDLite - person mAP
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.242
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.488
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.215
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.030
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.235
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.559
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.134
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.280
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.322
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.356
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.650
 ```

SSDLite Modified - person mAP
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.277
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.517
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.263
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.044
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.295
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.596
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.144
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.324
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.395
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.132
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.460
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.719
 ```
 
 The modified SSDLite does better than the standard, despite being quite smaller. Using more appropriate anchors and training only on person are the reason for this improvement. However, it still falls short of the standard SSD with ResNet. Neither of the two SSDLite version use the 38x38 grid, which is why they suffer with detection of small objects, perhaps and extremely good choice of anchors would alleviate this issue to a degree.
 
 # Speed
- The speed of the original SSDLite and the modified version is compared for the first 100 images (batch_size = 1) of the COCO validation set, benchmarking both the time taken by the model and the postprocessing time:
Times stated in seconds.

Modified SSDLite - 0.15 confidence threshold
```
Total time of model:  8.31
Mean time model:  0.08
Total time of pre nms:  0.07
Mean time pre nms:  0.00
Total time of nms:  0.60
Mean time nms:  0.01
Mean number of boxes processed by nms:  36.35

Total time taken:  8.980851173400879
Percentage of model:  92.57
Percentage of pre nms:  0.78
Percentage of nms:  6.65
 ```

SSDLite - 0.15 confidence threshold
```
Total time of model:  9.21
Mean time model:  0.09
Total time of pre nms:  0.25
Mean time pre nms:  0.00
Total time of nms:  0.46
Mean time nms:  0.00
Mean number of boxes processed by nms:  44.24

Total time taken:  9.915313005447388
Percentage of model:  92.85
Percentage of pre nms:  2.52
Percentage of nms:  4.63
 ```

Modified SSDLite - 0.015
```
Total time of model:  7.93
Mean time model:  0.08
Total time of pre nms:  0.09
Mean time pre nms:  0.00
Total time of nms:  10.04
Mean time nms:  0.10
Mean number of boxes processed by nms:  200.00

Total time taken:  18.06364417076111
Percentage of model:  43.91
Percentage of pre nms:  0.49
Percentage of nms:  55.60
 ```
 
 One issue that is worth highlighting is the time taken by NMS. When the number of boxes processed by NMS is high (capped at 200, obtained for low confidence thresholds), the time taken by NMS exceeds the processing time of the model! This is usually solved by choosing a different programming language for the implementation of NMS, such as C++, as Python is slow for this type of computation.
 
 Hardware: 
 CPU: Intel(R) Core(TM) i5-4460 CPU@ 3.20GHz
 GPU: Nvidia GeForce GTX 1660

 
References:
[1]: [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
[2]: [NVIDIA SSD model implementation](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD)
[3]: [SSDLite](https://arxiv.org/abs/1801.04381)
[4]: [COCO](https://arxiv.org/abs/1405.0312)
[5]: [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)
[6]: [Highly Scalable Deep Learning Training System with Mixed-Precision:Training ImageNet in Four Minutes](https://arxiv.org/abs/1807.11205)

