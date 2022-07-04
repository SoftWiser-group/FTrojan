# Backdoor Attack in Frequency Domain

## DEPENDENCIES
python==3.8.3  
numpy==1.19.4  
tensorflow==2.4.0  
opencv==4.5.1  
idx2numpy==1.2.3  
pytorch==1.7.0  

## Dataset Preparation
For GTSRB, PubFig(60 labels) and ImageNet(16 labels), you can download from *data/download.txt* and uncompress them in *data* directory. 
For cifar10 dataset, you can obtain from *tensorflow.keras.datasets*. 

## Change Config
You can modify the *param dict* in the train.py to train your own backdoored model. 
There are 6 parameters as follows:   
* dataset: CIFAR10, GTSRB, MNIST, PubFig, ImageNet16. Default: CIFAR10  
* target_label: The target label to backdoor. Default: 8  
* poisoning_rate: The ratio of poisoning sample. A float number range (0,1)  
* channel_list: Which channels to implant backdoor, [1,2] means UV, [0,1,2] means YUV.
* YUV: True, YUV; False, RGB  
* magnitude: The magnitude of trigger frequency
* clean_label: clean_label attack or changed label attack. For clean label attack, it is required to train a clean model saved to *model* directory.
* pos_list: trigger frequency position, for (32, 32) belongs to high-frequency, (15, 15) belongs to mid-frequency
## Run Backdoor Attack Code
For Tensorflow2.0, you can directly run(default cifar10 dataset):
```shell
python train.py
```

For Pytorch, you can run:
```shell
python th_train.py
```


