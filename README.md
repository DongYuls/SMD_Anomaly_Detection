# Residual Error based Anomaly Detection Using Auto-Encoder in SMD Machine Sound

![](__pycache__/SMD.PNG?raw=true)

## About

This is the implementation of the paper "Residual Error based Anomaly Detection Using Auto-Encoder in SMD Machine Sound" by Dong Yul Oh and Il Dong Yun. 

For more information check out the paper [[PDF](http://www.mdpi.com/1424-8220/18/5/1308/pdf)] on MDPI Sensors [[website](http://www.mdpi.com/1424-8220/18/5/1308)].

All of these scripts are based on the [[sample](http://www.mdpi.com/1424-8220/18/5/1308/s1)] dataset provided in our paper. Note that all rights reserved to [[Crevis Co., Ltd](http://www.crevis.co.kr/eng/main/main.php)], and only a few samples for the SMD machine sound can be released. The experimental results of this demo may differ (have poorer performance) from those in the paper because of the lack of training data.

## Getting started

### Dependencies

Python 3.5.0\
Tensorflow 1.4.0\
Psutil 5.4.1\
Numpy 1.13.3\
Matplotlib 1.2.0\
Librosa 0.5.1

### Training & Validation

**The code includes scripts for ...**\
`setup_dataset.py`: data pre-processing and building tfrecords.\
`model.py`: the network architecture as proposed in our paper.\
`main.py`: training script for this model.

**Usage:**\
`python setup_dataset.py` transforms the audio files (with STFT) in each class into tfrecords before training the model.\
`python setup_dataset.py test` checks if tfrecords has been created properly.\
`python main.py --train=true` trains the model only with normal data and learning their manifold.\
`python main.py` measures AUC the residual error for each abnormal and normal test case.


## BibTeX 

If you use this code in your project, please cite our paper:

```
@Article{s18051308,
    AUTHOR         = {Oh, Dong  Yul and Yun, Il  Dong},
    TITLE          = {Residual Error Based Anomaly Detection Using Auto-Encoder in SMD Machine Sound},
    JOURNAL        = {Sensors},
    VOLUME         = {18},
    YEAR           = {2018},
    NUMBER         = {5},
    ARTICLE NUMBER = {1308},
    URL            = {http://www.mdpi.com/1424-8220/18/5/1308},
    ISSN           = {1424-8220},
    DOI            = {10.3390/s18051308}
}
```





