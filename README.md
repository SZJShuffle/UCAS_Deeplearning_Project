# Introduction
This project is a big assignment on the UCAS Deep Learning course.We focus on the Identification of RBP(RNA-binding protein) binding sites based on convolutional neural networks.

RNA-binding proteins (RBPs) account for 5~10\% of the eukaryotic proteome, which plays an important role in biological processes such as gene regulation. However, experimental prediction of RBP binding sites remains a time-consuming, costly task. Conversely, learning from existing annotation knowledge and predicting RBP binding sites is a quick method. In this project,we use a deep learning based model,which ensembles 2 convolutional neural networks (CNNs) to identify RBP binding sites. After training the two networks separately, 
the outputs of the two networks are combined to increase the accuracy of predicting RBP binding sites on the RNA sequence.The results show that the network we trained successfully captured experimentally validated binding motifs.
# Workflow
 ![image](https://github.com/JShuffle/UCAS-Deeplearning-Project/blob/master/workflow.png)
# Setup
## Pre-requisites
- python 3.6
- pytorch 1.1
- sklearn

The programs support GPUs and CPUs and are able to automatically detect the operating environment. If GPUs are present, it will run with the GPU first.

To ensure that the entire process works properly,verify that the directory structure all files are the same as follow:
The directory structure of the training code and sample data is as follows:
```
│—— encoding.py
│—— motif_detection.py
│—— pre_processing.py
│—— train.py
│—— README.md
|—— workflow.png
|——	data
  |—— ALKBH5_Baltz2012.ls.negatives.fa
  |—— ALKBH5_Baltz2012.ls.positives.fa
  |—— ALKBH5_Baltz2012.train.negatives.fa
  |—— ALKBH5_Baltz2012.train.positives.fa
  |——               ...
```
If you need to train other protein data, you need to put the training data in the data directory.

Other protein data links:
https://pan.baidu.com/s/1DoNyS4hSPe5RtSpQTDTp9g 

password：ughc
## Usage
`train.py`
```
usage: train.py [-h] [-model_type MODEL_TYPE] [-protein PROTEIN]
                [-epoch EPOCH]

optional arguments:
  -h, --help            show this help message and exit
  -model_type MODEL_TYPE
                        it supports the following deep network
                        models:globalCNN,localCNN and ensemble, default=ensemble
  -protein PROTEIN      input the protein you want to train ,default=ALKBH5
  -epoch EPOCH          input the epoch you want to train ,default=1
```
`motif_detection.py`
```
usage: motif_detection.py [-h] [-protein PROTEIN]

optional arguments:
  -h, --help        show this help message and exit
  -protein PROTEIN  input the protein you have trained,i.e ALKBH5
```

## Quick Start
- **step1**

run `train.py`
```
>>>python train.py -model_type ensemble -protein ALKBH5 -epoch 5
```
`train.py` can automatically recognizes the detection training environment. If the GPU is available, it will automatically switch to GPU execution.

The training effect will be printed in real time during the run of `train.py`.

After `train.py` is executed, the `ALKBH5_param` directory will be created, and the network parameters obtained by the training will be stored in this directory:
```
>>>ls ./ALKBH5_param
globalCNN_ALKBH5.pkl
localCNN_ALKBH5.pkl
```
- **step2**

run`motif_detection.py`
```
>>>python motif_detection.py -protein ALKBH5
```
`motif_detection.py ` recognizes potential Motifs and outputs the results in the 'ALKBH5_Motif` directory:
```
>>>ls ./ALKBH5_Motif
filter0_logo.fa
filter10_logo.fa
filter11_logo.fa
filter12_logo.fa
filter13_logo.fa
filter14_logo.fa
filter15_logo.fa
filter1_logo.fa
filter2_logo.fa
filter3_logo.fa
filter4_logo.fa
filter5_logo.fa
filter6_logo.fa
filter7_logo.fa
filter8_logo.fa
filter9_logo.fa
filters_meme.txt
```
The output of `filters_meme.txt` can be further compared with the experimentally validated Motif using the `TOMTOM` software (http://meme-suite.org/tools/tomtom).

`filter i _logo.fa` can be used as an input to AME (http://meme-suite.org/tools/ame) to detect the enrichment of Motifs.

# Additional References
waiting for update...
