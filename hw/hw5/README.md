# Assignment 5

## Overview
In this assignment, we implemented multimodal deep-learning methods to improve classification accuracy. We experimented with the Written and spoken digits database.

## Prerequisites
For this assignment, I use these packages that are not in Python's standard library. Make sure you pip install these packages first.
* numpy
* torch
* matplotlib
* scikit-learn
* pandas
* matplotlib-venn

## Get Started
In the folder, there are two main items: the code folder and the report (Mu-Ruei_Tseng_HW5.pdf). To run the code, make sure to download the data from [here](https://www.kaggle.com/competitions/sp24-tamu-csce-633-600-machine-learning/data). Extract the data, rename it as 'data', and put it in the code folder. In the code folder, we provide 5 main scripts (train.py, test.py, inference.py, dimension_reduction.py, and comparison.py).

### train.py
This script is used to train the specific model we want. Simply run:
```
python train.py 
```
It will prompt the options for you to choose the model. 
* ex: Choose the model (0: for ImageNet_CNN, 1: for AudioNet_CNN, 2: for AudioNet_LSTM, 3: for HybridNet_CNN, 4: for HybridNet_CNN_LSTM) 
The corresponding checkpoint will be saved in the checkpoints folder.

### test.py and inference.py
Similar to train.py, after the model is trained, one can use test.py to evaluate the model's performance. Since we split the training data into train, val, and test sets, test.py will use the test set to evaluate the accuracy and f1 score. 

On the other hand, inference.py is used to generate the prediction results on the <b>x_test_wr.npy</b> and <b>x_test_sp.npy</b> data for submission to the Kaggle challenge. The result will be saved in the 'results' folder. The best result I achieved on Kaggle is labeled as "Mu-Ruei_Tseng_preds_best.csv".

### dimension_reduction.py
This file is used to perform dimension reduction and visualize the embedding of the training data in a 2-dimensional space. This allows us to observe the clusters formed by different classes and assess the quality of the embeddings in terms of class separability and overlap. The result will be saved in the "demo" folder.

### comparison.py
This file is used to generate the Venn diagram shown in the report. One must finished training all the models first before running this script.

