# Pytorch ML Pipeline

This repository contains a starter implementation of a Machine Learning pipeline with Pytorch.

The goal of this implementation is to be simple, flexible, and easy to extend to your own projects. Its modular structure will allow to build complex and various models, without compromising the integrity and simplicity of the code.

Notice that this implementation is a work in progress -- currently we restrict to a classification problem architecture, along with its specific metrics and building blocks. But this is not a limitation: once the simplicity of the repository is grasped, extending it to additional tasks will be easy enough.

## Table of Contents

* [Overview](#overview)
    * [Generic Pipeline](#generic-pipeline)
    * [Pytorch Building Blocks](#pytorch-building-blocks)
    * [Folder Structure](#folder-structure)
* [Installation](#installation)
    * [Prerequisites](#prerequisites)
    * [Cloning](#cloning)
* [Usage](#usage)
    * [Modular Usage](#modular-usage)
    * [Generic Usage](#generic-usage)

## Overview

Each machine learning pipeline can be decomposed in pretty much the same building blocks. With this in mind, we propose a modular implementation where one single class (Model) captures all pipeline features:
- its attributes correspond to the overall procedure building blocks (model architectures, criterions, hyperparameters, ...)
- its methods consitst in the actions one can perform (preprocess, train, save, ...)

### Generic Pipeline

Usual machine learning workflow involves the following steps:

1. Load and preprocess data
2. Extract features
3. Split into training, validation, and test sets
4. Train model
5. Evaluate model
6. Tune hyperparameters
7. Test model
8. Collect results

### Pytorch Building Blocks

Pytorch is becoming the most used deep learning framework in academic research.

Typical deep learning implementation in Pytorch requires the following building blocks:
* model architecture
* criterion (loss, ..)
* optimizer

Additional regularization techniques can be added to avoid overfitting. In particular we implemented:
* early stopping
* learning rate scheduler


### Folder Structure
We created a modular structure, where each folder represents a different building block and contains its possible implementations. Then, to build the overall model the user have two options: 
- specify the module names in a json parameters file
- load the modules directly in the main script

[//]:<> (user can easily choose one of the many possible combinations by specifiyng the module names in parameters.json or by building the modules)

[//]:<> (each file correspond to a specific version of the c. Hence, during the training the user can simply choose different elements in each folder according to the model they want to create.)


[//]:<> (The folder structure is meant to be a modular structure where each folder represents a different building block and could contain various versions of it. Hence, during the training the user can simply choose different elements in each folder according to the model they want to create.)


[//]:<> (and in the same folder different versions are collected )

[//]:<> (separated from the others and can be easily modified without affecting the )

```
.
├── README.md
├── data
│   └── download_data.py
├── main.py
├── model
│   ├── architectures
│   │   └── feedforwardnet.py
│   ├── earlystopping
│   │   └── earlystopping.py
│   ├── losses
│   │   └── binarycrossentropy.py
│   ├── model.py
│   ├── optimizers
│   │   └── adam.py
│   ├── parameters.json
│   ├── schedulers
│   │   └── steplr.py
│   └── train.py
├── outputs
│   ├── model.h5
│   └── model.json
└── requirements.txt
```

## Installation

### Prerequisites

Prerequisites can be installed through the requirements.txt file as below
```
$ pip install requirements.txt
```

### Cloning

Clone the repository to your local machine

```
$ git clone https://github.com/m-bini/pytorch-classification-pipeline
```

## Usage
-- to be finished --

### Modular Usage
-- to be finished --
Example with MNIST dataset:

```
# parameters path
parameters_path = "model/parameters.json"
# load training and validation datasets
train_set, test_set = download_mnist_datasets()

# create Model instance
M = Model(parameters_path=parameters_path)
# do training
M.do_train(train_set, test_set)
# save model weights and parameters
M.save(inplace=True)
M.history.plot() 

```

### Generic Usage
-- to be finished --



<!-- In the meantime, follow the main.py file


#### 1. Creating an instance of the Model class
```
from model.model import Model

M = Model()
```

#### 2. Loading the building blocks -->







<!-- ## Example workflow (see Workshop or Tutorial ipython notebook for more details!)
//
### 0. Format data

Your data should look something like this:

| ID   | Class    | Feature_1   | Feature_2   | Feature_n   |
|---    |---    |---    |---    |---    |
| instanceA    |  1     | sunny     | 98     | 1     |
| instanceB    |  0    |  overcast     | 87     | 0     |
| instanceC   |  0     |  rain    | n/a     | 1     |
| instanceD   | 1     |  sunny    | 73     | 1     |
| instanceE   | unknown     |  overcast    | 75     | 0     |

Where the first column contains the instance IDs, one column contains the value you want to predict with name (default = 'Class', specify your own column name using -y_name), and the remaining columns contain the predictive features (i.e. independent variables).

If you want to classify instances by class (in the above example 1 vs. 0), use ML_classification.py. You can specify what classes you want to include in your model using -cl_train. This is useful if your dataset also contains instances with other values in the Class column, such as "unknown", that you want to apply your trained model to. If you want to predict the value in the y_name column, use ML_regression.py. 


### 1. Clean your data

```
python ML_preprocess.py -df data.txt -na_method median -onehot t -
```

### 2. Define a testing set (test_set.py)

```
python test_set.py -df data_mod.txt -use 1,0 -type c -p 0.1 -save test_instances.txt
```

### 3. Select the best subset of features to use as predictors (Feature_Selection.py)

```
python Feature_Selection.py -df data_mod.txt -cl_train 1,0 -type c -alg lasso -p 0.01 -save top_feat_lasso.txt
```

### 4. Train and apply a classification (ML_classification.py) or regression (ML_regression.py) machine learning model

Example using the data shown above:
```
python ML_classification.py -df data_mod.txt -test test_instances.txt -cl_train 1,0 -alg SVM -apply unknown
```

Example of a multiclass prediction where classes are A, B, and C:
```
python ML_classification.py -df data_mod.txt -test test_instances.txt -cl_train A,B,C -alg SVM -apply unknown
```

Example of a regression prediction (e.g. predicting plant height in meters):
```
python ML_regression.py -df data_mod.txt -test test_instances.txt -y_name height -alg SVM -apply unknown
```

**For more options, run either ML_classification.py or ML_regression.py with no parameters or with -h**

### 5. Assess the results of your model (output from the ML_classification/ML_regression scripts with additional options in scripts_PostAnalysis

**See scripts_PostAnalysis/README.md for more information on additional post-modeling analysis and figure making.**

The following files are generated by default:

- **data.txt_results:** A summary of the model generated (e.g. what algorithm/parameters were used, number of instances/features, etc.) and the results from applying that model during validation and on the test set. These results are similar to what is printed on the commond line, but more performance metrics are provided, including performance metrics specific to your type of model (i.e. binary classification, multiclass, regression). For example, for binary and multiclass classification models you will see two additional sections: the Mean Balanced Confusion Matrix (CM) and the Final Full CM. The mean balanced CM is generated by taking the average number of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN) across all replicate (which have been downsampled randomly to be balanced). The Final Full CM represents the final TP, TN, FP, and FN results from the final pos/neg classifications (descirbed in data.txt_scores) for all instances in your input dataset. 

- **data.txt_scores:** This file includes the true value/class for each instance and the predicted value/class & predicted probability (pp) for each instance for each replicate of the model (-n). The pp score represents how confident the model was in its classification, where a pp=1 means it is certain the instance is positive and pp=0 means it is certain the instance is negative. For multiclass models, the class with the greatest pp is selected as the predicted class. For binary models, for each replicate, an instance is classified as pos if pp > threshold, which is defined as value between 0.001-0.999 that maximises the F-measure. While the performance metrics generated by the pipeline are calcuated for each replicate independently, we want to be able to make a final statement about which instances were called as positive and which were called as negative. You'll find those results in this file. To make this final call we calculated the mean threshold and the mean pp for each instance and called the instance pos if the mean pp > mean threshold. 

- **data.txt_imp:** the importance of each feature in your model. For RF and GTB this score represents the [Gini Index](https://medium.com/the-artificial-impostor/feature-importance-measures-for-tree-models-part-i-47f187c1a2c3), while for LogReg and SVM it is the [coefficient](https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d). SVM with non-linear kernels (i.e. poly, rbf) does not report importance scores.

- **data.txt_GridSearch:** the average model performance across the whole parameter space tested via the grid search (i.e. every possible combination of parameters). 

- **data.txt_BalancedID:** Not generated for ML_regression.py models. Each row lists the instances that were included in each replicate (-n) after downsampling. 

Additional Notes for Multiclass models:

- *An important note: For binary classification using balanced datasets, you would expect a ML model that was just randomly guessing the class to be correct ~50% of the time, because of this the random expectation for performance metrics like AUC-ROC and the F-measure are 0.50. This is not the case for multi-class predictions. Using our model above as an example, a ML model that was randomly guessing top, middle, or bottom, would only be correct ~33% of the time. That means models performing with >33% accuracy are performing better than random expectation.*

- *There are two types of performance metrics for multi-class models, commonly referred to as macro and micro metrics. Micro performance metrics are generated for each class in your ML problem. For example, from our model we will get three micro F-measures (F1-top, F1-middle, F1-bottom). These micro scores are available in the *_results* output file. Macro performance metrics are generated by taking the average of all the micro performance metrics. These scores are what are printed in the command line.*

## Additions:
The trained model is now saved using joblib's dump function. `joblib` is now an environment requirement.


## TO DO LIST

Major:
    - Implement feature selection during the training step

Minor 
    - Allow user to set custom seed
    - Add additional classification models: Naive Bayes, basic neural network (1-2 layers)
    - Look into using MCC as a performance metric - would be useful for selecting the threshold since it doesn't depend on the ratio of +/- instances (https://en.wikipedia.org/wiki/Matthews_correlation_coefficient)
    - Incorporate PCA summary features into pre-processing script
 -->
