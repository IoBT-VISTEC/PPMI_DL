# Interpretation of PPMI


## Introduction

This program is a python3 code to construct and interpret the deep learning model for the application of the Parkinson's disease diagnosis using SPECT image. 
The deep learning models were constructed based on the previous 3D-CNN architecture from various research articles. 
User can edit and modify the model to improve the accuracy of the model. The interpretation methods were modified from 2D-CNN works to be applicable with 3D-images data.

This code is divided into 2 parts:

1) The deep learning training for diagnosis the 3D SPECT images of Parkinson's disease.
2) The interpretation of the deep learning model to analyze the prediction results of 3D-images data.

## Prerequisites

The code needs **Python 3** to run.

Python 3 package to be installed before running the code. 

* tensorflow
* keras
* shap

These package can be installed using the Python Package manager
```sh
pip install tensorflow
pip install keras
pip install shap
```


## Usage (Training model)

To train or to test the deep learning model run:
```sh
python main_cnn_svm.py input
```

The file name "input" can editted to specify several model parameters:

```sh
cuda_visible_devices=0  ## The GPU ID to be used
PathOutput=out_test/    ## The output directory of the saved model
group=0                 ## The group to be train 0=["Control", "PD"] and 1=["PD", "SWEDD"]
n_model=4               ## Model types 0 = PD-Net, 1= PD-Net + Batch Norm, 2= Deep PD-Net, 3= Deep PD-Net + Batch Norm
epochs=30               ## Number of epochs
batch_size=4            ## Batch size for training
fold=9                  ## Fold number from 10-fold of data to be tested
init_train=1            ## 0= Load the previous train model from PathOutput, 1= Train for new model
```

## Usage (Interpreting model)
To interpret the deep learning model run:
```sh
python main_interpret.py input [Interpretation Type]
```
The interpretation types are:
* Deep_LIFT
* Grad-CAM
* Guided_Backprop
* Guided_GC
* Saliency
* SHAP


Dataset
=======
The PPMI data can be downloaded directly from the website.

https://www.ppmi-info.org/access-data-specimens/download-data/

<!-- ACKNOWLEDGEMENTS -->
## Citing

* [Arxiv] (https://arxiv.org/abs/1908.11199)
