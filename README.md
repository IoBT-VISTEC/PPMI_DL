# Interpretation of PPMI


## Introduction

This program is a Python 3 code to construct and interpret the deep learning model for the application of the Parkinson's disease diagnosis using SPECT image. 
The deep learning models were constructed based on the previous 3D-CNN architecture from various research articles. 
Users can edit and modify the model to improve the accuracy of the model. The interpretation methods were modified from 2D-CNN works to be applicable with 3D-images data.

This code is divided into 2 parts:

1) The deep learning training for diagnosis the 3D SPECT images of Parkinson's disease.
2) The interpretation of the deep learning model to analyze the prediction results of 3D-images data.

## Prerequisites

The code needs **Python 3** to run.

Python 3 package to be installed before running the code. 

* nibabel
* pydicom
* tensorflow
* keras
* shap

These package can be installed using the Python Package manager
```sh
pip install nibabel
pip install pydicom
pip install tensorflow
pip install keras
pip install shap
```
## Dataset
The PPMI data can be requested directly from the website.

* PPMI (https://www.ppmi-info.org/access-data-specimens/download-data/)

The details of data used in this code is located in "dat_csv/SPECT_180430_4_30_2018.csv". All the DICOM files should be placed in the folder name "dat_spect". This allow the program to read the image data filename and labeling the data to match the details in CSV file.

## Usage (Training model)

To train or to test the deep learning model, the users should enter the "src" folder and run:
```sh
python main_cnn_svm.py input
```

The file name "input" can be editted to specify several model parameters:

```sh
cuda_visible_devices=0  ## The GPU ID to be used
PathOutput=out_test/    ## The output directory of the saved model
group=0                 ## The group to be train 0=["Control", "PD"] and 1=["PD", "SWEDD"]
n_model=4               ## Model types 0 = PD-Net, 1= PD-Net + Batch Norm, 2= Deep PD-Net, 3= Deep PD-Net + Batch Norm
epochs=30               ## Number of epochs
batch_size=4            ## Batch size for training
fold=9                  ## Fold number from 10-fold of data to be tested
init_train=1            ## 0= Load the previous train model from PathOutput, 1= Train for new model
```

## Usage (Interpretation model)
To interpret the deep learning model, the users should change the directory to "src" folder and run:
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

This program is used to load the already trained model to generate the 3D heatmap from the interpretation model.
The program also analyzes some data to measure the performance of the interpretation model.
The file name "input" can be editted to specify several model parameters:

```sh
cuda_visible_devices=0  ## The GPU ID to be used
PathOutput=out_test/    ## The output directory of the saved model
group=0                 ## The group to be train 0=["Control", "PD"] and 1=["PD", "SWEDD"]
n_model=4               ## Model types 0 = PD-Net, 1= PD-Net + Batch Norm, 2= Deep PD-Net, 3= Deep PD-Net + Batch Norm
epochs=30               ## Number of epochs
batch_size=4            ## Batch size for training
fold=9                  ## Fold number from 10-fold of data to be tested
init_train=1            ## Any number can be used because this program can only load the saved model.
```
**Remark:** The parameters "group", "n_model" must be the same with the saved model so that the interpretation method can select the same layer with the saved model. 

**Remark:** The parameters "fold" must also be consistent with the saved model. This parameter will select the testing data that have not been used during the training.

## Usage for Linux user

For linux users, the shell script is provided for training and interpreting for all 10-fold data. This shell script will automatically adjust all the input parameters. The shell script can be run by

```sh
./run_all.sh
```

## For Developer

If you would like to apply this program to your applications or new models with these dataset, the details of each module are listed below:

### gen_data.py

### gen_image.py

### gen_input.py


<!-- ACKNOWLEDGEMENTS -->
## Citing

* [Arxiv] (https://arxiv.org/abs/1908.11199)
