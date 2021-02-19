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

The details of data used in this code is located in [dat_csv/SPECT_180430_4_30_2018.csv](./dat_csv/SPECT_180430_4_30_2018.csv). All the DICOM files should be placed in the folder name "dat_spect". This allow the program to read the image data filename and labeling the data to match the details in CSV file.

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

For linux user, the shell script is provided for training and interpreting for all 10-fold data. This shell script will automatically adjust all the input parameters. The shell script can be run by

```sh
./run_all.sh
```

## For Developer

If you would like to apply this program to your applications or new models with these dataset, the details for editing each module are listed below. The program was written so that the user can add more new method to analyze the data.

### To add new deep learning model
The user can edit the file [main_cnn_svm.py](./src/main_cnn_svm.py) to add more option for a new model to be used for 3D medical image data. Currently, we provide 4 models based on previous research for Parkinson's disease SPECT image.

```python
    if n_model==0:
        model = gen_model.model_cnn3D_0(input_shape, dim, lr=lr)
    elif n_model==1:
        model = gen_model.model_cnn3D_1(input_shape, dim, lr=lr)
    elif n_model==2:
        model = gen_model.model_cnn3D_2(input_shape, dim, lr=lr)
    elif n_model==3:
        model = gen_model.model_cnn3D_3(input_shape, dim, lr=lr)
```

Next, the user needs to edit the file [gen_model.py](./src/gen_model.py) to include the new model. This is an example of the existing model in the present program.
```python
def model_cnn3D_0(input_shape, dim, lr):
    Input_img = Input(shape=input_shape, name='input1')
    x = Conv3D(16, kernel_size=(7,7,7), strides=(4,4,4))(Input_img)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2) )(x)

    x = Conv3D(64, kernel_size=(5,5,5), strides=(1,1,1))(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2) )(x)

    x = Conv3D(256, kernel_size=(2,3,2), strides=(1,1,1))(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dense(dim)(x)
    main_output = Activation('softmax', name='main_output')(x)
    
    model = Model(inputs=Input_img, outputs=main_output)
    
    sgd = SGD(lr=lr, decay=0, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[ 'accuracy'])

    return model 
```

### To add new interpretation model
The user can edit the file [main_cnn_svm.py](./src/main_cnn_svm.py) to add more option for a new interpretaion model to be used for this data. Currently, we have modified 6 interpretation methods to be applicable for 3D medical images.

```python
if (interpret_name=='Grad-CAM'):
    preds = model.predict(Data_2)
    Data_2_map=gen_interpret.inp_gradcam(PathOutput, Data_2, Labels_2, preds, n_model, model, interpret_name)
```

Next, the user needs to edit the file [gen_interpret.py](./src/gen_interpret.py) to include the new model. This is an example of the existing model in the present program. The interpretation model need to be adjusted so that the model becomes consistent with the deep learning model that will be used. An example from Grad-CAM is shown below. We introduce some parameter of the layer that will be needed for the analysis.
```python
def inp_gradcam(PathOutput, Data_2, Labels_2, preds, n_model, model, interpret_name):
    if n_model==0 or n_model==1:
        deeplift_layer0="input1_0"
        deeplift_layer1="dense_2_0"
        compile_guided_layer="activation_3"
        grad_cam_layer="conv3d_2"
    elif n_model==2 or n_model==3:
        deeplift_layer0="input1_0"
        deeplift_layer1="dense_2_0"
        compile_guided_layer="activation_5"
        grad_cam_layer="conv3d_4"
```

<!-- ACKNOWLEDGEMENTS -->
## Citing

* [Arxiv] (https://arxiv.org/abs/1908.11199)
