##--------------Some parameter to setup
data_normalization = False
data_augmented = False
lr=1e-3 
lr_end=1e-5

import numpy as np
import pandas as pd
import sys

import os
import tensorflow as tf

import cv2
import scipy

from keras.models import load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.utils import np_utils

##--------------list of python module
import gen_input as gen_input
import gen_data  as gen_data
import gen_model as gen_model
import gen_vis as gen_vis


os.environ["CUDA_VISIBLE_DEVICES"], PathOutput, group, n_model, epochs, batch_size, fold, init_train = gen_input.read_input()
fname_0, n_group_0, fname_1, n_group_1, fname_2, n_group_2, df = gen_input.generate_idlist(group, fold)

##---- ID Name to data

Data_0,Aux_0,Labels_0,Dat_SBR_0,Dat_VIS_0 = gen_data.name_to_data(fname_0, n_group_0, df, group)
Data_1,Aux_1,Labels_1,Dat_SBR_1,Dat_VIS_1 = gen_data.name_to_data(fname_1, n_group_1, df, group)
Data_2,Aux_2,Labels_2,Dat_SBR_2,Dat_VIS_2 = gen_data.name_to_data(fname_2, n_group_2, df, group)

##---- Check input shape size

input_shape = Data_0[0].shape
aux_shape = Aux_0[0].shape
print(input_shape,aux_shape)
dim = np.amax(Labels_0)+1

#---------Calculate weight for each class
def weight_for_train(labels):
    e = [0 for k in range(dim)]
    for i in range(len(labels)):
        ii=int(labels[i])
        e[ii]=e[ii]+1
    #print('All Data',e)
    #print('-------------------------------')
    f=np.amax(e)/e
    weight ={}
    for i in range(dim):
        weight.update({i: float(f[i])})
    return weight, e


print('Train   ',weight_for_train(Labels_0))
print('Validate',weight_for_train(Labels_1))
print('Test    ',weight_for_train(Labels_2))

weight, e = weight_for_train(Labels_0)

##---Convert to one-hot vector

Labels_0 = np_utils.to_categorical(Labels_0, dim)
Labels_1 = np_utils.to_categorical(Labels_1, dim)
Labels_2 = np_utils.to_categorical(Labels_2, dim)

## -- Normalized data

if data_normalization: 
    Data_0_mean=np.mean(Data_0, axis = 0)
    Data_0=(Data_0-Data_0_mean)
    Data_1=(Data_1-Data_0_mean)
    Data_2=(Data_2-Data_0_mean)

## Data_0=(Data_0)*255
## Data_1=(Data_1)*255
## Data_2=(Data_2)*255

## -- Augmented data
if data_augmented:
    Data_0 = np.concatenate((Data_0, Data_0[:,::-1,:,:,:]), axis = 0)
    Data_1 = np.concatenate((Data_1, Data_1[:,::-1,:,:,:]), axis = 0)
    Labels_0 = np.concatenate((Labels_0, Labels_0), axis = 0)
    Labels_1 = np.concatenate((Labels_1, Labels_1), axis = 0)

## -- END for read data ----

## -- Train model  ------------
def my_learning_rate(epoch):
    x=np.exp(np.log(lr_end/lr)/(epochs-1))
    lrate=(x**epoch)*lr
    return lrate

if init_train==1:
    ##----- GENERATE MODEL FOR TRAINING
    
    if n_model==0:
        model = gen_model.model_cnn3D_0(input_shape, dim, lr=lr)
    elif n_model==1:
        model = gen_model.model_cnn3D_1(input_shape, dim, lr=lr)
    elif n_model==2:
        model = gen_model.model_cnn3D_2(input_shape, dim, lr=lr)
    elif n_model==3:
        model = gen_model.model_cnn3D_3(input_shape, dim, lr=lr)


    print(model.summary())
    
    ##----- TRAIN THE MODEL
    
    if not os.path.exists(PathOutput):
        os.makedirs(PathOutput)
    else:
        for dirName, subdirList, fileList in os.walk(PathOutput):
            for filename in fileList:
                os.remove(PathOutput+filename)
    
    logfile=PathOutput+'allnode_PIN.log'
    csv_logger = CSVLogger(logfile)
    filename="weights.{epoch:03d}-{val_loss:.2f}.hdf5"
    checkpointer = ModelCheckpoint(monitor='val_loss', filepath=PathOutput+filename, verbose=1, save_best_only=True, save_weights_only=True)
    
    #model.set_weights(init_weights)
     
    checkpointer.epochs_since_last_save = 0
    checkpointer.best = np.Inf
    lrate = LearningRateScheduler(my_learning_rate,verbose=1)
    model.fit(Data_0, Labels_0, epochs=epochs, batch_size=batch_size, validation_data=([Data_1,Labels_1]), 
              callbacks=[checkpointer,csv_logger,lrate], verbose=2, class_weight=weight)
    
    ##----------------------- Look for the best model to evaluate
    
    for dirName, subdirList, fileList in os.walk(PathOutput):
        fileList.sort()
    tmp=fileList[len(fileList)-1]
    print(tmp)
    filename=PathOutput+tmp
    model.load_weights(filename)
    model.save(PathOutput+'best_model.hd5')

else:
    model = load_model(PathOutput+'best_model.hd5')

###  ---------Evaluate best model  -------------------###
(loss, accuracy) = model.evaluate(Data_2, Labels_2, batch_size=batch_size, verbose=2)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

y_pred_DL=model.predict(Data_2, batch_size=batch_size)

###----------Visualization --------------------
gen_vis.cls_rep(y_pred_DL, Labels_2,["CNN0","CNN1"])


###---------Machine Learning ------------------
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier


if data_augmented:
    nn_0=Labels_0.shape[0]/2
    nn_1=Labels_0.shape[0]/2

    Labels_0=Labels_0[0:nn_0]
    Labels_1=Labels_1[0:nn_1]


classifier = OneVsRestClassifier(SVC(kernel='rbf'))
y_pred_SVM = classifier.fit(Dat_SBR_0, Labels_0).decision_function(Dat_SBR_2)

print("############ SVM ")
gen_vis.cls_rep(y_pred_SVM, Labels_2, ["SVM0","SVM1"])

y_pred_VIS=np_utils.to_categorical(Dat_VIS_2, dim)
print("############ Visual ")
gen_vis.cls_rep(y_pred_VIS, Labels_2, ["VIS0","VIS1"])

np.savez(PathOutput+'prediction', Labels=Labels_2, DL=y_pred_DL, SVM=y_pred_SVM, VIS=y_pred_VIS)