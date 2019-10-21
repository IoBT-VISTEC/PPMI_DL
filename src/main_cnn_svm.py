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


from keras.models import Sequential, load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.utils import np_utils

##--------------list of python module
import gen_input as gen_input
import gen_data  as gen_data
import gen_model as gen_model
import gen_vis as gen_vis


os.environ["CUDA_VISIBLE_DEVICES"], PathOutput, group, n_model, epochs, batch_size, fold, init_train = gen_input.read_input()

input_file="input_label_spect.csv"
if os.path.isfile(input_file)==False:
    gen_input.generate_input_label_file(input_file)

df = pd.read_csv(input_file)
df=df.dropna(subset=['VISINTRP'])
df=df.dropna(subset=['CAUDATE_R'])
df=df.dropna(subset=['NHY_0'])
df=df.dropna(subset=['MDS3_0'])
df=df.reset_index(drop=True)

### ---------- Select only group data --------------
df=df[df['Group'].isin(group)]
df=df[df['NHY_0'].isin([0,1,2,3,4,5])]
df=df.reset_index(drop=True)

### ----------- Only subject with the first visit will be used ---------------
lst=[]
for i in range(len(df['Subject'].unique())):
    ii = df['Subject'].unique()[i]
    tmp=df.loc[df['Subject'] == ii]
    e=[]
    for j in range(len(tmp)):
        ii=tmp.index[j]
        nd2=tmp.loc[ii, 'Acq Date']
        e.append(nd2)

    ii=tmp.index[e.index(min(e))]
    lst.append(ii)

df=df.ix[lst]
df=df.reset_index(drop=True)

### ----------- Only subject that has the progression score will be used ---------------
## lst=[]
## for i in range(len(df)):
##     if df.loc[i, 'INFODT_1']== df.loc[i, 'INFODT_0']:
##         lst.append(i)
## print("SAME DATE ROW NUMBER",lst)
## df=df.drop(lst)   
## df=df.reset_index(drop=True)

print('Number of Subject=',df['Subject'].nunique())
print(df['Group'].value_counts())

#--------- initialize the data filename list and labels list
n_dat = df.shape[0]
nn = int(n_dat/10)+1
tmp = np.zeros(n_dat, dtype=int)
fname = np.zeros((10,nn), dtype=int)
n_group=np.zeros((10,2), dtype=int)
df = df.sample(frac=1, random_state=1).reset_index(drop=True)

for i in range(n_dat):
    ix=i%10
    iy=int(i/10)
    fname[ix][iy]=int(df.loc[i, 'Image Data ID'])
    n_group[ix][group.index(df.loc[i, 'Group'])] +=1

##-------- Generate 10 fold data

fname=np.roll(fname, fold, axis=0)
n_group=np.roll(n_group, fold, axis=0)

fname_0=np.reshape(fname[0:8], nn*8)
fname_1=np.reshape(fname[8], nn)
fname_2=np.reshape(fname[9], nn)
n_group_0=np.sum(n_group[0:8], axis=0)
n_group_1=n_group[8]
n_group_2=n_group[9]


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

y_pred=model.predict(Data_2, batch_size=batch_size)
y_true=Labels_2

###----------Visualization --------------------
gen_vis.cls_rep(y_pred, y_true,["CNN0","CNN1"])


###---------Machine Learning ------------------
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score,train_test_split, GridSearchCV, KFold

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


if data_augmented:
    nn_0=Labels_0.shape[0]/2
    nn_1=Labels_0.shape[0]/2

    Labels_0=Labels_0[0:nn_0]
    Labels_1=Labels_1[0:nn_1]

all_x = Dat_SBR_0
all_y = np.argmax(Labels_0, axis=1)
y_true= np.argmax(Labels_2, axis=1)

lr = SVC(kernel='rbf')
lr.fit(all_x, all_y)
y_pred=lr.predict(Dat_SBR_2)

print("############ SVM ")
gen_vis.cls_rep_nm(y_pred, y_true, ["SVM0","SVM1"])

y_pred=Dat_VIS_2
print("############ Visual ")
gen_vis.cls_rep_nm(y_pred, y_true, ["VIS0","VIS1"])
