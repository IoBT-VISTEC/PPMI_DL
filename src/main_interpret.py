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

##################################################################
##########    END READ DATA
##################################################################
import matplotlib
##matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gen_image as gen_image
import gen_interpret as gen_interpret

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
    
#### ------------EDIT THIS TO CHANGE MODEL --------------------
### Function for DeepLIFT
if len(sys.argv)<3:
    print("No INTERPRET OPTION")
    sys.exit()
interpret_name=sys.argv[2]
if (interpret_name=='Deep_LIFT'):
    preds = model.predict(Data_2)
    Data_2_map=gen_interpret.inp_deeplift(PathOutput, Data_2, Labels_2, preds, n_model)

#### ------------(END) EDIT THIS TO CHANGE MODEL --------------------

### Generate data frame for output data

output_file0=PathOutput+"out_overlay_90.csv"
output_file1=PathOutput+"out_overlay_99.csv"
output_file2=PathOutput+"out_overlay_a85.csv"

if os.path.isfile(output_file0)==True:
    df_ol0 = pd.read_csv(output_file0)
    df_ol1 = pd.read_csv(output_file1)
    df_ol2 = pd.read_csv(output_file2)
else:
    df_ol0=pd.DataFrame()
    df_ol1=pd.DataFrame()
    df_ol2=pd.DataFrame()
    
true_map=np.zeros((2,*Data_2.shape[2:4]))
abs_error_map0=np.zeros((2,*Data_2.shape[2:4]))
abs_error_map1=np.zeros((2,*Data_2.shape[2:4]))
abs_error_map2=np.zeros((2,*Data_2.shape[2:4]))

for nn in range(Data_2.shape[0]):
    nlabels=np.argmax(Labels_2[nn])
    class_idx = np.argmax(preds[nn])
    spect_img = Data_2[nn,:,:,:,0]
    output_map = Data_2_map[nn]
    
    
    #Generate the binary image
    binary = gen_image.binary_img(gen_image.slice_ave(spect_img), nlabels)
    
    imgtoplot=gen_image.draw_spect_contour(gen_image.slice_ave(spect_img), binary)
    outname = PathOutput+'fig_'+group[nlabels]+"_{:03d}".format(nn)+'.png'
    gen_image.plot_without_boundary(imgtoplot, outname)
    
    imgtoplot = gen_image.draw_spect_contour(gen_image.slice_ave(output_map), binary)
    outname = PathOutput+'fig_'+group[nlabels]+"_{:03d}_".format(nn)+interpret_name+'.png'
    gen_image.plot_without_boundary(imgtoplot, outname)
    
    ####------------------- 90 percent
    heatmap_bin, TP_bin, FP_bin, FN_bin, TN_bin, TP_num, FP_num, FN_num, TN_num = gen_image.overlay_bin(binary, gen_image.slice_ave(output_map), 0.90)
    df_ol0.loc[nn,'Group']=group[nlabels]
    df_ol0.loc[nn,'Vis Inp']=group[int(Dat_VIS_2[nn])]
    df_ol0.loc[nn,'CNN']=group[class_idx]
    df_ol0.loc[nn,interpret_name+'_dice']=2*TP_num/(2*TP_num+FP_num+FN_num)
    df_ol0.loc[nn,interpret_name+'_sen']=TP_num/(TP_num+FN_num)
    df_ol0.loc[nn,interpret_name+'_spe']=TN_num/(TN_num+FP_num)
    
    imgtoplot = gen_image.draw_spect_contour(heatmap_bin, binary)
    outname = PathOutput+'fig_'+group[nlabels]+"_{:03d}_".format(nn)+interpret_name+'_90.png'
    gen_image.plot_without_boundary(imgtoplot, outname)
    
    abs_error_map0[nlabels] += abs(heatmap_bin-binary)
    
    ####------------------- 99 percent 
    heatmap_bin, TP_bin, FP_bin, FN_bin, TN_bin, TP_num, FP_num, FN_num, TN_num = gen_image.overlay_bin(binary, gen_image.slice_ave(output_map), 0.99)
    df_ol1.loc[nn,'Group']=group[nlabels]
    df_ol1.loc[nn,'Vis Inp']=group[int(Dat_VIS_2[nn])]
    df_ol1.loc[nn,'CNN']=group[class_idx]
    df_ol1.loc[nn,interpret_name+'_dice']=2*TP_num/(2*TP_num+FP_num+FN_num)
    df_ol1.loc[nn,interpret_name+'_sen']=TP_num/(TP_num+FN_num)
    df_ol1.loc[nn,interpret_name+'_spe']=TN_num/(TN_num+FP_num)
    
    imgtoplot = gen_image.draw_spect_contour(heatmap_bin, binary)
    outname = PathOutput+'fig_'+group[nlabels]+"_{:03d}_".format(nn)+interpret_name+'_99.png'
    gen_image.plot_without_boundary(imgtoplot, outname)
    
    abs_error_map1[nlabels] += abs(heatmap_bin-binary)
    
    ####------------------- 85 percent below max value
    heatmap_bin, TP_bin, FP_bin, FN_bin, TN_bin, TP_num, FP_num, FN_num, TN_num = gen_image.overlay_bin_a(binary, gen_image.slice_ave(output_map), 0.85)
    df_ol2.loc[nn,'Group']=group[nlabels]
    df_ol2.loc[nn,'Vis Inp']=group[int(Dat_VIS_2[nn])]
    df_ol2.loc[nn,'CNN']=group[class_idx]
    df_ol2.loc[nn,interpret_name+'_dice']=2*TP_num/(2*TP_num+FP_num+FN_num)
    df_ol2.loc[nn,interpret_name+'_sen']=TP_num/(TP_num+FN_num)
    df_ol2.loc[nn,interpret_name+'_spe']=TN_num/(TN_num+FP_num)
    
    imgtoplot = gen_image.draw_spect_contour(heatmap_bin, binary)
    outname = PathOutput+'fig_'+group[nlabels]+"_{:03d}_".format(nn)+interpret_name+'_a85.png'
    gen_image.plot_without_boundary(imgtoplot, outname)
    
    abs_error_map2[nlabels] += abs(heatmap_bin-binary)
    true_map[nlabels] += binary
    
    plt.close('all')
    
    
df_ol0.to_csv(output_file0, index=0)
df_ol1.to_csv(output_file1, index=0)
df_ol2.to_csv(output_file2, index=0)
num_subject=np.bincount(np.argmax(Labels_2, axis=1))
np.savez(PathOutput+'abs_error_map_'+interpret_name, num_subject=num_subject, ground_truth=true_map, case_90=abs_error_map0, case_99=abs_error_map1, case_a85=abs_error_map2)
print('###########  END  ##########  '+interpret_name)
