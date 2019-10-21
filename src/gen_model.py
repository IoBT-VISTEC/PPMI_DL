from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Activation,Dense,Dropout,TimeDistributed,Reshape,Flatten,GRU, Conv3D, LSTM, Convolution2D
from keras.layers import MaxPooling3D, MaxPooling2D, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD, Adam
from keras.layers import Dense
from keras.utils import np_utils

from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import numpy as np


def model_cnn3D_0(input_shape, dim, lr):
    np.random.seed(1337)
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
    
    #with tf.device('/cpu:0'):
    model = Model(inputs=Input_img, outputs=main_output)
    
    sgd = SGD(lr=lr, decay=0, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[ 'accuracy'])

    return model 

def model_cnn3D_1(input_shape, dim, lr):
    np.random.seed(1337)
    Input_img = Input(shape=input_shape, name='input1')
    x = Conv3D(16, kernel_size=(7,7,7), strides=(4,4,4))(Input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2) )(x)

    x = Conv3D(64, kernel_size=(5,5,5), strides=(1,1,1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2) )(x)

    x = Conv3D(256, kernel_size=(2,3,2), strides=(1,1,1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dense(dim)(x)
    main_output = Activation('softmax', name='main_output')(x)
    
    #with tf.device('/cpu:0'):
    model = Model(inputs=Input_img, outputs=main_output)
    
    sgd = SGD(lr=lr, decay=0, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[ 'accuracy'])

    return model 

def model_cnn3D_2(input_shape, dim, lr):
    np.random.seed(1337)
    Input_img = Input(shape=(input_shape), name='input1')
    x = Conv3D(16, kernel_size=(5,5,5), strides=(1,1,1))(Input_img)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2) )(x)
    
    x = Conv3D(32, kernel_size=(3,3,3), strides=(1,1,1))(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2) )(x)
    
    x = Conv3D(64, kernel_size=(3,3,3), strides=(1,1,1))(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2) )(x)
    
    x = Conv3D(128, kernel_size=(3,3,3), strides=(1,1,1))(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2) )(x)
    
    x = Conv3D(256, kernel_size=(2,3,2), strides=(1,1,1))(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dense(dim)(x)
    main_output = Activation('softmax', name='main_output')(x)
    
    #with tf.device('/cpu:0'):
    model = Model(inputs=Input_img, outputs=main_output)
    
    sgd = SGD(lr=lr, decay=0, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[ 'accuracy'])

    return model 


def model_cnn3D_3(input_shape, dim, lr):
    np.random.seed(1337)
    Input_img = Input(shape=(input_shape), name='input1')
    x = Conv3D(16, kernel_size=(5,5,5), strides=(1,1,1))(Input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2) )(x)
    
    x = Conv3D(32, kernel_size=(3,3,3), strides=(1,1,1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2) )(x)
    
    x = Conv3D(64, kernel_size=(3,3,3), strides=(1,1,1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2) )(x)
    
    x = Conv3D(128, kernel_size=(3,3,3), strides=(1,1,1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2) )(x)
    
    x = Conv3D(256, kernel_size=(2,3,2), strides=(1,1,1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dense(dim)(x)
    main_output = Activation('softmax', name='main_output')(x)
    
    #with tf.device('/cpu:0'):
    model = Model(inputs=Input_img, outputs=main_output)
    
    sgd = SGD(lr=lr, decay=0, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[ 'accuracy'])

    return model 

