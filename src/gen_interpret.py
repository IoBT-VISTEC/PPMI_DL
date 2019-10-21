import numpy as np

import sys
sys.path.append('.')
import deeplift
from deeplift.layers import NonlinearMxtsMode
from deeplift.conversion import kerasapi_conversion as kc
from deeplift.util import get_integrated_gradients_function

def inp_deeplift(PathOutput, Data_2, Labels_2, preds, n_model):
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
    
    saved_model_file = PathOutput+'best_model.hd5'
    
    revealcancel_model = kc.convert_model_from_saved_files(
                                h5_file=saved_model_file,
                                nonlinear_mxts_mode=NonlinearMxtsMode.RevealCancel)
    
    revealcancel_func = revealcancel_model.get_target_contribs_func(find_scores_layer_name=deeplift_layer0, pre_activation_target_layer_name=deeplift_layer1)
    
    from collections import OrderedDict
    method_to_task_to_scores = OrderedDict()
    for method_name, score_func in [
                                   ('revealcancel', revealcancel_func),
    ]:
        print("Computing scores for:",method_name)
        method_to_task_to_scores[method_name] = {}
        for task_idx in range(2):
            print("\tComputing scores for task: "+str(task_idx))
            scores = np.array(score_func(
                        task_idx=task_idx,
                        input_data_list=[Data_2],
    #                     input_references_list=[np.zeros_like(Data_2)],
                        input_references_list=[np.average(Data_2, axis=0)],
                        batch_size=4,
                        progress_update=None))
            method_to_task_to_scores[method_name][task_idx] = scores
    
    # Generate the heatmap
    Data_2_map = np.zeros(Data_2.shape[0:4])
    
    for nn in range(Data_2.shape[0]):
        Data_test=Data_2[nn:nn+1]
        nlabels=np.argmax(Labels_2[nn])
        class_idx = np.argmax(preds[nn])
        
        
        ## Copy DeepLIFT image
        deeplift_map=method_to_task_to_scores['revealcancel'][class_idx][nn,:,:,:,0]
        
        Data_2_map[nn] = deeplift_map
        
    return Data_2_map


### -------------GRAD-CAM ----------------###
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential, Model, load_model
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
import os

def image_resize(img,nx,ny,nz):
    width = ny
    height = nz
    img1 = np.zeros((img.shape[0], width, height))

    for idx in range(img.shape[0]):
        tmp = img[idx, :, :]
        img_sm = cv2.resize(tmp, (height, width), interpolation=cv2.INTER_LINEAR)
        img1[idx, :, :] = img_sm

    width = nx
    height = ny
    img2 = np.zeros((width, height, img1.shape[2]))

    for idx in range(img1.shape[2]):
        tmp = img1[:, :, idx]
        img_sm = cv2.resize(tmp, (height, width), interpolation=cv2.INTER_LINEAR)
        img2[:, :, idx] = img_sm
    return img2

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='block5_conv3'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name, PathOutput):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model=load_model(PathOutput+'best_model.hd5')
    return new_model

def grad_cam(input_model, image, category_index, layer_name):

    nb_classes = 2
    
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    x = Lambda(target_layer, output_shape = target_category_loss_output_shape)(input_model.output)
    model = Model(inputs=input_model.input, outputs=x)

    loss = K.sum(model.output)

    conv_output =  [l for l in model.layers if l.name==layer_name][0].output
    
    
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1, 2))
    cam = np.ones(output.shape[0 : 3], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, :, i]

    cam = np.maximum(cam, 0)
    if np.max(cam) != 0:
        heatmap = cam / np.max(cam)
    else:
        heatmap = cam
    
    return heatmap


def inp_gradcam(PathOutput, Data_2, Labels_2, preds, n_model, model):
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

    register_gradient()
    guided_model = modify_backprop(model, 'GuidedBackProp', PathOutput)
    saliency_fn = compile_saliency_function(guided_model, compile_guided_layer)
    
    # Generate the heatmap
    Data_2_map = np.zeros(Data_2.shape[0:4])
    
    for nn in range(Data_2.shape[0]):
        Data_test=Data_2[nn:nn+1]
        nlabels=np.argmax(Labels_2[nn])
        class_idx = np.argmax(preds[nn]) 
        
        #### Generate Grad-CAM
        img = Data_test[0]
        heatmap=grad_cam(model, Data_test, class_idx, grad_cam_layer)
        img_h=image_resize(heatmap, img.shape[0], img.shape[1],img.shape[2])
        
        ### Generate Guided Grad-CAM and Guided Backprop
        saliency = saliency_fn([Data_test, 0])
        saliency=np.array(saliency[0])
        
        gradcam = saliency[0] * img_h[..., np.newaxis]
    
        gradcam = gradcam[:,:,:,0]
        saliency = saliency[0,:,:,:,0]
        
        Data_2_map[nn] = img_h
        
    return Data_2_map

