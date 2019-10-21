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