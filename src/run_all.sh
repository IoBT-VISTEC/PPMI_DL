#!/bin/sh
##while ps -p 28566 > /dev/null; do sleep 60; done;

n_gpu=0
read_file=input
foldername=out_test
for group in 0 
do
for nfold in 0 1 2 3 4 5 6 7 8 9
do
for n_model in 0 1 2 3
do

filename=cls_spect_${n_model}${nfold}

echo "cuda_visible_devices=${n_gpu}
PathOutput=${foldername}/${filename}/
group=$group
n_model=$n_model
epochs=30
batch_size=4
fold=$nfold
init_train=1" > $read_file

### Generate model
python main_cnn_svm.py $read_file > ${foldername}/${filename}.log &&
echo "Job $filename Complete"
####-------------------------------------------------

### Interpret the model
python main_interpret.py $read_file Deep_LIFT >  ${foldername}/overlay_${n_model}${nfold}.log && 
python main_interpret.py $read_file Grad-CAM >>  ${foldername}/overlay_${n_model}${nfold}.log && 
python main_interpret.py $read_file Guided_Backprop >>  ${foldername}/overlay_${n_model}${nfold}.log && 
python main_interpret.py $read_file Guided_GC >>  ${foldername}/overlay_${n_model}${nfold}.log && 
python main_interpret.py $read_file Saliency >>  ${foldername}/overlay_${n_model}${nfold}.log && 
python main_interpret.py $read_file SHAP >>  ${foldername}/overlay_${n_model}${nfold}.log && 
echo "Job $filename Complete"
####-------------------------------------------------


done
done
done


