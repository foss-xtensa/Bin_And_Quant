import os
import subprocess
from subprocess import PIPE
import sys

tflite_model='/home/ms75986/Desktop/Cadence/bin_quant/Bin_And_Quant/slim/inception_models/inception_v1_224_quant.tflite'
model_name='inception_v1'


command = "python3 ./tflite_eval.py --alsologtostderr  --dataset_dir=../../MobileNet/models/research/imagenet-data     --dataset_name=imagenet     --dataset_split_name=validation     --model_name=" + model_name + " --batch_size=10 --tflite_file=" + tflite_model 

orig_acc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
orig_acc = orig_acc.stdout.decode('utf-8')
orig_acc = float(orig_acc[-6:-1])
print("orig_acc", orig_acc)


