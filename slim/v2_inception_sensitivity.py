import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import subprocess
from subprocess import PIPE
import sys

import flatbuffers
import matplotlib.pyplot as plt
import numpy as np
import pprint
import re
import sys

# This hackery allows us to import the Python files we've just generated.
sys.path.append("./tflite/")
import Model



def load_model_from_file(model_filename):
  with open(model_filename, "rb") as file:
    buffer_data = file.read()
  model_obj = Model.Model.GetRootAsModel(buffer_data, 0)
  model = Model.ModelT.InitFromObj(model_obj)
  return model

def save_model_to_file(model, model_filename):
  builder = flatbuffers.Builder(1024)
  model_offset = model.Pack(builder)
  builder.Finish(model_offset, file_identifier=b'TFL3')
  model_data = builder.Output()
  with open(model_filename, 'wb') as out_file:
    out_file.write(model_data)



def test_model_accuracy(tflite_model, model_name):
    command = "python3 ./tflite_eval.py --alsologtostderr  --dataset_dir=../../MobileNet/models/research/imagenet-data     --dataset_name=imagenet     --dataset_split_name=validation     --model_name=" + model_name + " --batch_size=10 --tflite_file=" + tflite_model +" --eval_size=" +eval_size

    orig_acc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    orig_acc = orig_acc.stdout.decode('utf-8')
    if len(orig_acc) > 20:
        try:
            accuracy = float(orig_acc[-6:-1])
        except:
            accuracy = float(orig_acc[-5:-1])
        return accuracy
    else:
        return float(0)


tflite_model='/home/ms75986/Desktop/Cadence/bin_quant/Bin_And_Quant/slim/inception_models/inception_v2_224_quant.tflite'
model_name='inception_v2'
eval_size='1000'
num_layers = 60 #starts with zero
pert = 0.03


inital_accuracy = test_model_accuracy(tflite_model, model_name)
print("inital accuracy:", inital_accuracy)

#load the model
model = load_model_from_file(tflite_model)


modified_file = 'inception_v2_modified_sensitivity.tflite'

params = []
#code to identify large layers
for num,buffer in enumerate(model.buffers):
      if buffer.data is not None:
          params.append([num,len(buffer.data)])
params.sort(reverse=True,key=lambda tup: tup[1])

start_layer = 0
layer_acc = []
for curr_layer in range(start_layer,num_layers): ########## change range from 0,num_layers
    print("************* Processing layer with parameters: **************", curr_layer, params[curr_layer])
    acc = 0
    for times in range(2):
        #perturbation code to identify the sensitive bin
        model = load_model_from_file(tflite_model)
        for num,buffer in enumerate(model.buffers):
            if buffer.data is not None and num == params[curr_layer][0]:
                original_weights = np.frombuffer(buffer.data, dtype=np.uint8)
                v2 = np.add(original_weights,0)      
                break
        if times ==0:
            v2 = np.uint8(v2 + pert*v2)
        else:
            v2 = np.uint8(v2 - pert*v2)
        buffer.data = v2
        save_model_to_file(model, modified_file)
        acc+=test_model_accuracy(modified_file, model_name)

    print("Reported sensitivity acc and inferred sensitivity accuracy :", inital_accuracy, acc/2)
    layer_acc.append([curr_layer, acc/2])

print(layer_acc)
print(model_name)
  


