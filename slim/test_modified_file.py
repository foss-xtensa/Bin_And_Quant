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



#tflite_model='/home/ms75986/Desktop/Cadence/bin_quant/Bin_And_Quant/slim/inception_models/inception_v1_224_quant.tflite'
tflite_model='/home/ms75986/Desktop/Cadence/bin_quant/Bin_And_Quant/slim/mobilenet_models/mobilenet_v2_224_100/model.tflite'
model_name='mobilenet_v2' #'inception_v1'



modified_file = 'mobilenet_modified.tflite'
top_model_file = 'mobilenet_top_acc_model.tflite'
num_layers = 2 #starts with zero



to_test = top_model_file
model = load_model_from_file(to_test)

params = []
#code to identify large layers
for buffer in model.buffers:
      if buffer.data is not None:
          params.append(len(buffer.data))
params.sort(reverse=True)

model = load_model_from_file(to_test)
curr_layer = 0

while curr_layer <= num_layers:
    #load the inital layer parameters of the FC layer
    for buffer in model.buffers:
      if buffer.data is not None and len(buffer.data) >= 200000:
        original_weights = np.frombuffer(buffer.data, dtype=np.uint8)
        v2 = np.add(original_weights,0)
        print(len(buffer.data), np.unique(v2), len(np.unique(v2)))
        #break
    curr_layer +=1
  


