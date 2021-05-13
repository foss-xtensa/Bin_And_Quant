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



model_name='inception_v1'



modified_file = 'mobilenet_modified.tflite'
top_model_file ='inception_v1_result_models/inception_top_acc_model_1_.tflite'
num_layers = 3 #starts with zero



to_test = top_model_file
model = load_model_from_file(to_test)

params = []
#code to identify large layers
for num,buffer in enumerate(model.buffers):
      if buffer.data is not None:
          params.append([num,len(buffer.data)])
params.sort(reverse=True,key=lambda tup: tup[1])

model = load_model_from_file(to_test)
curr_layer = 0

while curr_layer <= num_layers:
    #load the inital layer parameters of the FC layer
    for num,buffer in enumerate(model.buffers):
        if buffer.data is not None and num == params[curr_layer][0]:
            original_weights = np.frombuffer(buffer.data, dtype=np.uint8)
            v2 = np.add(original_weights,0)
            print(len(buffer.data), np.unique(v2), len(np.unique(v2)))
            break
    curr_layer +=1
  


