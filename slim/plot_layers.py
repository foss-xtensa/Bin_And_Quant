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


model_name = 'mobilenet_v2'
#tflite_model='/home/ms75986/Desktop/Cadence/bin_quant/Bin_And_Quant/slim/mobilenet_models/mobilenet_v2_1.0_224_quant.tflite'
tflite_model='/home/ms75986/Desktop/Cadence/bin_quant/Bin_And_Quant/slim/inception_models/inception_v1_224_quant.tflite'

#load the model
model = load_model_from_file(tflite_model)

params = []
#code to identify large layers
for num,buffer in enumerate(model.buffers):
      if buffer.data is not None:
          params.append([num,len(buffer.data)])
params.sort(reverse=True,key=lambda tup: tup[1])

curr_layer = 2

for num,buffer in enumerate(model.buffers):
    if buffer.data is not None and num == params[curr_layer][0]:
        original_weights = np.frombuffer(buffer.data, dtype=np.uint8)
        v2 = np.add(original_weights,0)
        v2_min = v2.min()
        v2_max = v2.max()
        break

RangeValues = [v2_min, np.mean(v2) - np.std(v2), np.mean(v2), np.mean(v2) + np.std(v2), v2_max]

print(RangeValues, len(v2))

fig, (ax1, ax2) = plt.subplots(1, 2)
name = model_name + '_' + str(curr_layer) + ' original hist bins, binned hist' 
fig.suptitle(name)
ax1.hist(v2)
for num in RangeValues:
    ax1.axvline(x=num)
ax2.hist(v2, bins=RangeValues)
plt.show()




