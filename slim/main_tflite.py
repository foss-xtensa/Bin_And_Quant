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

def test_model_accuracy(tflite_model, model_name):
    command = "python3 ./tflite_eval.py --alsologtostderr  --dataset_dir=../../MobileNet/models/research/imagenet-data     --dataset_name=imagenet     --dataset_split_name=validation     --model_name=" + model_name + " --batch_size=10 --tflite_file=" + tflite_model 

    orig_acc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    orig_acc = orig_acc.stdout.decode('utf-8')
    #accuracy = float(orig_acc[-6:-1])
    #return accuracy
    try:
        accuracy = float(orig_acc[-6:-1])
    except:
        accuracy = accuracy = float(orig_acc[-5:-1])
    return accuracy

inital_accuracy = test_model_accuracy(tflite_model, model_name)
print("inital accuracy:", inital_accuracy)

#load the model
model = load_model_from_file(tflite_model)
max_bins = 16
curr_acc = inital_accuracy - 3

top_acc = curr_acc
top_acc_bins = []

layer_size = 1048570
modified_file = 'mobilenet_modified.tflite'
top_model_file = 'mobilenet_top_acc_model.tflite'

#load the inital layer parameters of the FC layer
for buffer in model.buffers:
  if buffer.data is not None and len(buffer.data) > layer_size:
    original_weights = np.frombuffer(buffer.data, dtype=np.uint8)
    v2 = np.add(original_weights,0)
    v2_min = v2.min()
    v2_max = v2.max()

RangeValues = [v2_min, np.mean(v2) - np.std(v2), np.mean(v2), np.mean(v2) + np.std(v2), v2_max]
total_bins = len(RangeValues)
while ((curr_acc <= inital_accuracy - 0.2) and total_bins <= max_bins):
  model = load_model_from_file(tflite_model)
  for buffer in model.buffers:
    if buffer.data is not None and len(buffer.data) > layer_size:
      original_weights = np.frombuffer(buffer.data, dtype=np.uint8)
      v2 = np.add(original_weights,0)

      for x in range(len(RangeValues) - 1):
        indices = np.where(np.logical_and(v2>=RangeValues[x], v2<=RangeValues[x+1]))
        v2[indices] = np.uint8((RangeValues[x] + RangeValues[x+1])/2)

      buffer.data = v2
  save_model_to_file(model, modified_file)
  curr_acc = test_model_accuracy(modified_file, model_name)

  print("Accuracy for range values:", curr_acc, RangeValues)

  if curr_acc >= top_acc:
    top_acc = curr_acc
    top_bins = RangeValues
    save_model_to_file(model, top_model_file)

  #perturbation code to identify the sensitive bin
  bin_acc = []
  for times in range(len(RangeValues) - 1):
    model = load_model_from_file(tflite_model)
    for buffer in model.buffers:
      if buffer.data is not None and len(buffer.data) > layer_size:
        original_weights = np.frombuffer(buffer.data, dtype=np.uint8)
        v2 = np.add(original_weights,0)      

        indices = np.where(np.logical_and(v2>=RangeValues[times], v2<=RangeValues[times+1]))
        v2[indices] = np.uint8(v2[indices] + 0.2*v2[indices])

        buffer.data = v2

    save_model_to_file(model, modified_file)
    bin_acc.append(test_model_accuracy(modified_file, model_name))

  print("The most sensitive bin here is:", bin_acc, np.array(bin_acc).argmin())
  to_modify = np.array(bin_acc).argmin()
  middle_bin = (RangeValues[to_modify] + RangeValues[to_modify+1]) / 2
  RangeValues.insert(to_modify+1, middle_bin)
  print("New Range Values:", RangeValues, len(RangeValues))
  total_bins = len(RangeValues)


print("Reported top_accuracy and infered test accuracy :", top_acc, test_model_accuracy(top_model_file, model_name))
print("The RangeValues are:", RangeValues)
print("total_bins", len(RangeValues) - 1)
print("The model is saved in", top_model_file)
  


