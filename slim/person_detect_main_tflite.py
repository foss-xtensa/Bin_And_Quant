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
    command = "python3 ./person_tflite_eval.py --alsologtostderr  --dataset_dir=../../MobileNet/models/research/slim/datasets/visual_data/     --dataset_name=visualwakewords     --dataset_split_name=val --eval_image_size=96 --use_grayscale=True --model_name=" + model_name + " --batch_size=10 --tflite_file=" + tflite_model +" --eval_size=" +eval_size

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


#tflite_model='/home/ms75986/Desktop/Cadence/bin_quant/Bin_And_Quant/slim/mobilenet_models/mobilenet_v1_1.0_224_quant.tflite'
#change#
tflite_model='person_detect_uint8_top_acc_model_6_.tflite'

model_name='mobilenet_v1'
eval_size='1000'

inital_accuracy = test_model_accuracy(tflite_model, model_name)
print("inital accuracy:", inital_accuracy)

#load the model
model = load_model_from_file(tflite_model)


modified_file = 'person_detect_uint8_modified.tflite'
num_layers = 9 #starts with zero

params = []
#code to identify large layers
for num,buffer in enumerate(model.buffers):
      if buffer.data is not None:
          params.append([num,len(buffer.data)])
params.sort(reverse=True,key=lambda tup: tup[1])
#change#
#curr_layer = 1
#top_model_file = 'inception_top_acc_model_'+str(curr_layer)+'_.tflite'
#top_acc = inital_accuracy
start_layer = 8

for curr_layer in range(start_layer,num_layers): ########## change range from 0,num_layers
    print("************* Processing layer with parameters: **************", curr_layer, params[curr_layer])

    #reload the model to avoid missing data in buffer
    max_bins = 32 ##more than 2 layers
    top_acc_bins = []

    #load the latest model for layer > 0
    if curr_layer >start_layer:
        max_bins = 32
        tflite_model = top_model_file
        curr_acc = top_acc - 1 ####
    else:
        curr_acc = inital_accuracy - 2 ###
    top_acc = curr_acc
    model = load_model_from_file(tflite_model)
    top_model_file = 'person_detect_uint8_top_acc_model_'+str(curr_layer)+'_.tflite'


    #load the inital layer parameters of the FC layer
    for num,buffer in enumerate(model.buffers):
      if buffer.data is not None and num == params[curr_layer][0]:
        original_weights = np.frombuffer(buffer.data, dtype=np.uint8)
        v2 = np.add(original_weights,0)
        v2_min = v2.min()
        v2_max = v2.max()
        break

    #if curr_layer == start_layer:
    RangeValues = [1,50,100,110,125,150,165,175,200,255]
    #else:
    #    RangeValues = [v2_min, np.mean(v2) - np.std(v2), np.mean(v2), np.mean(v2) + np.std(v2), v2_max]
    total_bins = len(RangeValues)

    #while ((curr_acc <= inital_accuracy-0.2) and total_bins <= max_bins):
    #while ((curr_acc <= inital_accuracy and total_bins <= max_bins+1)):
    curr_iter = 1
    while(total_bins <= max_bins +1):
      model = load_model_from_file(tflite_model)
      for num,buffer in enumerate(model.buffers):
        if buffer.data is not None and num == params[curr_layer][0]:
          original_weights = np.frombuffer(buffer.data, dtype=np.uint8)
          v2 = np.add(original_weights,0)

          for x in range(len(RangeValues) - 1):
            indices = np.where(np.logical_and(v2>=RangeValues[x], v2<=RangeValues[x+1]))
            v2[indices] = np.uint8((RangeValues[x] + RangeValues[x+1])/2)
          break

      buffer.data = v2
      save_model_to_file(model, modified_file)
      curr_acc = test_model_accuracy(modified_file, model_name)

      print("Accuracy for range values:", curr_acc, RangeValues, np.unique(v2))
      
      if curr_acc > top_acc: #initial top_acc - 0.3
        print("saving top acc model, RangeValues", top_model_file, RangeValues)
        top_acc = curr_acc
        top_bins = RangeValues
        save_model_to_file(model, top_model_file)

      #avoid sensitivity analysis for the last bin
      if total_bins == max_bins+1:
          break

      #perturbation code to identify the sensitive bin
      if curr_iter == 1:
          bin_acc = []
          for times in range(len(RangeValues) - 1):
            model = load_model_from_file(tflite_model)
            for num,buffer in enumerate(model.buffers):
              if buffer.data is not None and num == params[curr_layer][0]:
                original_weights = np.frombuffer(buffer.data, dtype=np.uint8)
                v2 = np.add(original_weights,0)      
                indices = np.where(np.logical_and(v2>=RangeValues[times], v2<=RangeValues[times+1]))
                break
            v2[indices] = np.uint8(v2[indices] + 0.2*v2[indices])
            buffer.data = v2
            save_model_to_file(model, modified_file)
            bin_acc.append(test_model_accuracy(modified_file, model_name))

      else:
          bin_acc.pop(to_modify) #remove the sensitivity accuracy of pre-divided bin
          for times in range(2):
              model = load_model_from_file(tflite_model)
              for num,buffer in enumerate(model.buffers):
                if buffer.data is not None and num == params[curr_layer][0]:
                    original_weights = np.frombuffer(buffer.data, dtype=np.uint8)
                    v2 = np.add(original_weights,0)
                    indices = np.where(np.logical_and(v2>=RangeValues[to_modify+times], v2<=RangeValues[to_modify+times+1]))
                    break
              v2[indices] = np.uint8(v2[indices] + 0.2*v2[indices])
              buffer.data = v2
              save_model_to_file(model, modified_file)
              bin_acc.insert(to_modify+times,test_model_accuracy(modified_file, model_name))
      curr_iter +=1 
          
      print("The most sensitive bin here is:", bin_acc, np.array(bin_acc).argmin())
      to_modify = np.array(bin_acc).argmin()
      middle_bin = (RangeValues[to_modify] + RangeValues[to_modify+1]) / 2
      RangeValues.insert(to_modify+1, middle_bin)
      print("New Range Values:", RangeValues, len(RangeValues))
      total_bins = len(RangeValues)
          

    print("Reported top_accuracy and infered test accuracy :", top_acc, test_model_accuracy(top_model_file, model_name))
    print("The RangeValues are:", top_bins)
    print("total_bins", len(top_bins) - 1)
    print("The total parameters in layer modified", params[curr_layer])
    print("The model is saved in", top_model_file)
    curr_layer +=1
  


