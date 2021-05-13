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
import pickle
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


def decompress(curr_dir, model_name, tflite_name, curr_layer):
    model = load_model_from_file(tflite_name)
    for num,buffer in enumerate(model.buffers):
      if buffer.data is not None and num == curr_layer:#params[curr_layer][0]:

        config_name=curr_dir+'config_'+ str(curr_layer)+'.dictionary'

        with open(config_name, 'rb') as config_dictionary_file_load:
            h_load = pickle.load(config_dictionary_file_load)

        bin_path = curr_dir+str(curr_layer)+'_huffman_out.bin'
        h_load.decompress(bin_path)

        
        file_path = curr_dir+str(curr_layer)+'_huffman_out_decompressed.txt'
        my_file = open(file_path, "r")
        content = my_file.read()
        arr_text = list(content)
        #arr = np.array(arr, dtype=np.uint8)
        #os.remove(file_path)

        mapping_name = curr_dir + 'mapping_' + str(curr_layer)+'.dictionary'
        with open(mapping_name, 'rb') as mapping_dictionary_file_load:
            mapping = pickle.load(mapping_dictionary_file_load)

        inv_map = {v: k for k, v in mapping.items()}

        #keys = mapping.keys()
        arr = np.zeros((len(arr_text),), dtype=np.uint8)
        for num,ele in enumerate(arr):
            arr[num] = inv_map[arr_text[num]]

        #for ele in keys:
        #    arr[arr==ele] = mapping[ele]
        buffer.data = arr
        break
    
    decompressed_name = model_name + '_decompressed'+ '.tflite'
    save_model_to_file(model, decompressed_name)
    return decompressed_name

model_name='person_mobilenet_v1'
curr_dir='/home/ms75986/Desktop/Cadence/bin_quant/Bin_And_Quant/slim/'+ model_name + '_huffman_compressed/'
tflite_model=curr_dir + model_name+ '_compressed.tflite'

layers_to_compress=[4,8,11,18,86]

model = load_model_from_file(tflite_model)
params = []
#code to identify large layers
for num,buffer in enumerate(model.buffers):
      if buffer.data is not None:
          params.append([num,len(buffer.data)])
params.sort(reverse=True,key=lambda tup: tup[1])

layer_ids=[4,8,11,18,86]
#for ele in layers_to_compress:
#    layer_ids.append(params[layers_to_compress[ele]][0])

for x in range(len(layer_ids)):
    tflite_model = decompress(curr_dir, model_name, tflite_model, layer_ids[x])

