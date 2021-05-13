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
import huffman
import string
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


def compress(model_name, tflite_name, curr_layer):

    model = load_model_from_file(tflite_name)
    for num,buffer in enumerate(model.buffers):
      if buffer.data is not None and num == curr_layer:
        original_weights = np.frombuffer(buffer.data, dtype=np.uint8)
        v2 = np.add(original_weights,0)
        unique_values = np.unique(v2)
        np.save('test.npy', v2)
        mapping={}
        #create unique keys in uint8
        count=len(unique_values)+1 #remove +1
        start=0
        curr_count=1
        keys = []
        alphabet_lower = list(string.ascii_lowercase)
        alphabet_upper = list(string.ascii_uppercase)
        alphabet_lower.extend(alphabet_upper)

        keys = alphabet_lower[:count]

        '''
        while curr_count<count:
            if start not in unique_values:
                keys.append(start)
                start+=1
                curr_count+=1
            else:
                start+=1
        '''

        for num,val in enumerate(unique_values):
            mapping[unique_values[num]] = keys[num]
        #mapping={26:'a'}

        curr_dir='/home/ms75986/Desktop/Cadence/bin_quant/Bin_And_Quant/slim/' + model_name + '_huffman_compressed/'

        if not os.path.exists(curr_dir):
            os.mkdir(curr_dir)
        mapping_name = curr_dir + 'mapping_' + str(curr_layer)+'.dictionary'

        with open(mapping_name, 'wb') as mapping_dictionary_file:
            pickle.dump(mapping, mapping_dictionary_file)

        v2_str = ""
        for ele in v2:
            v2_str+=mapping[ele]

        path = curr_dir+str(curr_layer)+'_huffman_out.txt'  #add the original size to the file name
        #np.save(path, v2)
        with open(path, 'w') as f:
            f.write(v2_str)
            #f.write(v2.tostring())

        h=huffman.HuffmanCoding(path)
        output_path = h.compress()

        config_name=curr_dir+'config_'+ str(curr_layer)+'.dictionary'

        with open(config_name, 'wb') as config_dictionary_file:
            pickle.dump(h, config_dictionary_file)

        buffer.data = np.array([0], dtype=np.uint8) #original_weights[:1]

        os.remove(path)
    compressed_name = curr_dir + model_name + '_compressed.tflite'
    save_model_to_file(model, compressed_name)
    return compressed_name


tflite_model= 'person_detect_models/person_detect_uint8_top_acc_model_6_.tflite'
#'inception_v2__top_acc_model_8_.tflite' #'inception_v1_result_models/inception_v1_top_acc_model_5_.tflite' #'model2_mobilenet_v2_top_acc_model_5_.tflite' #'v1_mob_layer1-2_16bins_64_42acc.tflite'
model_name='person_mobilenet_v1'
layers_to_compress=[0,1,2,4,6]

model = load_model_from_file(tflite_model)
params = []
#code to identify large layers
for num,buffer in enumerate(model.buffers):
      if buffer.data is not None:
          params.append([num,len(buffer.data)])
params.sort(reverse=True,key=lambda tup: tup[1])
print(params)
layer_ids=[]
for ele in range(len(layers_to_compress)):
    layer_ids.append(params[layers_to_compress[ele]][0])


for x in range(len(layer_ids)):
    tflite_model = compress(model_name, tflite_model, layer_ids[x])

