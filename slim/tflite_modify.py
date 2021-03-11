import flatbuffers
import matplotlib.pyplot as plt
import numpy as np
import pprint
import re
import sys

sys.path.append("/home/ms75986/Desktop/Cadence/bin_quant/Bin_And_Quant/slim/tflite/")
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


model = load_model_from_file('/home/ms75986/Desktop/Cadence/bin_quant/Bin_And_Quant/slim/inception_models/inception_v1_224_quant.tflite')
layer_unique = []
weights_modified = []

for num, buffer in enumerate(model.buffers):
  if buffer.data is not None and len(buffer.data) >= 1025000 and len(buffer.data) <= 1045024:
    original_weights = np.frombuffer(buffer.data, dtype=np.uint8)
    print(num, len(buffer.data),original_weights )
    #layer_shapes.append(original_weights.shape)
    

    # This is the line where the weights are altered.
    # Try replacing it with your own version, for example:
    munged_weights = np.add(original_weights, 0)
    v2 = munged_weights
    #v2 = np.add(v2, 10)


    #'''
    v2_min = v2.min()
    v2_max = v2.max()
    RangeValues = [v2_min, (v2_min+ np.mean(v2) - np.std(v2))/2,  np.mean(v2) - np.std(v2), 
                   (np.mean(v2) - np.std(v2) + np.mean(v2))/2 ,np.mean(v2), (np.mean(v2)+ np.mean(v2) + np.std(v2))/2,\
                   np.mean(v2) + np.std(v2), (np.mean(v2) + np.std(v2) + v2_max)/2,\
                   v2_max]
    '''
    RangeValues = [v2_min, (RangeValues[0] + RangeValues[1]) /2, (v2_min+ np.mean(v2) - np.std(v2))/2, (RangeValues[1] + RangeValues[2]) /2, np.mean(v2) - np.std(v2), (RangeValues[2] + RangeValues[3]) /2, 
                   (np.mean(v2) - np.std(v2) + np.mean(v2))/2 , (RangeValues[3] + RangeValues[4]) /2, np.mean(v2), \
                   (RangeValues[4] + RangeValues[5]) /2, (np.mean(v2)+ np.mean(v2) + np.std(v2))/2,\
                   (RangeValues[5] + RangeValues[6]) /2, np.mean(v2) + np.std(v2), (RangeValues[6] + RangeValues[7]) /2,\
                   (np.mean(v2) + np.std(v2) + v2_max)/2,(RangeValues[7] + RangeValues[8]) /2,  v2_max]
    '''
    # RangeValues = [v2_min, (v2_min+ np.mean(v2) - np.std(v2))/2, np.mean(v2) - np.std(v2), (RangeValues[2] + RangeValues[3]) /2, 
    #                (np.mean(v2) - np.std(v2) + np.mean(v2))/2 , ((np.mean(v2) - np.std(v2) + np.mean(v2))/2 + (RangeValues[3] + RangeValues[4]) /2)/2, (RangeValues[3] + RangeValues[4]) /2, ((RangeValues[3] + RangeValues[4]) /2+  np.mean(v2))/2,  np.mean(v2), \
    #                ((RangeValues[4] + RangeValues[5]) /2 + np.mean(v2) )/2, (RangeValues[4] + RangeValues[5]) /2,((RangeValues[4] + RangeValues[5]) /2 +  (np.mean(v2)+ np.mean(v2) + np.std(v2))/2)/2 , (np.mean(v2)+ np.mean(v2) + np.std(v2))/2,\
    #                (RangeValues[5] + RangeValues[6]) /2, np.mean(v2) + np.std(v2),\
    #                (np.mean(v2) + np.std(v2) + v2_max)/2,  v2_max]

    #RangeValues = [1, 40.299270174403205, 79.59854034880641,80, 86.97340919723035,90, 94.3482780456543,96, 101.72314689407824,105, 109.09801574250218,120, 160, 182.0490078712511,200, 225, 255]


    print(RangeValues)
    for x in range(len(RangeValues) - 1):
            indices = np.where(np.logical_and(v2>=RangeValues[x], v2<=RangeValues[x+1]))
            v2[indices] = np.uint8((RangeValues[x] + RangeValues[x+1])/2)

    
    print(np.unique(v2), v2)
    layer_unique.append(np.unique(v2))
    weights_modified.append(v2)

    #'''
    buffer.data = v2

save_model_to_file(model, '/home/ms75986/Desktop/Cadence/bin_quant/Bin_And_Quant/slim/inception_models/inception_v1_224_quant_modified.tflite')
