# Bin_And_Quant

In the era of deploying complex and storage heavy deep learning model for various signal processing task, B&Q method focuses on compressing the deep learning model without compromising on the accuracy. 
We applied B&Q on 4 different deep learning models and achieved significant compression with almost no loss in accuracy. (ICASSP paper: https://ieeexplore.ieee.org/abstract/document/9053927) 

Bin & Quant: 
As the name suggests, the weights values of a particular layer are grouped into different bins and are quantized. This compressed weight values are then saved as either bit array or using huffman encoding as a second stage compression to save storage memory. 
	
In this repository, we provide the technique to automate the B&Q approach i.e., the program will choose the appropriate bin values and number of bins for almost no loss in accuracy. 

In this repository, the code to apply B&Q on MicroSpeech, MobilenetV1, MobilenetV2, InceptionV1, InceptionV2 & PersonDetect models are provided. All the above models were already quantized to uint8. Therefore, the B&Q applied on these models further reduces the number of bits required for storage from 8 bits to 3-6 bits. 

# 1. MicroSpeech
micro_speech_uint8.ipynb:
In the first release, we have automated the B&Q approach on the tensorflow mico_speech uint8 quantized model. This program operates directly on the TFLite file. The ipython notebook downloads the latest mico_speech model from tensorflow and applies B&Q to reduce the model size of the uint8 quantized micro_speech model. 

micro_speech_int8.ipynb:
This iPython notebook applies B&Q on the ckpt file downloaded from tensorflow and generates the B&Q'ed tflite file for inference. This is the latest int8 model automation tool where, the original ckpt file and the conversion process of ckpt to tflite file should be ready to use. 

The trained MicroSpeech models is provided in speech_micro_train_2020_05_10.tgz. 

# 2. Vision Models 
The slim/ folder in the main branch provides the code to apply B&Q on the vision models. 

The results of B&Q applied on uint8 models: 
