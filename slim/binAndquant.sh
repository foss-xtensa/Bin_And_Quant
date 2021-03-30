#Bin and Quant based on sensitivity analysis 
#1. Load the inital model and return the original accuracy
#2. Set the required accuracy to 1% deviation of the original accuracy
#3. Variables: 1. Rangevalues as a list, current accuracy, original accuracy
#4. Two python codes: 1. Testing 2. Automation 


#load a model
tflite_model='/home/ms75986/Desktop/Cadence/bin_quant/Bin_And_Quant/slim/inception_models/inception_v1_224_quant.tflite'
model_name='inception_v1'

#Run inference and save the returned accuracy
acc=$(python3 tflite_eval.py     --alsologtostderr  --dataset_dir=../../MobileNet/models/research/imagenet-data     --dataset_name=imagenet     --dataset_split_name=validation     --model_name="$model_name" --batch_size=10 --tflite_file="$tflite_model" 2>&1 >/dev/null)

echo "printed accuracy"
echo $acc
echo ${acc:(-5)}

