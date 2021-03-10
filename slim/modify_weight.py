import tensorflow as tf
from nets.mobilenet import mobilenet_v2
import numpy as np


'''
tf.disable_eager_execution()
tf.reset_default_graph()

file_input = tf.placeholder(tf.string, ())

image = tf.image.decode_jpeg(tf.read_file(file_input))

images = tf.expand_dims(image, 0)
images = tf.cast(images, tf.float32) / 128.  - 1
images.set_shape((None, None, None, 3))
images = tf.image.resize_images(images, (224, 224))


with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=True)):
  logits, endpoints = mobilenet_v2.mobilenet(images)


ema = tf.train.ExponentialMovingAverage(0.999)
vars_ = ema.variables_to_restore()


saver = tf.train.Saver(vars_)


#'''
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#sess = tf.InteractiveSession()


with tf.Session() as sess:
  saver = tf.train.import_meta_graph('/home/ms75986/Desktop/Cadence/bin_quant/MobileNet/models/research/slim/v2_224_100/model.ckpt-2685865.meta', clear_devices=True)
  saver.restore(sess,  '/home/ms75986/Desktop/Cadence/bin_quant/MobileNet/models/research/slim/v2_224_100/model.ckpt-2685865')
  # for variable in tf.trainable_variables():
  #   print(variable)
  v1 = sess.graph.get_tensor_by_name('MobilenetV2/Logits/Conv2d_1c_1x1/weights:0')
  v2_r = sess.run(v1)
  print(v2_r.max(), v2_r.min(), v2_r.shape)
  v2_d1 = np.shape(v2_r)[2]
  v2_d2 = np.shape(v2_r)[3]
  v2 = np.reshape(v2_r,(v2_d1*v2_d2,1))

  RangeValues = [np.min(v2),-0.15,-0.1,0.0]
  for x in range(len(RangeValues)*2 -2):
        if x <len(RangeValues)-1:
            indices = np.where(np.logical_and(v2>=RangeValues[x], v2<=RangeValues[x+1]))
            v2[indices] = (RangeValues[x] + RangeValues[x+1])/2
        else:
            x = x+1
            RangeValues[0] = - np.max(v2)
            indices = np.where(np.logical_and(v2<=-RangeValues[x-len(RangeValues)], v2>=-RangeValues[x-len(RangeValues)+1]))
            v2[indices] = (-RangeValues[x-len(RangeValues)] - RangeValues[x-len(RangeValues)+1])/2
  v2_new = v2.reshape(1,1,v2_d1,v2_d2)
  print("modified final_fc_weights:", v2_new)
  sess.run(tf.assign(v1,v2_new)) 

  saver.save(sess, '/home/ms75986/Desktop/Cadence/bin_quant/MobileNet/models/research/slim/v2_224_100/model.ckpt-2685865-modified')


