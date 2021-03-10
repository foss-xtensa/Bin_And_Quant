# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow.compat.v1 as tf
import tf_slim as slim
import numpy as np

from tensorflow.contrib import quantize as contrib_quantize

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_bool(
    'quantize', False, 'whether to use quantized graph or not.')

tf.app.flags.DEFINE_bool('use_grayscale', False,
                         'Whether to convert input images to grayscale.')

FLAGS = tf.app.flags.FLAGS


def main(_):

  from tensorflow.compat.v1 import ConfigProto
  from tensorflow.compat.v1 import InteractiveSession
  config = ConfigProto()
  config.gpu_options.allow_growth = True
  session = InteractiveSession(config=config)

  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size, num_epochs=1)
    [image, label] = provider.get(['image', 'label'])
    label -= FLAGS.labels_offset
    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False,
        use_grayscale=FLAGS.use_grayscale)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    #image = tf.train.limit_epochs(image,num_epochs=1)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=1 * FLAGS.batch_size)
    ########################### load data and convert to numpy array

    #import pdb
    #pdb.set_trace()
    total = 0
    correct_predictions = 0
    total_batches = int(50000/FLAGS.batch_size)
    quantization=True
    #label_checker = {}

    #interpreter = tf.lite.Interpreter(model_path="/home/ms75986/Desktop/Cadence/bin_quant/MobileNet/models/research/slim/quant8.tflite")
    #interpreter = tf.lite.Interpreter(model_path="/home/ms75986/Desktop/Cadence/bin_quant/MobileNet/models/research/slim/mobilenet_v2_modified.tflite")
    interpreter = tf.lite.Interpreter(model_path="/home/ms75986/Desktop/Cadence/bin_quant/slim/inception_models/inception_v1_224_quant.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with tf.Session() as sess:
        sess.run([tf.local_variables_initializer(),tf.global_variables_initializer(),])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #tf.train.start_queue_runners()
        try:
            while not coord.should_stop():
        #with slim.queues.QueueRunners(sess):
            #for i in range(total_batches):
            #while not coord.should_stop():
                np_image, np_label = sess.run([images, labels])
                #print(len(np_image))
                #height, width, _ = np_image[0].shape
                #print(np_image[0].shape, np_label)


                #'''

                #logits_ = np.zeros((2,1001))
                #interpreter = tf.lite.Interpreter(model_path="/home/ms75986/Desktop/Cadence/bin_quant/MobileNet/models/research/slim/mobilenet_v2_1.0_224.tflite")
                #interpreter.allocate_tensors()

                #input_details = interpreter.get_input_details()
                #output_details = interpreter.get_output_details()

                num=0
                while(num < FLAGS.batch_size):
                    #if np_label[num] in label_checker:
                    #    label_checker[np_label[num]] += 1
                    #else:
                    #    label_checker[np_label[num]] = 1

                    #input_data = np.expand_dims(np_image[num], axis=1).astype(np.float32)

                    if quantization:
                        #input_details = interpreter.get_input_details()
                        #output_details = interpreter.get_output_details()
                        input_shape = input_details[0]['shape']
                        input_scale, input_zero_point = input_details[0]["quantization"]
                        #print(input_shape, input_scale, input_zero_point)
                        input_data = np.array(np_image[num], dtype=np.float32) /input_scale + input_zero_point
                        #input_data = input_data /input_scale + input_zero_point
                        input_data = input_data.astype(input_details[0]["dtype"])
                        #input_data = input_data.reshape(1,224,224,3)
                    else:
                        input_data = np.array(np_image[num], dtype=np.float32)

                    interpreter.set_tensor(input_details[0]['index'], input_data.reshape(1,224,224,3))
                    interpreter.invoke()
                    logits = interpreter.get_tensor(output_details[0]['index'])
                    #logits_[num,:] = logits
                    top_prediction = logits.argmax()
                    correct_predictions += (top_prediction == np_label[num])
                    #print(top_prediction)
                    num+=1
                #'''
    
                #logits = tf.convert_to_tensor(logits_, dtype=tf.float32, name='MobilenetV2/Logits/output') 
                total = total+ FLAGS.batch_size

                if total % 1000 == 0:
                    print("finished processing 1000 samples",total)
                    #break
        except:
            coord.request_stop()
            coord.join(threads)

    print("Accuracy", (correct_predictions*100) / total)
    #import pdb
    #pdb.set_trace()
    #print(label_checker)
                    
    exit(0)
    #'''
    #logits, _ = network_fn(images)

    if FLAGS.quantize:
      contrib_quantize.create_eval_graph()

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'Recall_5': slim.metrics.streaming_recall_at_k(
            logits, labels, 5),
    })

    # Print the summaries to screen.
    for name, value in names_to_values.items():
      summary_name = 'eval/%s' % name
      op = tf.summary.scalar(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    slim.evaluation.evaluate_once(
        master=FLAGS.master,
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=list(names_to_updates.values()),
        variables_to_restore=variables_to_restore)


if __name__ == '__main__':
  tf.app.run()
