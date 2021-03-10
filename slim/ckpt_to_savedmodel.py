import os
import tensorflow.compat.v1 as tf

trained_checkpoint_prefix = 'cifar10-mobilenet/model.ckpt-529360'
export_dir = os.path.join('export_dir', '0')

graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    # Restore from checkpoint
    loader = tf.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
    loader.restore(sess, trained_checkpoint_prefix)

    # Export checkpoint to SavedModel
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.tag_constants.TRAINING, tf.saved_model.tag_constants.SERVING],
                                         strip_default_attrs=True)
    builder.save()
