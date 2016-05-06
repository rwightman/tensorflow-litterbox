from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tarfile
import glob

import numpy as np
from six.moves import urllib
import tensorflow as tf
import pandas as pd

FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', './',
    """Path to output_graph.pb and output_labels.txt, """)

tf.app.flags.DEFINE_string(
    'test_dir', '/data/sfarm/test',
    """Path to test images, """)

tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")

tf.app.flags.DEFINE_integer('num_top_predictions', 10,
                            """Display this many predictions.""")

LABEL_FILE = 'output_labels.txt'

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'output_graph.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference(image_list):
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing
  """

  outputs = []

  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    #count = 0

    for filename in image_list:

      if not tf.gfile.Exists(filename):
        tf.logging.fatal('File does not exist %s', filename)
      
      image_data = tf.gfile.FastGFile(filename, 'rb').read()

      prediction = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
      prediction = np.squeeze(prediction)

      del image_data

      # Creates node ID --> English string lookup.
      #node_lookup = NodeLookup()

      #top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
      #print(predictions)
      image_name = os.path.basename(filename)
      output = [image_name] + prediction.tolist()
      outputs.append(output)
      #for node_id, score in enumerate(prediction):
      #  print('%s (score = %.5f)' % (node_id, score))
      #count = count + 1
      #if count > 200:
      #  break

    return outputs


def main(_):

  #image = (FLAGS.image_file if FLAGS.image_file else
  #         os.path.join(FLAGS.model_dir, 'cropped_panda.jpg'))

  #extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
  labels = []
  with open(os.path.join(FLAGS.model_dir, LABEL_FILE), 'r') as f:
    labels = f.read().splitlines()
  print(labels)

  file_list = []
  file_glob = os.path.join(FLAGS.test_dir, '*.jpg')
  file_list.extend(glob.glob(file_glob))
  outputs = run_inference(file_list)

  columns = ["Img"] + labels
  df = pd.DataFrame(outputs, columns = columns)

  sorted_columns = ["Img"] + sorted(labels)
  sorted_df = df[sorted_columns]
  
  sorted_df.to_csv(os.path.join(FLAGS.model_dir,'output.csv'), index=False)

if __name__ == '__main__':
  tf.app.run()
