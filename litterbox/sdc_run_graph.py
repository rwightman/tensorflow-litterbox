# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
import numpy as np
import threading
import queue
import time
import os
import cv2
import argparse
from datetime import datetime


def get_image_files(folder, types=('.jpg', '.jpeg', '.png')):
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False):
        filenames += [os.path.join(root, f) for f in files if os.path.splitext(f)[1].lower() in types]
    filenames = list(sorted(filenames))
    return filenames


class RwightmanModel(object):

    def __init__(self, alpha=0.9, graph_path='', checkpoint_path='', metagraph_path=''):
        if graph_path:
            assert os.path.isfile(graph_path)
        else:
            assert os.path.isfile(checkpoint_path) and os.path.isfile(metagraph_path)
        self.graph = tf.Graph()
        with self.graph.as_default():
            if graph_path:
                # load a graph with weights frozen as constants
                graph_def = tf.GraphDef()
                with open(graph_path, "rb") as f:
                    graph_def.ParseFromString(f.read())
                    _ = tf.import_graph_def(graph_def, name="")
                self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            else:
                # load a meta-graph and initialize variables form checkpoint
                saver = tf.train.import_meta_graph(metagraph_path)
                self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
                saver.restore(self.session, checkpoint_path)
        self.model_input = self.session.graph.get_tensor_by_name("input_placeholder:0")
        self.model_output = self.session.graph.get_tensor_by_name("output_steer:0")
        self.last_steering_angle = 0  # None
        self.alpha = alpha

    def predict(self, image):
        feed_dict = {self.model_input: image}
        steering_angle = self.session.run(self.model_output, feed_dict=feed_dict)
        if self.last_steering_angle is None:
            self.last_steering_angle = steering_angle
        steering_angle = self.alpha * steering_angle + (1 - self.alpha) * self.last_steering_angle
        self.last_steering_angle = steering_angle
        return steering_angle


class ProcessorThread(threading.Thread):

    def __init__(self, name, q, model):
        super(ProcessorThread, self).__init__(name=name)
        self.q = q
        self.model = model
        self.outputs = []

    def run(self):
        print('Entering processing loop...')
        while True:
            item = self.q.get()
            if item is None:
                print("Exiting processing loop...")
                break
            output = self.model.predict(item)
            self.outputs.append(output)
            self.q.task_done()
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.5, help='Path to the metagraph path')
    parser.add_argument('--graph_path', type=str, help='Path to the metagraph path')
    parser.add_argument('--metagraph_path', type=str, help='Path to the metagraph path')
    parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint path')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the images')
    parser.add_argument('--target_csv', type=str, help='Path to target csv for optional RMSE calc.')
    args = parser.parse_args()
    print(args.alpha)
    print('%s: Initializing model.' % datetime.now())
    model = RwightmanModel(
        alpha=args.alpha,
        graph_path=args.graph_path,
        metagraph_path=args.metagraph_path,
        checkpoint_path=args.checkpoint_path)
    # Push one empty image through to ensure Tensorflow is ready, random wait on first frame otherwise
    model.predict(np.zeros(shape=[480, 640, 3]))

    q = queue.Queue(20)
    processor = ProcessorThread('process', q, model)
    processor.start()
    image_files = get_image_files(args.data_dir)

    print('%s: starting execution on (%s).' % (datetime.now(), args.data_dir))
    start_time = time.time()
    timestamps = []
    for f in image_files:
        # Note, all cv2 based decode and split/merge color channel switch resulted in faster throughput, lower
        # CPU usage than PIL Image or cv2 + python array slice reversal.
        image = cv2.imread(f)
        b, g, r = cv2.split(image)  # get BGR channels
        image = cv2.merge([r, g, b])  # merge as RGB
        q.put(image)
        timestamps.append(os.path.splitext(os.path.basename(f))[0])

    q.put(None)
    processor.join()

    duration = time.time() - start_time
    images_per_sec = len(image_files) / duration
    print('%s: %d images processed in %s seconds, %.1f images/sec'
          % (datetime.now(), len(image_files), duration, images_per_sec))

    columns_ang = ['frame_id', 'steering_angle']
    df_ang = pd.DataFrame(data={columns_ang[0]: timestamps, columns_ang[1]: processor.outputs}, columns=columns_ang)
    df_ang.to_csv('./output_angle.csv', index=False)

    if args.target_csv:
        targets_df = pd.read_csv(args.target_csv, header=0, index_col=False)
        targets = np.squeeze(targets_df.as_matrix(columns=[columns_ang[1]]))
        predictions = np.asarray(processor.outputs)
        mse = ((predictions - targets) ** 2).mean()
        rmse = np.sqrt(mse)
        print("RMSE: %f, MSE: %f" % (rmse, mse))

if __name__ == '__main__':
    main()
