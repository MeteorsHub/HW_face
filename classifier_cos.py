from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import sys

import numpy as np
import tensorflow as tf

import facenet


def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            train_dataset = facenet.get_dataset(args.train_data_dir)
            test_dataset = facenet.get_dataset(args.test_data_dir)

            # Check that there are at least one training image per class
            for cls in train_dataset:
                assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')
            for cls in test_dataset:
                assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

            paths, labels = facenet.get_image_paths_and_labels(train_dataset)

            print('Number of classes: %d' % len(train_dataset))
            print('Number of images: %d' % len(paths))

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * args.batch_size
                end_index = min((i + 1) * args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('train_data_dir', type=str,
                        help='Path to the data directory containing aligned train face patches.')
    parser.add_argument('test_data_dir', type=str,
                        help='Path to the data directory containing aligned test face patches.')
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
                        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
                        help='Use this number of images from each class for training and the rest for testing',
                        default=10)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
