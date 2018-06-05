from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import math
import sys

import numpy as np
import tensorflow as tf

import facenet


def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            dataset = facenet.get_dataset(args.data_dir)
            for cls in dataset:
                assert (len(cls.image_paths) == 1, 'There must be 1 sample for each class')

            paths, labels = facenet.get_image_paths_and_labels(dataset)

            print('Number of classes: %d' % len(dataset))
            print('Number of train images: %d' % len(paths))

            # Load the model
            print('Loading pretrain feature extraction model')
            saver = facenet.load_model(args.src_model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            first_embeddings, second_embeddings = tf.split(embeddings, 2, axis=0)

            def cos_distance(eb1, eb2):
                upper = tf.reduce_sum(eb1 * eb2, axis=1)
                lower = tf.sqrt(tf.reduce_sum(tf.square(eb1), 1) * tf.reduce_sum(tf.square(eb2), 1))
                return tf.acos(upper / lower) / math.pi

            cos_dist = cos_distance(first_embeddings, second_embeddings)

            loss = 1.0001 - cos_dist  # make the dist larger for images from different person
            loss = tf.reduce_sum(loss, 0)

            train_op = tf.train.GradientDescentOptimizer(args.learning_rate).minimize(loss, tf.train.get_global_step())

            n_images = len(paths)
            n_batches_per_compare = int(math.ceil((n_images - 1) / args.batch_size))

            print('begin training')
            for i in range(100):
                # for i in range(n_images):
                c_paths = copy.copy(paths)
                del c_paths[i]

                b_paths = [paths[i] for _ in range(args.batch_size)]
                first_images = facenet.load_data(b_paths, False, False, args.image_size)

                for j in range(10):
                    # for j in range(n_batches_per_compare):
                    start_index = i * args.batch_size
                    end_index = min((i + 1) * args.batch_size, n_images - 1)
                    paths_batch = c_paths[start_index:end_index]
                    second_images = facenet.load_data(paths_batch, False, False, args.image_size)

                    images = np.concatenate((first_images, second_images), 0)

                    feed_dict = {images_placeholder: images, phase_train_placeholder: True}
                    _, loss_val = sess.run([train_op, loss], feed_dict=feed_dict)
                print('loss on %d compare: %1.5f' % (i, loss_val))

            facenet.save_model(args.dst_model, sess, saver)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str,
                        help='Path to the data directory containing aligned train face patches.')
    parser.add_argument('src_model', type=str,
                        help='Pretrained model. Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('dst_model', type=str,
                        help='Saved model.')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--learning_rate', type=float,
                        help='finetune learning rate', default=0.001)
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
