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

            train_paths, train_labels = facenet.get_image_paths_and_labels(train_dataset)
            test_paths, test_labels = facenet.get_image_paths_and_labels(test_dataset)

            print('Number of classes: %d' % len(train_dataset))
            print('Number of train images: %d' % len(train_paths))
            print('Number of test images: %d' % len(test_paths))

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for train images')
            nrof_images = len(train_paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * args.batch_size
                end_index = min((i + 1) * args.batch_size, nrof_images)
                paths_batch = train_paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
            train_emb_array = np.copy(emb_array)

            print('Calculating features for test images')
            nrof_images = len(test_paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * args.batch_size
                end_index = min((i + 1) * args.batch_size, nrof_images)
                paths_batch = test_paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
            test_emb_array = emb_array

            # subtract mean
            mean = np.mean(train_emb_array, axis=0)
            # mean = np.mean(np.concatenate((train_emb_array, test_emb_array), 0), axis=0)
            train_emb_array -= mean
            test_emb_array -= mean

            print('Calculating cos distance')
            class_names = [cls.name.replace('_', ' ') for cls in train_dataset]
            dist_dict = np.ones((len(test_emb_array), len(train_emb_array)), np.float32) * np.inf
            for i in range(len(test_emb_array)):
                for j in range(len(train_emb_array)):
                    dist_dict[i, j] = facenet.distance(test_emb_array[i:i + 1], train_emb_array[j:j + 1], 1)[0]

            predictions = np.zeros((len(test_emb_array), 5), np.int32)

            top5_in_num = 0
            with open('predictions_cos.txt', 'w') as f:
                for i in range(len(predictions)):
                    predictions[i] = np.argsort(dist_dict[i])[:5]
                    outstr = '%s' % class_names[test_labels[i]]
                    for j in range(5):
                        outstr += ' [%s, dist: %.5f]' % (
                        class_names[train_labels[predictions[i, j]]], dist_dict[i, predictions[i, j]])
                    print(outstr)
                    f.write(outstr + '\r\n')
                    if np.isin(test_labels[i], predictions[i]):
                        top5_in_num += 1
                accuracy = np.mean(np.equal(np.array(train_labels)[predictions[:, 0]], np.array(test_labels)))
                print('Top1 accuracy: %.6f' % accuracy)
                print('Top5 accuracy: %.6f' % (top5_in_num / len(predictions)))
                f.write('Top1 accuracy: %.6f \r\n' % accuracy)
                f.write('Top5 accuracy: %.6f \r\n' % (top5_in_num / len(predictions)))


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
