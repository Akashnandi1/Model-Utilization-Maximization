import os
import json
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from utils import create_DomainNet_data
from augmentation import preprocess_image
import logging


def dataset_creator(cfg, test = False, train_test_split = 0.2):

    dataset_name = cfg['data']['dataset']
    src = cfg['data']['source']
    adp = cfg['data']['target']
    method = cfg['model']['method']

    # if cfg['model']['method'] == 'st':
    #     sub_data = cfg['data']['source']
    # elif cfg['model']['method'] == 'ta':
    #     sub_data = cfg['data']['target']
    # else:
    #     sub_data = cfg['data']['source']

    img_processor_train = preprocess_image(img_size = cfg['model']['img_size'], color_distort = True, test_crop = True, is_training = True)
    img_processor_test = preprocess_image(img_size = cfg['model']['img_size'], color_distort = True, test_crop = True, is_training = False)

    def process_img_train(image_path, label):
        image_string = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = img_processor_train(image)
        label_hot = tf.one_hot(label, total_classes)
        return image, label_hot

    def process_img_test(image_path, label):
        image_string = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = img_processor_test(image)
        label_hot = tf.one_hot(label, total_classes)
        return image, label_hot


    if dataset_name == 'DomainNet':
        src_train_df, src_test_df, adp_train_df, adp_test_df, total_classes = create_DomainNet_data(src, adp)

        if method == 'st':
            train_label = list(src_train_df['label'])
            train_img_path = list(src_train_df['path'])

            test_label = list(src_test_df['label'])
            test_img_path = list(src_test_df['path'])

            train_ds = tf.data.Dataset.from_tensor_slices((train_img_path, train_label))
            test_ds = tf.data.Dataset.from_tensor_slices((test_img_path, test_label))

        elif method == 'ta':
            train_label = list(adp_train_df['label'])
            train_img_path = list(adp_train_df['path'])

            test_label = list(adp_test_df['label'])
            test_img_path = list(adp_test_df['path'])

            train_ds = tf.data.Dataset.from_tensor_slices(train_img_path)
            test_ds = tf.data.Dataset.from_tensor_slices((test_img_path, test_label))
        else:
            raise ValueError

        train_total_samples = len(train_img_path)

    elif dataset_name == 'SVHN':
        (train_ds, test_ds), ds_info = tfds.load('svhn_cropped',
                                                 split=['train', 'test'],
                                                 shuffle_files=True,
                                                 as_supervised=True,
                                                 with_info=True)
        train_total_samples = len(train_ds)
        total_classes = 10


    elif dataset_name == 'MNIST':
        (train_ds, test_ds), ds_info = tfds.load('mnist',
                                                 split=['train', 'test'],
                                                 shuffle_files=True,
                                                 as_supervised=True,
                                                 with_info=True)

        train_total_samples = len(train_ds)
        total_classes = 10
    else:
        logging.info('incorrect dataset name')
        raise ValueError

    # shuffle size reduced to 2048 from train_total_samples to reduce time delay
    train_ds = (train_ds
                .map(process_img_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                .shuffle(2048)
                .batch(cfg['model']['batch_size'], drop_remainder=True)
                .prefetch(tf.data.experimental.AUTOTUNE))

    test_ds = (test_ds
               .map(process_img_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
               .batch(cfg['model']['batch_size'], drop_remainder=True)
               .prefetch(tf.data.experimental.AUTOTUNE))


    return train_ds, test_ds, total_classes