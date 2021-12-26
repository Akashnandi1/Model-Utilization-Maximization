import tensorflow as tf
from logger import set_logger
from os.path import join, splitext, basename
import os
from determinism import set_seeds, set_global_determinism
from base_model import select_model
from dataset import dataset_creator
from train import train
import logging

def exp_pipeline(cfg):

    #options = tf.data.Options()
    #options.deterministic = True
    #set_seeds()
    #set_global_determinism()

    log_file = join(cfg['exp']['log_folder'] + "/exp_log.log")

    set_logger(log_file)

    # Source Train
    if cfg['model']['method'] == 'st':
        # load data
        train_ds, test_ds, total_classes = dataset_creator(cfg)
        if cfg['exp']['load'] == None:
            model = select_model(cfg, total_classes)
        else:
            model = tf.keras.models.load_model(cfg['exp']['load'])

        train(cfg, model, train_ds, test_ds)

    # Weight Selection
    if cfg['model']['method'] == 'sw':
        if cfg['exp']['load'] != None:
            model = tf.keras.models.load_model(cfg['exp']['load'])
        else:
            logging.info('Model path not provided for weight selection procedure')
            raise ValueError('Model path not provided for weight selection procedure')

    # Target Adapt
    if cfg['model']['method'] == 'ta':
        # load data
        train_ds, test_ds, total_classes = dataset_creator(cfg)
        if cfg['exp']['load'] != None:
            model = tf.keras.models.load_model(cfg['exp']['load'])
            train(cfg, model, train_ds, test_ds)
        else:
            logging.info('Model path not provided for target adaptation procedure')
            raise ValueError('Model path not provided for target adaptation procedure')




