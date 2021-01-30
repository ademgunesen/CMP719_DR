import tensorflow as tf
import utils_cmp719
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorboard.plugins.hparams import api as hp
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
import time
import os
import pprint 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from visualizer import plot_history, plot_confusion_matrix, plot_loss_and_acc
import utils as util
from utils import make_log_dir, write_to_log, save_var, save_model, send_as_mail
from configurators import train_config, generator_config, dataset_config
from train import train_model, evaluate_model


from tensorflow.keras import layers

comp_info = util.get_computer_info()

HP_LR = hp.HParam('learning_rate', hp.Discrete([1e-3,5e-4,1e-4]))
METRIC_ACCURACY = 'accuracy'

def train_test_model(hparams) : 
    t_conf = train_config(name = "Esma",
                        BATCH_SIZE = 12, 
                        EPOCHS=200, 
                        LEARNING_RATE = hparams[HP_LR])

    g_conf = generator_config(name = "default")
    d_conf = dataset_config(name = "default")

    log_dir = make_log_dir("out/")
    t_conf.save(save_dir = log_dir)
    g_conf.save(save_dir = log_dir)
    d_conf.save(save_dir = log_dir)

    with tf.summary.create_file_writer('logs/fit10').as_default():
        hp.hparams_config(
        hparams=[HP_LR],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
        )
    
    optimizer=tf.keras.optimizers.SGD(lr=t_conf.LEARNING_RATE)
    
    model = utils_cmp719.choose_nets('seresnet18', 5)
    model.build(input_shape=(t_conf.BATCH_SIZE ,t_conf.IMG_HEIGHT, t_conf.IMG_WIDTH, 3))
    model, history = train_model(model,t_conf,g_conf,d_conf,optimizer)
    accuracy = evaluate_model(model,t_conf,g_conf,d_conf)

    #Visualize history
    plot_loss_and_acc(history, save=True, saveDir=log_dir, fname='train_')
    return accuracy

def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

session_num = 0

for learning_rate in HP_LR.domain.values:
    hparams = {HP_LR: learning_rate}
    run_name = "run-%d" % session_num
    print('--- Starting trial: %s' % run_name)
    print({h.name: hparams[h] for h in hparams})
    run('logs/fit10/' + run_name, hparams)
    session_num += 1