from configurators import train_config, generator_config, dataset_config
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, LearningRateScheduler
from tensorflow.keras.applications import DenseNet121,DenseNet169,DenseNet201
from efficientnet.tfkeras import EfficientNetB3, EfficientNetB0, EfficientNetB5

from visualizer import plot_history, plot_confusion_matrix, plot_loss_and_acc
from utils import make_log_dir, write_to_log, save_var, save_model, send_as_mail, load_model, make_subfolder
import utils as util
import pandas as pd
import time
import numpy as np
from configurators import train_config, generator_config, dataset_config

def train_model(model, train_conf,generator_conf,dataset_conf,optimizer, seed = 42):
    X_train, X_valid, train_dest_path, valid_dest_path = dataset_conf.get_dataset()
    log_dir = train_conf.log_dir
    train_datagen = ImageDataGenerator(
                        rotation_range      = generator_conf.rotation_range,
                        #width_shift_range  = 0.1,
                        #height_shift_range = 0.1,
                        shear_range         = generator_conf.shear_range,
                        zoom_range          = generator_conf.zoom_range,
                        #channel_shift_range= 20,
                        horizontal_flip     = generator_conf.horizontal_flip,
                        vertical_flip       = generator_conf.vertical_flip,
                        fill_mode           = generator_conf.fill_mode,
                        rescale             = generator_conf.rescale)
    test_datagen = ImageDataGenerator(rescale=generator_conf.rescale)

    train_generator = test_datagen.flow_from_dataframe(
            dataframe   =X_train,
            directory   =train_dest_path,
            x_col       ='id_code',
            y_col       ='diagnosis',
            target_size =(train_conf.IMG_HEIGHT, train_conf.IMG_WIDTH),
            class_mode  ='categorical',
            batch_size  =train_conf.BATCH_SIZE,
            seed        =seed,
            shuffle     =True)

    validation_generator = test_datagen.flow_from_dataframe(
            dataframe=X_valid,
            directory=valid_dest_path,
            x_col="id_code",
            y_col="diagnosis",
            target_size=(train_conf.IMG_HEIGHT, train_conf.IMG_WIDTH),
            class_mode='categorical',
            batch_size=train_conf.BATCH_SIZE,
            seed=seed,
            shuffle=False)

    inputs = Input(shape=(train_conf.IMG_HEIGHT, train_conf.IMG_WIDTH, 3))

    model.summary()
    weights_dir = make_subfolder("Weights", log_dir)
    my_filepath = weights_dir + 'weights-improvement-{epoch:02d}.hdf5'
    cp = tf.keras.callbacks.ModelCheckpoint(
        filepath=my_filepath,
        save_freq='epoch'
    )
    es = EarlyStopping(monitor='val_loss', patience = train_conf.ES_PATIENCE)

    model.compile(
        optimizer=optimizer,
        #loss=tfa.losses.WeightedKappaLoss(num_classes=5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    STEP_SIZE_TRAIN=train_generator.n//(train_generator.batch_size)
    STEP_SIZE_VALID=validation_generator.n//(validation_generator.batch_size)

    #Fit-train model
    start_time = time.time()
    history = model.fit(
                    train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=train_conf.EPOCHS,
                    validation_steps=STEP_SIZE_VALID,
                    validation_data=validation_generator,
                    #class_weight=[0.4, 25., 3, 33., 38.],
                    callbacks=[cp,es]
                    ).history
    elapsed_time = time.time() - start_time
    pretrain_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    write_to_log(log_dir, "\n\nElapsed time in the pretrain: " + pretrain_time)
    save_var(history, log_dir + "history_pretrained")
    model.save_weights(log_dir+"model_cmp719_senet18"+'.h5')
    return model, history

def evaluate_model(model,train_conf,generator_conf,dataset_conf,seed = 42):
    log_dir = train_conf.log_dir
    X_train, X_valid, train_dest_path, valid_dest_path = dataset_conf.get_dataset()

    test_datagen = ImageDataGenerator(rescale=generator_conf.rescale)
    validation_generator = test_datagen.flow_from_dataframe(
        dataframe=X_valid,
        directory=valid_dest_path,
        x_col="id_code",
        y_col="diagnosis",
        target_size=(train_conf.IMG_HEIGHT, train_conf.IMG_WIDTH),
        class_mode='categorical',
        batch_size=train_conf.BATCH_SIZE,
        seed=seed,
        shuffle=False)

    scores = model.evaluate(validation_generator)
    write_to_log(log_dir, "\nTest Loss Pretrained:{}".format(scores[0]))
    write_to_log(log_dir, "\nTest Accuracy Pretrained:{}".format(scores[1]))
    print("Test Loss Pretrained:{}".format(scores[0]))
    print("Test Accuracy Pretrained:{}".format(scores[1]))   

    #Save Confusion matrix
    y_pred = model.predict(validation_generator)
    y_pred = (tf.argmax(y_pred,1)).numpy() #Convert prob to class label 0-5
    y_true = X_valid["diagnosis"].astype(np.int64)
    report = plot_confusion_matrix(y_true, y_pred, classes=[0, 1, 2, 3, 4], save=True, saveDir=log_dir, normalize=True)

    return scores[1]

