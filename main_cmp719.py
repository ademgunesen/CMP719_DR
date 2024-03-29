import tensorflow as tf
import utils_cmp719
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)
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
from utils import train_config, make_log_dir, write_to_log, save_var, save_model, send_as_mail

from tensorflow.keras import layers

model = utils_cmp719.choose_nets('inceptionv4', 5)

comp_info = util.get_computer_info()

train_dest_path=comp_info["Datasets"]+'APTOS_Adems'
valid_dest_path=comp_info["Datasets"]+'APTOS_Adems'
test_dest_path='Not defined'
seed = 1

fold_df = pd.read_csv(comp_info["Datasets"]+'trainLabels15.csv')
fold_df['diagnosis'] = fold_df['diagnosis'].astype('str')

fold_df_19 = pd.read_csv(comp_info["Datasets"]+'label.csv')
fold_df_19['diagnosis'] = fold_df_19['diagnosis'].astype('str')
fold_df_19 = fold_df_19[fold_df_19['data'] == 'new']

X_train = fold_df[fold_df['fold_0'] == 'train']
X_valid = fold_df[fold_df['fold_0'] == 'validation']
X_train_1 = pd.DataFrame(X_train[X_train['diagnosis'] == '1'])
X_valid_1 = pd.DataFrame(X_valid[X_valid['diagnosis'] == '1'])
X_train_3 = X_train[X_train['diagnosis'] == '3']
X_valid_3 = X_valid[X_valid['diagnosis'] == '3']
X_train_4 = X_train[X_train['diagnosis'] == '4']
X_valid_4 = X_valid[X_valid['diagnosis'] == '4']

X_train = X_train.append(X_train_1, ignore_index=True)
X_train = X_train.append(X_train_1, ignore_index=True)
X_train = X_train.append(X_train_1, ignore_index=True)
X_train = X_train.append(X_train_1, ignore_index=True)
X_train = X_train.append(X_train_1, ignore_index=True)
X_train = X_train.append(X_train_3, ignore_index=True)
X_train = X_train.append(X_train_3, ignore_index=True)
X_train = X_train.append(X_train_3, ignore_index=True)
X_train = X_train.append(X_train_4, ignore_index=True)
X_train = X_train.append(X_train_4, ignore_index=True)
X_train = X_train.append(X_train_4, ignore_index=True)
X_train = X_train.append(fold_df_19, ignore_index=True)

X_valid = X_valid.append(X_valid_1, ignore_index=True)
X_valid = X_valid.append(X_valid_1, ignore_index=True)
X_valid = X_valid.append(X_valid_1, ignore_index=True)
X_valid = X_valid.append(X_valid_1, ignore_index=True)
X_valid = X_valid.append(X_valid_1, ignore_index=True)
X_valid = X_valid.append(X_valid_3, ignore_index=True)
X_valid = X_valid.append(X_valid_3, ignore_index=True)
X_valid = X_valid.append(X_valid_3, ignore_index=True)
X_valid = X_valid.append(X_valid_4, ignore_index=True)
X_valid = X_valid.append(X_valid_4, ignore_index=True)
X_valid = X_valid.append(X_valid_4, ignore_index=True)

X_valid = X_valid.reset_index(drop=True)

X_train = X_train.iloc[0:45312]
X_valid = X_valid.iloc[0:10404]






candidate_config_list = []
candidate_config_list.append(train_config(name = "inceptionv4", IMG_HEIGHT = 256, IMG_WIDTH = 256, 
                                        BATCH_SIZE = 12, EPOCHS=50,
                                        LEARNING_RATE=4*(1e-5)))


for conf in candidate_config_list : 

    log_dir = make_log_dir("out/")
    write_to_log(log_dir, "train_dest_path: \n" + train_dest_path)
    write_to_log(log_dir, "\n\nvalid_dest_path: \n" + valid_dest_path)
    write_to_log(log_dir, "\n\ntest_dest_path: \n" + test_dest_path)

    conf.save(save_dir = log_dir)

    train_datagen = ImageDataGenerator(
                        rotation_range=360,
                        #width_shift_range=0.1,
                        #height_shift_range=0.1,
                        #shear_range=0.2,
                        zoom_range=0.1,
                        #channel_shift_range=20,
                        horizontal_flip=True,
                        vertical_flip=True,
                        #fill_mode = "nearest",
                        rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
            dataframe=X_train,
            directory=train_dest_path,
            x_col='id_code',
            y_col='diagnosis',
            target_size=(conf.IMG_HEIGHT, conf.IMG_WIDTH),
            class_mode='categorical',
            batch_size=conf.BATCH_SIZE,
            seed=seed)

    validation_generator = test_datagen.flow_from_dataframe(
            dataframe=X_valid,
            directory=valid_dest_path,
            x_col="id_code",
            y_col="diagnosis",
            target_size=(conf.IMG_HEIGHT, conf.IMG_WIDTH),
            class_mode='categorical',
            batch_size=conf.BATCH_SIZE,
            seed=seed,
            shuffle=False)

    optimizer=tf.keras.optimizers.Adam(lr=conf.LEARNING_RATE)
    
    model.build(input_shape=(conf.BATCH_SIZE ,conf.IMG_HEIGHT, conf.IMG_WIDTH, 3))
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
        #metrics=[quadratic_kappa()]
    )
    x = Input(shape =  (256,256,3,),batch_size = 12)
    model(x)
    model.summary()
    #model.load_weights("model_cmp719_senet"+'.h5')
    #print("Loaded model from disk")
    
    STEP_SIZE_TRAIN=train_generator.n//(train_generator.batch_size)
    STEP_SIZE_VALID=validation_generator.n//(validation_generator.batch_size)
    print(STEP_SIZE_TRAIN)
    print(STEP_SIZE_VALID)
    

    #Fit-train model
    start_time = time.time()
    history_trained = model.fit(
                            train_generator,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            epochs=conf.EPOCHS,
                            validation_steps=STEP_SIZE_VALID,
                            validation_data=validation_generator
                            ).history
    elapsed_time = time.time() - start_time
    train_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    write_to_log(log_dir, "\n\nElapsed time in the train: " + train_time)
    save_var(history_trained, log_dir + "history_trained")
    #Evaluate trained model
    scores = model.evaluate(validation_generator)
    write_to_log(log_dir, "\nTest Loss:{}".format(scores[0]))
    write_to_log(log_dir, "\nTest Accuracy:{}".format(scores[1]))
    print("Test Loss:{}".format(scores[0]))
    print("Test Accuracy:{}".format(scores[1]))
    #save_model(model, log_dir + "model_cmp719_senet")
    model.save_weights(log_dir+"model_cmp719_senet"+'.h5')
    print("Saved model to disk")

    #Evaluate trained model
    scores = model.evaluate(validation_generator)
    write_to_log(log_dir, "\nTest Loss:{}".format(scores[0]))
    write_to_log(log_dir, "\nTest Accuracy:{}".format(scores[1]))
    print("Test Loss:{}".format(scores[0]))
    print("Test Accuracy:{}".format(scores[1]))

    #Save Confusion matrix
    y_pred = model.predict_generator(validation_generator)
    y_pred = (tf.argmax(y_pred,1)).numpy() #Convert prob to class label 0-5
    y_true = X_valid["diagnosis"].astype(np.int64)
    report = plot_confusion_matrix(y_true, y_pred, classes=[0, 1, 2, 3, 4], save=True, saveDir=log_dir)
    report_perc = plot_confusion_matrix(y_true, y_pred, classes=[0, 1, 2, 3, 4], save=True, saveDir=log_dir, normalize=True, title='Confusio matrice percentage')


    #Visualize history

    plot_loss_and_acc(history_trained, save=True, saveDir=log_dir)
    write_to_log(log_dir, "\n\n" + report)
    #send_as_mail(log_dir)
