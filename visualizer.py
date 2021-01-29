import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from configurators import train_config, generator_config, dataset_config
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import utils as util

def plot_history(history):
    '''
    Takes history dictionary and plot accuracy and loss graph
    '''
    plot_acc(history)
    plot_loss(history)

def plot_loss_and_acc(history, save=False, saveDir='out/', fname=''):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 18))

    ax1.plot(history['loss'], label='Train loss')
    ax1.plot(history['val_loss'], label='Validation loss')
    ax1.legend(loc='best')
    ax1.set_title('Loss')

    ax2.plot(history['accuracy'], label='Train accuracy')
    ax2.plot(history['val_accuracy'], label='Validation accuracy')
    ax2.legend(loc='best')
    ax2.set_title('Accuracy')

    plt.xlabel('Epochs')
    if(save):
        plt.savefig(saveDir + fname + 'loss_and_acc.png')
    plt.show(block=False)

def plot_acc(history):
    '''
    TODO:Instead of history whats should be the input?
    Takes array of Plot training & validation accuracy values
    '''
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def plot_loss(history):
    '''
    TODO:Instead of history whats should be the input?
    Plot training & validation loss values
    '''
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save=False,
                          saveDir='out/'):
    """
    This function calculates, prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    report = classification_report(y_true, y_pred, digits=3)
    print(report)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if(save):
        plt.savefig(saveDir + title+ '.png')
    plt.show(block=False)
    return report


'''
def show_data_stat(labelList, dataList):
    #Plot data distrubition on each class
'''
def visualise_batches(train_conf,generator_conf,dataset_conf, seed = 42):
    X_train, X_valid, train_dest_path, valid_dest_path = dataset_conf.get_dataset()
    log_dir = train_conf.log_dir
    train_datagen = ImageDataGenerator(
                        rotation_range      = generator_conf.rotation_range,
                        width_shift_range   = generator_conf.width_shift_range,
                        height_shift_range  = generator_conf.height_shift_range,
                        shear_range         = generator_conf.shear_range,
                        zoom_range          = generator_conf.zoom_range,
                        #channel_shift_range= 20,
                        horizontal_flip     = generator_conf.horizontal_flip,
                        vertical_flip       = generator_conf.vertical_flip,
                        fill_mode           = generator_conf.fill_mode,
                        rescale             = generator_conf.rescale)
    test_datagen = ImageDataGenerator(rescale=generator_conf.rescale)

    train_generator = train_datagen.flow_from_dataframe(
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

    for i in range(3):
        batch = train_generator.next()
        util.show_images([batch[0][0],batch[0][1],batch[0][2],batch[0][3]])

if __name__ == "__main__":

    t_conf = train_config(name = "Esma",
                    BATCH_SIZE = 12, 
                    EPOCHS=5)

    g_conf = generator_config(name = "default")
    d_conf = dataset_config(name = "default")
    visualise_batches(t_conf,g_conf,d_conf)
