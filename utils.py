import numpy as np
import matplotlib.pyplot as plt
import os
import random
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import model_from_json
import pickle
import datetime 
import yagmail
from  tensorflow.keras.callbacks import Callback
import json

#PART FOR TEST, DELETE LATER
import copy
############################

def dataGenerator(dirName, fileList, labelList, augmentation=True, batchSize=8, shuffle=True, preprocessing=False):
    imgs = []
    labels = []
    dataNum=0
    if(shuffle == True):
	    shuffleDataset(fileList, labelList)
    while True:
        for tmp in range(batchSize):
            if(dataNum == len(fileList)):
                dataNum = 0
                if(shuffle == True):
                    shuffleDataset(fileList, labelList)
            #if(os.path.isfile(os.path.expanderuser('~/' + dirName + '/' + fileList[dataNum]))):
            #img = io.imread(dirName + '/' + fileList[dataNum])
            img=image.load_img(dirName + '/' + fileList[dataNum]+'.jpg', grayscale=False, target_size=(1024, 1024))
            img = image.img_to_array(img)
            if(augmentation==True):
                datagen = ImageDataGenerator(
                    rotation_range=360,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.2,
                    zoom_range=0.2,
                    channel_shift_range=20,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode = "nearest")
                img=datagen.random_transform(img)
                img=img/255.0
            else:
                img=img/255.0
    
            imgs.append(img)
            labels.append(labelList[dataNum])
            dataNum += 1
        yield (np.asarray(imgs), np.asarray(labels))

def shuffleDataset(fileList, labelList):
    tmp_zip=list(zip(fileList, labelList))
    random.shuffle(tmp_zip)
    fileList, labelList = zip(*tmp_zip) 
    return fileList, labelList
    
def preProcessing(image):
    #image = downscale_local_mean(image, (5, 5))
    #image=image/255.0
    gaus_img = skimage.filters.gaussian(image, sigma=image.shape[0]/10, truncate=0)
    tmp = copy.deepcopy(image)
    addWeighted(image, 4, gaus_img, -4, 1, tmp)
    print("SHOW IMAGE")
    show_images([image, tmp])
    return image

def addWeighted(src1, alpha, src2, beta, gamma, dst):
    #TODO:Saturatin will be added
    dst = (src1*alpha) + (src2*beta) + gamma
    print("source image and dst are different")

def orginize_dataset_folder(srcFileName='/home/vivente/Desktop/diabetic-retinopathy/dataset_19'):
    '''
    Divides dataset 80% - 20% as training and valdiation set.
    Reorganize the folder architecture as   data
                                            |__train
                                            |       |__folder_1
                                            |       |__folder_2
                                            |       |..
                                            |__validation
    '''
    df = pd.read_csv("labels/trainLabels19.csv")
    fileList, labelList = shuffleDataset(df["id_code"], df["diagnosis"])
    trainFile, validFile = fileList[:-732], fileList[2930:3662]
    trainLabel, validLabel= labelList[:-732], labelList[2930:3662]
    #Training
    for index in range(len(trainFile)):
        print(index, trainFile[index], trainLabel[index])
        img=Image.open(srcFileName + '/' + trainFile[index] + '.jpg')
        switcher = {
            0: "data/train/0",
            1: "data/train/1",
            2: "data/train/2",
            3: "data/train/3",
            4: "data/train/4",
        }
        destFileName = switcher.get(trainLabel[index], "nothing")
        if(destFileName == "nothing"):
            return -1 
        img.save(destFileName + '/' + trainFile[index] + '.jpg')
    #Validation
    for index in range(len(validFile)):
        print(index, validFile[index], validLabel[index])
        img=Image.open(srcFileName + '/' + validFile[index] + '.jpg')
        switcher = {
            0: "data/validation/0",
            1: "data/validation/1",
            2: "data/validation/2",
            3: "data/validation/3",
            4: "data/validation/4",
        }
        destFileName = switcher.get(validLabel[index], "nothing")
        if(destFileName == "nothing"):
            return -1 
        img.save(destFileName + '/' + validFile[index] + '.jpg')

def save_var(var, file_name):
    '''
    Saves any type of variable with the given filename(can be a path)
    '''
    out_file = open(file_name,'wb')
    pickle.dump(var,out_file)
    out_file.close()
    
def read_var(file_name):   
    infile = open(file_name,'rb')
    var = pickle.load(infile)
    infile.close()
    return var

class train_config:

    def __init__(self,  name= "default_config",
                        NUM_SAMPLE = 0,
                        IMG_HEIGHT = 256,
                        IMG_WIDTH = 256,
                        IMG_CHANNEL = 3,
                        BATCH_SIZE = 32,
                        EPOCHS = 30,
                        WARMUP_EPOCHS = 5,
                        LEARNING_RATE = 4*(1e-4),
                        WARMUP_LEARNING_RATE = 4*(1e-3),
                        ES_PATIENCE = 5,
                        RLROP_PATIENCE = 3,
                        DECAY_DROP = 0.5
    ):
    
        self.name = name
        self.NUM_SAMPLE = NUM_SAMPLE
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_CHANNEL = IMG_CHANNEL
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCHS = EPOCHS
        self.WARMUP_EPOCHS = WARMUP_EPOCHS
        self.LEARNING_RATE = LEARNING_RATE
        self.WARMUP_LEARNING_RATE = WARMUP_LEARNING_RATE
        self.ES_PATIENCE = ES_PATIENCE
        self.RLROP_PATIENCE = RLROP_PATIENCE
        self.DECAY_DROP = DECAY_DROP
        self.log_dir = "default_log_dir"
        '''
        LR_WARMUP_EPOCHS_1st = 2
        LR_WARMUP_EPOCHS_2nd = 5
        STEP_SIZE = len(X_train) // BATCH_SIZE
        TOTAL_STEPS_1st = WARMUP_EPOCHS * STEP_SIZE
        TOTAL_STEPS_2nd = EPOCHS * STEP_SIZE
        WARMUP_STEPS_1st = LR_WARMUP_EPOCHS_1st * STEP_SIZE
        WARMUP_STEPS_2nd = LR_WARMUP_EPOCHS_2nd * STEP_SIZE
        '''

    def save(self, save_dir = ""):
        self.log_dir = save_dir
        save_var(self, save_dir + self.name)
        log_dir = save_dir
        write_to_log(log_dir, "\n\nConfiguration name: ")
        write_to_log(log_dir, self.name)
        write_to_log(log_dir, "\nNUM_SAMPLE: ")
        write_to_log(log_dir, str(self.NUM_SAMPLE))
        write_to_log(log_dir, "\nIMG_HEIGHT: ")
        write_to_log(log_dir, str(self.IMG_HEIGHT))
        write_to_log(log_dir, "\nIMG_WIDTH: ")
        write_to_log(log_dir, str(self.IMG_WIDTH))
        write_to_log(log_dir, "\nIMG_CHANNEL: ")
        write_to_log(log_dir, str(self.IMG_CHANNEL))
        write_to_log(log_dir, "\nBATCH_SIZE: ")
        write_to_log(log_dir, str(self.BATCH_SIZE))
        write_to_log(log_dir, "\nEPOCHS: ")
        write_to_log(log_dir, str(self.EPOCHS))
        write_to_log(log_dir, "\nWARMUP_EPOCHS: ")
        write_to_log(log_dir, str(self.WARMUP_EPOCHS))
        write_to_log(log_dir, "\nLEARNING_RATE: ")
        write_to_log(log_dir, str(self.LEARNING_RATE))
        write_to_log(log_dir, "\nWARMUP_LEARNING_RATE: ")
        write_to_log(log_dir, str(self.WARMUP_LEARNING_RATE))
        write_to_log(log_dir, "\nES_PATIENCE: ")
        write_to_log(log_dir, str(self.ES_PATIENCE))
        write_to_log(log_dir, "\nRLROP_PATIENCE: ")
        write_to_log(log_dir, str(self.RLROP_PATIENCE))
        write_to_log(log_dir, "\nDECAY_DROP: ")
        write_to_log(log_dir, str(self.DECAY_DROP))


def save_model(model, name):
    model_json = model.to_json()
    with open(name+'.json', 'w') as json_file:
        json_file.write(model_json)

    model.save_weights(name+'.h5')
    print("Saved model to disk")

def load_model(name):
    json_file = open(name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name+'.h5')
    print("Loaded model from disk")
    return loaded_model

def make_subfolder(dirname,parent_path):
    path = os.path.join(parent_path, dirname)
    os.mkdir(path)
    print("Directory '%s' created" %dirname)
    return path + '/'

def make_log_dir(parent_path = ""):
    current_date = datetime.datetime.now() 
    dirname = current_date.strftime("%Y_%B_%d-%H_%M_%S")
    path = make_subfolder(dirname,parent_path)
    return path

def write_to_log(log_dir ="", log_entry = ""):
    with open(log_dir + "/log.txt", "a") as file:
        file.write(log_entry)

def send_as_mail(log_dir):
    log = log_dir + '/log.txt'
    conf_mat = log_dir + '/confusionMatrix.png'
    loss_and_acc = log_dir + '/loss_and_acc.png'
    contents = [ "Train sonuçları ve konfigürasyonu ekte yer almaktadır",
    log, loss_and_acc, conf_mat
    ]
    with yagmail.SMTP('viventedevelopment', 'yeniparrola2.1') as yag:
        yag.send('ademgunesen+viventedev@gmail.com', 'Train Sonuçları', contents)

def quadratic_kappa(y_true, y_pred):
    """This function calculates the Quadratic Kappa Metric used for Evaluation in the PetFinder competition
    at Kaggle. It returns the Quadratic Weighted Kappa metric score between the actual and the predicted values 
    of adoption rating."""
    N=5
    w = np.zeros((N,N))
    O = confusion_matrix(y_true, y_pred)
    for i in range(len(w)): 
        for j in range(len(w)):
            w[i][j] = float(((i-j)**2)/(N-1)**2)
    
    act_hist=np.zeros([N])
    for item in y_true: 
        act_hist[item]+=1
    
    pred_hist=np.zeros([N])
    for item in y_pred: 
        pred_hist[item]+=1
                         
    E = np.outer(act_hist, pred_hist);
    E = E/E.sum();
    O = O/O.sum();
    
    num=0
    den=0
    for i in range(len(w)):
        for j in range(len(w)):
            num+=w[i][j]*O[i][j]
            den+=w[i][j]*E[i][j]
    return (1 - (num/den))

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1
        
        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred, 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('out/model_arabasamak.h5')

        return

def get_computer_info():
    f = open('computer_info.json',)
    computer_info = json.load(f) 
    print("Working on "+computer_info['name'])
    return computer_info


def show_images(images: list, titles: list="Untitled    ", colorScale='gray', rows = 0, columns = 0) -> None:
    n: int = len(images)
    if rows == 0:
        rows=int(math.sqrt(n))
    if columns == 0:
        columns=(n/rows)
    f = plt.figure()
    for i in range(n):
        # Debug, plot figure
        f.add_subplot(rows, columns, i + 1)
        plt.imshow(images[i], cmap=colorScale)
        plt.title(titles[i])
    plt.show(block=True)
