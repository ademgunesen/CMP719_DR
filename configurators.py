import json
from utils import save_var, write_to_log, get_computer_info
import pandas as pd


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
                        ES_PATIENCE = 10,
                        RLROP_PATIENCE = 3,
                        DECAY_DROP = 0.5,
                        CLASS_WEIGHT = [1., 1., 1., 1. ,1.],
                        log_dir = ''
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
        self.CLASS_WEIGHT = CLASS_WEIGHT
        self.log_dir = log_dir

    def save_json(self):
        train_info={
            "name"                  : self.name,
            "NUM_SAMPLE"            : self.NUM_SAMPLE,
            "IMG_HEIGHT"            : self.IMG_HEIGHT,
            "IMG_WIDTH"             : self.IMG_WIDTH,
            "IMG_CHANNEL"           : self.IMG_CHANNEL,
            "BATCH_SIZE"            : self.BATCH_SIZE,
            "EPOCHS"                : self.EPOCHS,
            "WARMUP_EPOCHS"         : self.WARMUP_EPOCHS,
            "LEARNING_RATE"         : self.LEARNING_RATE,
            "WARMUP_LEARNING_RATE"  : self.WARMUP_LEARNING_RATE,
            "ES_PATIENCE"           : self.ES_PATIENCE,
            "RLROP_PATIENCE"        : self.RLROP_PATIENCE,
            "DECAY_DROP"            : self.DECAY_DROP,
            "CLASS_WEIGHT"          : self.CLASS_WEIGHT,
            "log_dir"               : self.log_dir
        }
        with open(self.log_dir + 'train_config_'+ self.name + '.json','w') as json_file:
            json.dump(train_info, json_file, sort_keys = True, indent = 4)

    def save(self, save_dir = ""):
        self.log_dir = save_dir
        log_dir = save_dir
        save_var(self, log_dir + 'train_config_'+ self.name)
        self.save_json()
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
        write_to_log(log_dir, "\nCLASS_WEIGHT: ")
        write_to_log(log_dir, str(self.CLASS_WEIGHT))
        write_to_log(log_dir, "\nlog_dir: ")
        write_to_log(log_dir, str(self.log_dir))

    def load_json(self, json_file=''):
        with open(json_file) as j_file:
            t_info = json.load(j_file)
        self.name=t_info["name"]
        self.NUM_SAMPLE=t_info["NUM_SAMPLE"]
        self.IMG_HEIGHT=t_info["IMG_HEIGHT"]
        self.IMG_WIDTH=t_info["IMG_WIDTH"]
        self.IMG_CHANNEL=t_info["IMG_CHANNEL"]
        self.BATCH_SIZE=t_info["BATCH_SIZE"]
        self.EPOCHS=t_info["EPOCHS"]
        self.WARMUP_EPOCHS=t_info["WARMUP_EPOCHS"]
        self.LEARNING_RATE=t_info["LEARNING_RATE"]
        self.WARMUP_LEARNING_RATE=t_info["WARMUP_LEARNING_RATE"]
        self.ES_PATIENCE=t_info["ES_PATIENCE"]
        self.RLROP_PATIENCE=t_info["RLROP_PATIENCE"]
        self.DECAY_DROP=t_info["DECAY_DROP"]
        self.CLASS_WEIGHT=t_info["CLASS_WEIGHT"]
        self.log_dir=t_info["log_dir"]

class generator_config:

    def __init__(self,  name= "default_config",
                        rotation_range=20,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0,
                        zoom_range=0.1,
                        channel_shift_range=0,
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode = "constant",
                        rescale=1./255,
                        log_dir = ''
        ):
    
        self.name = name
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.fill_mode = fill_mode
        self.rescale = rescale
        self.log_dir = log_dir

    def save_json(self):
        generator_info={
            "name"                  : self.name,
            "rotation_range"        : self.rotation_range,
            "width_shift_range"     : self.width_shift_range,
            "height_shift_range"    : self.height_shift_range,
            "shear_range"           : self.shear_range,
            "zoom_range"            : self.zoom_range,
            "channel_shift_range"   : self.channel_shift_range,
            "horizontal_flip"       : self.horizontal_flip,
            "vertical_flip"         : self.vertical_flip,
            "fill_mode"             : self.fill_mode,
            "rescale"               : self.rescale,
            "log_dir"               : self.log_dir
        }
        with open(self.log_dir + 'generator_config_'+ self.name + '.json','w') as json_file:
            json.dump(generator_info, json_file, sort_keys = True, indent = 4)

    def save(self, save_dir = ""):
        self.log_dir = save_dir
        log_dir = save_dir
        save_var(self, log_dir + 'generator_config_'+ self.name)
        self.save_json()

    def load_json(self, json_file=''):
        with open(json_file) as j_file:
            g_info = json.load(j_file)
        self.name = g_info['name']
        self.rotation_range = g_info['rotation_range']
        self.width_shift_range = g_info['width_shift_range']
        self.height_shift_range = g_info['height_shift_range']
        self.shear_range = g_info['shear_range']
        self.zoom_range = g_info['zoom_range']
        self.channel_shift_range = g_info['channel_shift_range']
        self.horizontal_flip = g_info['horizontal_flip']
        self.vertical_flip = g_info['vertical_flip']
        self.fill_mode = g_info['fill_mode']
        self.rescale = g_info['rescale']
        self.log_dir = g_info['log_dir']

class dataset_config:

    def __init__(self,  name        = "default_dataset",
                        train_path  = 'APTOS_Adems',
                        valid_path  = 'APTOS_Adems',
                        test_path   = 'APTOS_Adems',
                        csv_file    = 'label.csv',
                        log_dir     = ''
        ):
        
        self.name       = name
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path  = test_path 
        self.csv_file   = csv_file  
        self.log_dir    = log_dir

    def save_json(self):
        dataset_info={
            "name"          : self.name,
            "train_path"    : self.train_path,
            "valid_path"    : self.valid_path,
            "test_path"     : self.test_path,
            "csv_file"      : self.csv_file,  
            "log_dir"       : self.log_dir,
        }
        with open(self.log_dir + 'dataset_config_'+ self.name + '.json','w') as json_file:
            json.dump(dataset_info, json_file, sort_keys = True, indent = 4)

    def save(self, save_dir = ""):
        self.log_dir = save_dir
        log_dir = save_dir
        save_var(self, log_dir + 'dataset_config_'+ self.name)
        self.save_json()

    def load_json(self, json_file=''):
        with open(json_file) as j_file:
            d_info = json.load(j_file)
        self.name       = d_info['name']
        self.train_path = d_info['train_path']
        self.valid_path = d_info['valid_path']
        self.test_path  = d_info['test_path']
        self.csv_file   = d_info['csv_file']
        self.log_dir    = d_info['log_dir']

    def get_dataset(self):
        comp_info = get_computer_info()
        train_dest_path=comp_info["Datasets"]+self.train_path
        valid_dest_path=comp_info["Datasets"]+self.valid_path
        test_dest_path =comp_info["Datasets"]+self.test_path
        fold_df = pd.read_csv(comp_info["Datasets"]+self.csv_file)
        fold_df['diagnosis'] = fold_df['diagnosis'].astype('str')
        X_train = fold_df[fold_df['fold_0'] == 'train']
        X_valid = fold_df[fold_df['fold_0'] == 'validation']
        X_valid = X_valid.reset_index(drop=True)

        X_train_0 = pd.DataFrame(X_train[X_train['diagnosis'] == '0'])
        X_train_1 = pd.DataFrame(X_train[X_train['diagnosis'] == '1'])
        X_train_2 = pd.DataFrame(X_train[X_train['diagnosis'] == '2'])
        X_train_3 = pd.DataFrame(X_train[X_train['diagnosis'] == '3'])
        X_train_4 = pd.DataFrame(X_train[X_train['diagnosis'] == '4'])

        X_valid_0 = pd.DataFrame(X_valid[X_valid['diagnosis'] == '0'])
        X_valid_1 = pd.DataFrame(X_valid[X_valid['diagnosis'] == '1'])
        X_valid_2 = pd.DataFrame(X_valid[X_valid['diagnosis'] == '2'])
        X_valid_3 = pd.DataFrame(X_valid[X_valid['diagnosis'] == '3'])
        X_valid_4 = pd.DataFrame(X_valid[X_valid['diagnosis'] == '4'])

        print("Distribution of training set:")
        print(len(X_train_0),len(X_train_1),len(X_train_2),len(X_train_3),len(X_train_4))
        print("Distribution of validation set:")
        print(len(X_valid_0),len(X_valid_1),len(X_valid_2),len(X_valid_3),len(X_valid_4))
        X_train = X_train.append(X_train_1, ignore_index=True)
        X_train = X_train.append(X_train_3, ignore_index=True)
        X_train = X_train.append(X_train_3, ignore_index=True)
        X_train = X_train.append(X_train_3, ignore_index=True)
        X_train = X_train.append(X_train_3, ignore_index=True)
        X_train = X_train.append(X_train_4, ignore_index=True)
        X_train = X_train.append(X_train_4, ignore_index=True)
        X_train = X_train.append(X_train_4, ignore_index=True)
        X_train = X_train.append(X_train_4, ignore_index=True)
        #17132

        X_valid = X_valid.append(X_valid_1, ignore_index=True)
        X_valid = X_valid.append(X_valid_3, ignore_index=True)
        X_valid = X_valid.append(X_valid_3, ignore_index=True)
        X_valid = X_valid.append(X_valid_3, ignore_index=True)
        X_valid = X_valid.append(X_valid_3, ignore_index=True)
        X_valid = X_valid.append(X_valid_4, ignore_index=True)
        X_valid = X_valid.append(X_valid_4, ignore_index=True)
        X_valid = X_valid.append(X_valid_4, ignore_index=True)
        X_valid = X_valid.append(X_valid_4, ignore_index=True)
        #4285

        X_train = X_train.iloc[0:17124]
        X_valid = X_valid.iloc[0:4284]
        return X_train, X_valid, train_dest_path, valid_dest_path
