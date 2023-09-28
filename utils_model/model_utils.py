import os
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import backend as K

from utils_model.data_helper import *
from utils_model.process_bpm import *

'''
In this python file, we define the auxialiar functions to the model, such as the loss functions, the metrics, the callbacks, etc. 
'''


''' LOAD INFO OF A MODEL '''

class FileProcessor:
    def __init__(self, info_dir,root):
        self.info_dir = info_dir
        self.params = nested_dict(2, str)
        self.root = root
    def read_file(self):
        """
        This function reads the txt file and gets the dictionary
        """
        for dataset in self.info_dir.keys():
            with open(self.info_dir[dataset]['dir']+'/info.txt') as f:
                for i, line in enumerate(f.readlines()):
                    if i > 0:
                    #stop when the next line is empty
                        if line == '\n':
                            break
                        key = line.split(' ')[0]
                        val = line.replace(key, '')
                        val = val.replace('\n', '')
                        val = val.replace(' ', '')
                        val = val.replace('"', '')
                        self.params[dataset][key] = val

    def clean_dictionary(self):
        """
        This function cleans the dictionary to make sure it's in the right format.
        """
        for dataset in self.info_dir.keys():
            for key in self.params[dataset].keys():
                try:
                    self.params[dataset][key] = int(self.params[dataset][key])
                except:
                    try:
                        self.params[dataset][key] = float(self.params[dataset][key])
                    except:
                        if '[' in self.params[dataset][key]:
                            self.params[dataset][key] = self.params[dataset][key].strip('][').split(',')
                            for i, element in enumerate(self.params[dataset][key]):
                                self.params[dataset][key][i] = self.params[dataset][key][i].replace('"', '')
                                self.params[dataset][key][i] = self.params[dataset][key][i].replace('\'', '')

    def convert_numbers(self):
        """
        This function converts the values in the lists to numbers if they are numbers.
        """
        for dataset in self.info_dir.keys():
            for key in self.params[dataset].keys():
                if isinstance(self.params[dataset][key], list):
                    for i, value in enumerate(self.params[dataset][key]):
                        try:
                            self.params[dataset][key][i] = int(self.params[dataset][key][i])
                        except:
                            pass

    def write_keys(self):
        """
        This function writes two new keys in info_dir: 'info' and 'weights'. This is to know where the files are located
        """
        for dataset in self.info_dir.keys():
            if dataset == 'Overall':
                dat = '-'.join(self.params[dataset]['datasets'])
            else:
                dat = self.params[dataset]['datasets'][0]
            info = 'datasets_' + str(dat) + '_landmarks_' + self.params[dataset]['landmarks'] + '_win_secs_' + str(self.params[dataset]['win_secs']) + \
                '_act_filter_' + self.params[dataset]['act_filter'] + '_loss_' + self.params[dataset]['loss'] + \
                '_norm_' + self.params[dataset]['norm'] + '_lr_scheduler_' + self.params[dataset]['lr_scheduler'] + '_optimizer_' + self.params[dataset]['optimizer'] + '_epochs_' + str(
                self.params[dataset]['epochs']) + \
                '_batch_size_' + str(self.params[dataset]['batch_size']) + '_'
            self.info_dir[dataset]['info'] = info + self.info_dir[dataset]['date'] 

            # Choose the last weight
            checkpoints_dir = self.info_dir[dataset]['dir'] + '/checkpoints/'
            # checkpoints_dir = self.root + self.info_dir[dataset]['info'] + '/checkpoints/'
            weights = [f.path for f in os.scandir(checkpoints_dir)]
            weights = weights[-1]
            self.info_dir[dataset]['weights'] = weights


def choose_best_model(root, datasets, loss, normaliz, windows):
    """
    Choose the best model for a given dataset.

    Parameters:
    - root (str): the root directory where the models are located
    - datasets (list): the datasets to choose the model from
    - loss (str): the loss function to be used
    - normaliz (str): the normalization method to be used
    - windows (str): the windows method to be used

    Returns:
    - info_dir (dict): a dictionary containing the information of the chosen model for each dataset
    """

    # Create a nested dictionary to store the information of the chosen model
    info_dir = nested_dict(2, str)

    for dataset in datasets:
        if dataset == 'Overall':
            substr = 'MR-NIRP-PURE-LGI-PPGI'
        else:
            substr = '_'+dataset+'_'
        min_loss = 100000
        
        # Get all the model directories
        models_dir = [f.path for f in os.scandir(root) if f.is_dir()]
        print('-----------------\n',dataset)

        # Find the best loss
        for model_dir in models_dir:
            if (normaliz in model_dir) and (str(windows) in model_dir) and (substr in model_dir) and (loss in model_dir):
                checkpoints_dir = os.path.join(model_dir, 'checkpoints')
                weights = [f.path for f in os.scandir(checkpoints_dir)]
                val_losses = [float(weight.split('\\')[-1].split('-')[-1][:-3]) for weight in weights]
                if val_losses:
                    curr_min_loss = np.min(val_losses)
                    if curr_min_loss < min_loss:
                        min_loss = curr_min_loss
                        min_dir = model_dir
                        
        # Choose all the models with the minimum loss
        models_list = []
        for model_dir in models_dir:
            if (normaliz in model_dir) and (str(windows) in model_dir) and (substr in model_dir) and (loss in model_dir):
                checkpoints_dir = os.path.join(model_dir, 'checkpoints')
                weights = [f.path for f in os.scandir(checkpoints_dir)]
                val_losses = [float(weight.split('\\')[-1].split('-')[-1][:-3]) for weight in weights]
                if val_losses:
                    curr_min_loss = np.min(val_losses)
                    if curr_min_loss == min_loss:
                        models_list.append(model_dir)
                        
        if models_list:
            min_dir = np.random.choice(models_list)
            date_str = min_dir.split('\\')[-1].split('_')[-1]
            info_dir[dataset]['dir'] = min_dir
            info_dir[dataset]['date'] = date_str
            print('dirs:',models_list)
            print('Chosen model:',min_dir)
            print('date',date_str)
            print('best loss:',min_loss)
    return info_dir



''' MODEL HELPER FUNCTIONS'''

''' -- PLOT ALL THE METRICS OF THE MODEL'''
def plot_save_metrics(history,model_dir,save=True):
    plt.rcParams["figure.figsize"] = (10,3)
    metrics = []
    for key in history.history.keys():
        if 'val' not in key: 
            metrics.append(key)
    for metric in metrics: 
        print(metric)
        plt.plot(history.history[metric],label=metric)
        if metric != 'lr':
            plt.plot(history.history['val_'+metric],label='val_'+metric)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
        fig = plt.gcf() # This is to be able to save and show the figure
        if save:
            fig.savefig(model_dir+'plots\\'+ metric +'.png')
        plt.show()
        
''' LOSS FUNCTION'''

class DtwLoss(tf.keras.losses.Loss):
    def _init_(self):
        # super(DtwLoss, self)._init_()
        self.batch_size = 24

    def call(self, y_true, y_pred):
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        print(tf.shape(y_true), tf.shape(y_pred))
        self.batch_size = 24
        tmp = []
        for item in range(self.batch_size):
            # tf.print(f'Working on batch: {item}\n')
            s = y_true[item, :]
            t = y_pred[item, :]
            n, m = len(s), len(t)
            dtw_matrix = []
            for i in range(n + 1):
                line = []
                for j in range(m + 1):
                    if i == 0 and j == 0:
                        line.append(0)
                    else:
                        line.append(np.inf)
                dtw_matrix.append(line)

            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = tf.abs(s[i - 1] - t[j - 1])
                    last_min = tf.reduce_min([dtw_matrix[i - 1][j], dtw_matrix[i][j - 1], dtw_matrix[i - 1][j - 1]])
                    dtw_matrix[i][j] = tf.cast(cost, dtype=tf.float32) + tf.cast(last_min, dtype=tf.float32)

            temp = []
            for i in range(len(dtw_matrix)):
                temp.append(tf.stack(dtw_matrix[i]))

            tmp.append(tf.stack(temp)[n, m])
        return tf.reduce_mean(tmp)
    

class All(keras.losses.Loss):
    def __init__(self,name="custom_mse"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):

        return self.loss(y_true,y_pred)
    
     
class CustomLoss(keras.losses.Loss):
    def __init__(self,loss,regularization_factor=0.1, name="custom_mse"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor
        self.loss = loss

    def call(self, y_true, y_pred):
        # mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        # reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
        # return mse + reg * self.regularization_factor
        print(y_true.shape,y_pred.shape)
        # val = dtw.distance(y_true,y_pred)
        res =  self.loss(y_true,y_pred)
        return res

def pearson_coefficient_loss(y_true, y_pred):
    y_true_mean = tf.reduce_mean(y_true)
    y_pred_mean = tf.reduce_mean(y_pred)
    y_true_centered = y_true - y_true_mean
    y_pred_centered = y_pred - y_pred_mean
    numerator = tf.reduce_sum(y_true_centered * y_pred_centered)
    denominator = tf.sqrt(tf.reduce_sum(tf.square(y_true_centered))) * tf.sqrt(tf.reduce_sum(tf.square(y_pred_centered)))
    return 1 - numerator / denominator

def bpm_loss(y_true, y_pred):
    get_BPM(y_true,30)
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred) + 0.1 * tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def choose_metric(name_loss,custom=None):
    if name_loss == 'mse':
        loss = tf.keras.losses.MeanSquaredError()
    elif name_loss == 'mae':
        loss = tf.keras.losses.MeanAbsoluteError()
    elif name_loss == 'mape':
        loss = tf.keras.losses.MeanAbsolutePercentageError()
    elif name_loss == 'cos_sim':
        loss = tf.keras.losses.CosineSimilarity()
    elif name_loss == 'r':
        loss = pearson_coefficient_loss  
    # elif name_loss == 'dtw':
        # loss = CustomLoss(dtw.distance)
        # loss = CustomLoss(pearson_coefficient_loss)
        # loss = DtwLoss() 
    elif name_loss == 'bpm':
        print('get the bpm loss')
        # loss = CustomLoss(custom)
        loss = bpm_loss
    elif name_loss == 'rmse':
        loss = root_mean_squared_error
    elif name_loss == 'custom':
        loss = [root_mean_squared_error,pearson_coefficient_loss]
    else: 
        print('Name of loss not valid!')
        loss=None
    return loss



''' LEARNING RATE SCHEDULERS'''

# plot_lr = False
# if plot_lr:
#     lr = [1e-3]
#     for i in range(100):
#         lr.append(lr_exp(i,lr[-1]))
#     plt.plot(lr)

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

def lr_exp(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def choose_lr_scheduler(name_lr_scheduler,lr_custom = None):
    if name_lr_scheduler == 'exponential_decay':
        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=.1,decay_steps=30,decay_rate=0.01)
    elif name_lr_scheduler == 'cosine_decay':
        lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-2,decay_steps=10000)
    elif name_lr_scheduler == 'cosine_decay_restarts':
        lr_scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=.1,first_decay_steps=5)
    elif name_lr_scheduler == 'inverse_time_decay':
        lr_scheduler = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=.1,decay_steps=1,decay_rate = 0.5)
    elif name_lr_scheduler == 'piece_wise_constant_decay':
        lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(initial_learning_rate=1e-2,decay_steps=10000)
    elif name_lr_scheduler == 'polynomial_decay':
        lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=1e-2,decay_steps=10000)
    elif name_lr_scheduler == 'plateau':
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=10, min_lr=1e-5,min_delta=0.0001) #factor 0.1
    elif name_lr_scheduler == 'custom':
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_custom) # lr_exp function
    elif name_lr_scheduler == 'cyclical':
        steps_per_epoch = 2
        lr_scheduler = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=1e-4,
            maximal_learning_rate=1e-2,
            scale_fn=lambda x: 1/(2.**(x-1)),
            step_size=2 * steps_per_epoch
        )
    else: 
        print('Name of lr_scheduler not valid!')
        lr_scheduler=None
    step = np.arange(0, 200)
    if (name_lr_scheduler!='plateau') and (name_lr_scheduler != 'custom'):
        lr = lr_scheduler(step)
        plt.plot(step, lr)
        plt.xlabel("Steps")
        plt.ylabel("Learning Rate")
        plt.show()
    return lr_scheduler



''' OPTIMIZERS '''


def choose_optimizer(name_optimizer):
    learning_rate=0.001
    if name_optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    if name_optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif name_optimizer == 'sgd':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif name_optimizer == 'adafactor':
        optimizer = tf.keras.optimizers.Adafactor(learning_rate=learning_rate)
    elif name_optimizer == 'nadam':
        optimizer = tf.keras.optimizers.experimental.Nadam(learning_rate=learning_rate)
    elif name_optimizer == 'adamw':
        optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=learning_rate)
    elif name_optimizer == 'ftrl':
        optimizer = tf.keras.optimizers.experimental.Ftrl(learning_rate=learning_rate)
    elif name_optimizer == 'adamax':
        optimizer = tf.keras.optimizers.experimental.Adamax(learning_rate=learning_rate)
    elif name_optimizer == 'adagrad':
        optimizer = tf.keras.optimizers.experimental.Adagrad(learning_rate=learning_rate)
    elif name_optimizer == 'adadelta':
        optimizer = tf.keras.optimizers.experimental.Adadelta(learning_rate=learning_rate)
    else: 
        print('Name of optimizer not valid!')
        optimizer=None
    return optimizer


def choose_optimizer_with_lr(name_optimizer,lr):
    if name_optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(lr)
    if name_optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr)
    elif name_optimizer == 'sgd':
        optimizer = tf.keras.optimizers.RMSprop(lr)
    elif name_optimizer == 'adafactor':
        optimizer = tf.keras.optimizers.Adafactor(lr)
    elif name_optimizer == 'nadam':
        optimizer = tf.keras.optimizers.experimental.Nadam(lr)
    elif name_optimizer == 'adamw':
        optimizer = tf.keras.optimizers.experimental.AdamW(lr)
    elif name_optimizer == 'ftrl':
        optimizer = tf.keras.optimizers.experimental.Ftrl(lr)
    elif name_optimizer == 'adamax':
        optimizer = tf.keras.optimizers.experimental.Adamax(lr)
    elif name_optimizer == 'adagrad':
        optimizer = tf.keras.optimizers.experimental.Adagrad(lr)
    elif name_optimizer == 'adadelta':
        optimizer = tf.keras.optimizers.experimental.Adadelta(lr)
    else: 
        print('Name of optimizer not valid!')
        optimizer=None
    return optimizer



''' CALLBACKS '''

class CustomCallback(keras.callbacks.Callback):

    def __init__(self,model_dir):
        super().__init__()
        self.model_dir = model_dir

    def on_epoch_end(self, epoch,logs=None):
        keys = list(logs.keys())
        with open(self.model_dir + 'info.txt', 'a') as f:
            f.write("\nEpoch {}\n".format(epoch))
            for key in keys:
                f.write(' '+key+': '+str(logs[key]))


def choose_callbacks(params,write_dir,model_dir,logdir,lr_scheduler):
   callbacks = []
   # -- CALLBACKS
   early_stopping = keras.callbacks.EarlyStopping(
      monitor="val_loss",
      min_delta=0.0001,
      patience=30,
      verbose=1)
   # tb = tf.keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1,write_graph=True)
   checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                                filepath=model_dir+'checkpoints\\model.{epoch:02d}-{val_loss:.3f}.h5',
                                                monitor = 'val_loss',
                                                save_best_only = True,
                                                save_weights_only = True,
                                                save_freq='epoch')  
   if params['early_stopping']:
      callbacks = [early_stopping]

   if params['lr_scheduler']=='plateau':
      callbacks += [lr_scheduler]
   if write_dir: 
      callbacks += [checkpoint,CustomCallback(model_dir)]
   return callbacks

def create_dirs(params,write_dir):
   dat = '-'.join(params['datasets'])
   timestr = time.strftime("%Y%m%d-%H%M%S")
   info = 'datasets_{}_landmarks_{}_win_secs_{}_act_filter_{}_loss_{}_norm_{}_lr_scheduler_{}_optimizer_{}_epochs_{}_batch_size_{}_{}'.format(
      dat, params['landmarks'], params['win_secs'], params['act_filter'], params['loss'], params['norm'],
      params['lr_scheduler'], params['optimizer'], params['epochs'], params['batch_size'], timestr)
   model_dir = '.\\info\\models\\{}\\'.format(info)
   logdir = '.\\info\\logs\\{}\\'.format(info)
   if write_dir:
      if not os.path.exists(model_dir):
         os.makedirs(model_dir)
         os.makedirs(model_dir + 'checkpoints')
         os.makedirs(model_dir + 'plots')
         os.makedirs(logdir)
      with open(model_dir + 'info.txt', 'w') as f:
         f.write('Params:\n')
         for key, value in params.items():
               f.write('{} {}\n'.format(key, value))
   return model_dir,logdir