import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from utils_model.data_helper import *
from utils_model.model_utils import *

'''
In this python script, we define functions to prepare the data for training and testing the model.
'''

''' FUNCTIONS TO PREPARE THE DATA AND RUN THE MODEL'''

def get_names(params, data):
    '''
    Function that selects the data for the K-fold cross validation.

    Args:
        params (dict): A dictionary of parameters for the function. Should contain the following keys:
            - 'datasets': A list of strings representing the names of the datasets to be used for training and/or testing.
            - 'datasets_only_test': A list of strings representing the names of the datasets that should only be used for testing.
            - 'K_fold': An integer representing the number of folds for K-fold cross validation.
        data (dict): A dictionary containing the data to be used for training and/or testing. Should be nested such that
            the top-level keys correspond to dataset names, and the values are dictionaries of data for that dataset.

    Returns:
        A list of dictionaries, where each dictionary corresponds to a single fold of the K-fold cross validation.
        Each dictionary has two keys: 'train' and 'test'. The values are dictionaries themselves, where the keys correspond
        to dataset names and the values are lists of subject names (strings).
    '''

    # Take all the names of subjects
    names = {}
    for dataset in params['datasets']:
        names[dataset] = list(data[dataset].keys())

    # Split data into train and test sets for each fold of K-fold cross validation
    if params['K_fold'] != 1:
        # Create a list of empty dictionaries, one for each fold
        names_splits = [nested_dict(2, list) for i in range(params['K_fold'])]
        # Use sklearn's KFold class to generate train/test indices for each dataset
        kf = sklearn.model_selection.KFold(n_splits=params['K_fold'], shuffle=False)
        for dataset in params['datasets']:
            if dataset not in params['datasets_only_test']:
                # If dataset is not only for testing, split it into train/test for each fold
                for i, (train_index, test_index) in enumerate(kf.split(names[dataset])):
                    # Add the subject names for the training and testing sets to the corresponding dictionaries
                    names_splits[i]['train'][dataset] = [names[dataset][i] for i in train_index]
                    names_splits[i]['test'][dataset] = [names[dataset][i] for i in test_index]
            else:
                # If dataset is only for testing, add all subject names to the testing set for each fold
                names_splits[i]['test'][dataset] = names[dataset]
    else: 
        # If K_fold=1, split each dataset into a single train/test set
        names_splits = [nested_dict(2, list)]
        for dataset in params['datasets']:
            if dataset not in params['datasets_only_test']:
                # If dataset is not only for testing, split it into train/test
                train_ind, test_ind = train_test_split(np.arange(0,len(names[dataset])), test_size=0.2,random_state=10)
                names_splits[0]['train'][dataset] = [names[dataset][i] for i in train_ind]
                names_splits[0]['test'][dataset] = [names[dataset][i] for i in test_ind]
            else:
                # If dataset is only for testing, add all subject names to the testing set
                names_splits[0]['test'][dataset] = names[dataset]

    # print the names of the subjects in each fold
    for i in range(params['K_fold']):
        print(f"Fold {i}:")
        print(f"  Train: {names_splits[i]['train']}")
        print(f"  Test:  {names_splits[i]['test']}")

    return names_splits


def get_data(names_splits, params, data, samples, n_methods, all_landmarks, landmarks,ground_truth='gt'):
    '''Selects the data for the K-fold cross validation.
    
    Args:
    - names_splits (list of dicts): List of dictionaries containing the names of subjects for training and testing
                                    for each fold.
    - params (dict): A dictionary containing various parameters such as 'act_filter', 'methods_list', and 'K_fold'.
    - data (dict): A dictionary containing the preprocessed data for each dataset.
    - samples (int): The number of samples in each window.
    - n_methods (int): The number of feature extraction methods used.
    - all_landmarks (int): The total number of landmarks.
    - landmarks (list of ints): A list of integers representing the indices of the landmarks to be used.
    
    Returns:
    - data_splits (list of dicts): A list of dictionaries containing the data for each fold. For each fold, there are
                                   four keys: 'x_train', 'x_test', 'y_train', and 'y_test'. The values for each key
                                   are 3D tensors of shape (#videos x #windows, samples, n_methods x n_landmarks).
    '''

    # First select the activities to include
    if params['act_filter'] == 'resting':
        act_filter = {'LGI-PPGI':['resting'],'PURE':['01','03'],'MR-NIRP':['still']}
    elif params['act_filter'] == 'resting+':
        act_filter = {'LGI-PPGI':['resting','rotation'],'PURE':['01','02','03','04','05','06'],'MR-NIRP':['still']}
    else: 
        act_filter = {'LGI-PPGI':['resting','talk','gym','rotation'],'PURE':['01','02','03','04','05','06'],'MR-NIRP':['still','motion'] }

    # Now i have all the splits, get the data for each split.
    data_splits = [{} for i in range(params['K_fold'])]
    for i in range(params['K_fold']):
        names_train = names_splits[i]['train']
        names_test = names_splits[i]['test']

        datasets_train = list(names_train.keys())
        datasets_test = list(names_test.keys())
        
        # if ground_truth is 'bpm', then create a parameter called methods_list that includes the 'bpm' method and removes the 'gt' method
        if ground_truth == 'bpm':
            methods_list = params['methods_list'] + ['bpm']
            methods_list.remove('gt')
        else: 
            methods_list = params['methods_list']
        

        # Get the data for training and testing
        data_train = get_d(data, datasets=datasets_train, names=names_train, activities=act_filter,
                            methods=methods_list)
        data_test = get_d(data, datasets=datasets_test, names=names_test, activities=act_filter,
                            methods=methods_list)

        # Divide the data into inputs and labels
        x_train, y_train = get_inputs_and_labels(data_train,ground_truth=ground_truth)
        x_test, y_test = get_inputs_and_labels(data_test,ground_truth=ground_truth)
        # Swap dimensions to have: [windows, n_samples, methods, landmarks]
        x_train = swap_dims(x_train, (0, samples, n_methods, all_landmarks))  
        y_train = swap_dims(y_train, (0, samples, 1, 1))
        x_test = swap_dims(x_test, (0, samples, n_methods, all_landmarks))  
        y_test = swap_dims(y_test, (0, samples, 1, 1))

        # Take the wanted landmarks
        x_train = x_train[:, :, :, landmarks]
        x_test = x_test[:, :, :, landmarks]

        # Reshape to (videoxwindows,samples,n_methodsxn_landmarks)
        x_train = x_train.view().reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2]*x_train.shape[3])
        y_train = y_train.view().reshape(y_train.shape[0],y_train.shape[1],y_train.shape[2]*y_train.shape[3])
        x_test = x_test.view().reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2]*x_test.shape[3])
        y_test = y_test.view().reshape(y_test.shape[0],y_test.shape[1],y_test.shape[2]*y_test.shape[3])

        data_splits[i]['x_train'] = x_train
        data_splits[i]['x_test'] = x_test
        data_splits[i]['y_train'] = y_train
        data_splits[i]['y_test'] = y_test
        # print the infrmation of the data
        print('Fold',i,':')
        print('[windows, n_samples, methods, landmarks]. (x_train,y_train) || (x_test,y_test) :',x_train.shape,y_train.shape,x_test.shape,y_test.shape)

    return  data_splits

def process(params, data, ground_truth='gt'):
    '''
    Process the data to be used in the model.

    Args:
        params (dict): a dictionary of parameters used in data processing and model training.
        data (dict): a dictionary of the raw data to be processed.
        fps (int): the number of frames per second in the raw data.

    Returns:
        data_splits (list): a list of dictionaries, where each dictionary contains the data splits for one fold of the K-fold cross-validation.
        names_splits (list): a list of dictionaries, where each dictionary contains the video names for the training and test sets for one fold of the K-fold cross-validation.
        params (dict): the input `params` dictionary with some additional parameters added based on the processed data.
    '''

    # -- SET SAMPLES PER WINDOW
    params['samples'] = int(params['fps'] * params['win_secs'])

    # -- NUMBER OF METHODS
    n_methods = len(params['methods_list']) - 1
    params['n_methods'] = n_methods

    # -- CHOOSE LANDMARKS
    landmarks = select_landmarks(params['landmarks'])
    params['n_landmarks'] = len(landmarks)
    all_landmarks = 455

    # -- SPLIT THE DATA INTO TRAINING AND TESTING SETS
    names_splits = get_names(params, data)
    data_splits = get_data(names_splits, params, data, params['samples'], n_methods, all_landmarks, landmarks, ground_truth=ground_truth)

    return data_splits, names_splits, params




def run_model(model,params,x_train,x_test,y_train,y_test,write_dir=True,save_plot=True):
   '''
   Inputs: 
   - params: a dictionary of parameters for the model, including the following keys: 
   * 'datasets': list of datasets being used
   * 'landmarks': landmarks used in the model
   * 'win_secs': window size for the data
   * 'act_filter': activation function used
   * 'loss': loss function used
   * 'norm': normalization used
   * 'lr_scheduler': learning rate scheduler used
   * 'optimizer': optimizer used
   * 'epochs': number of epochs for training
   * 'batch_size': batch size for training
   * 'metrics': list of metrics to track during training
   * 'dropout': dropout rate for the model
   * 'n_units': number of units in the model
   * 'units': type of units in the model
   - x_train: input training data
   - x_test: input test data
   - y_train: output training data
   - y_test: output test data
   - samples: number of samples in the data
   - n_landmarks: number of landmarks used in the data
   - n_methods: number of methods used in the data
   - write_dir (optional): whether to write the model, log and info to directories. Default is True.
   - save_plot (optional): whether to save the plots of the model's training and validation. Default is True.

   Outputs:
   - Trained model and its information, including the model's parameters, training and validation loss and accuracy, and the model's checkpoints and plots (if `write_dir` and `save_plot` are True).

   Description: 
   This function trains a model on the input data and outputs its information to directories. The model's architecture and hyperparameters are specified in the `params` dictionary. The training and validation performance are monitored using the metrics specified in `params`. The model is trained for the number of epochs specified in `params` and with the batch size specified in `params`. The training process also uses early stopping and learning rate scheduling, which are specified in `params`. The model and its information are saved in directories (if `write_dir` is True). The model's training and validation performance is plotted (if `save_plot` is True).
   '''
   
   # -- CREATE DIRS
   model_dir,logdir = create_dirs(params,write_dir)

   # -- HYPERPARAMS
   # Choose metrics
   loss = choose_metric(params['loss'])
   metrics = [choose_metric(metric) for metric in params['metrics']]
   # Choose lr scheduler and optimizer
   lr_scheduler = choose_lr_scheduler(params['lr_scheduler'])
   if params['lr_scheduler'] != 'plateau':
      optimizer = choose_optimizer_with_lr(params['optimizer'], lr_scheduler)
   else:
      optimizer = choose_optimizer(params['optimizer'])
   
   # Choose batch size and epochs
   batch_size = params['batch_size']
   if params['early_stopping']:
      epochs = 1500
   else:
      epochs = params['epochs']

   lr_metric = get_lr_metric(optimizer)
   metrics += [lr_metric]

   # -- CALLBACKS
   callbacks = choose_callbacks(params,write_dir,model_dir,logdir,lr_scheduler)

   # -- MODEL
   model.compile( loss = loss,
                  metrics=metrics,
                  optimizer=optimizer,
                  # run_eagerly=False,
                  # steps_per_execution=4,
                  # jit_compile=True
                  )
   history = model.fit(x_train[:], y_train[:],validation_data=(x_test[:],y_test[:]),
                     epochs=epochs, batch_size=batch_size, verbose=2,shuffle=True,
                     callbacks=callbacks,
                     # workers=1,use_multiprocessing=False,
                     )
   if write_dir:
      plot_save_metrics(history,model_dir,save=save_plot)
   else:
      plot_save_metrics(history,None,save=save_plot)

   return model,history


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
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1,patience=10, min_lr=1e-6,min_delta=0.0001) #factor 0.1
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
