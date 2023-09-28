import numpy as np
from collections import defaultdict
import random 
from scipy import signal
import os
import seaborn as sns
from collections import defaultdict



'''
In this python file, we define functions to load the rPPG signal datasets and process them.
'''


''' LOAD THE DATASET'''

def load_dataset(path,datasets=['PURE'],methods = ['gt','cpu_GREEN','cpu_RED','cpu_BLUE','cpu_PCA','cpu_ICA','cpu_LGI','cpu_CHROM','cpu_POS', 'cpu_PBV', 'cpu_OMIT','GRGB']):
    # Do a dic in which for each datasets the characteristics are written
    fps = {}
    data = nested_dict(4, list)
    if 'LGI-PPGI' in datasets:
        fps['LGI-PPGI'] = {'gt':60,'rPPG':25}
        names = ['alex','angelo','cpi','david','felix','harun']
        activities = ['gym','resting','rotation','talk']
        root = path+'\\lgi_ppgi_npy_all\\'
        for name in names: 
            for act in activities:
                for method in methods:
                    try:
                        sig_rPPG = np.load(root+ method+'_'+name+'_'+act+'.npy')
                        if method=='gt':
                            data['LGI-PPGI'][name][act][method] = np.expand_dims(sig_rPPG,0)
                        else:
                            data['LGI-PPGI'][name][act][method] = sig_rPPG
                    except:
                        # print('Could not load ',method,' ',name,' ',act, '. Either gt or the rppg are missing')
                        pass
    if 'PURE' in datasets:
        fps['PURE'] = {'gt':60,'rPPG':30}
        names = ['01','02','03','04','05','06','08','09','10']
        activities = ['01','02','03','04','05','06']
        root = path + '\\PURE_npy_all\\'
        for name in names: 
            for act in activities:
                for method in methods:
                    try:
                        sig_rPPG = np.load(root+ method+'_'+name+'-'+act+'.npy')
                        if method=='gt':
                            data['PURE'][name][act][method] = np.expand_dims(sig_rPPG,0)
                        else:
                            data['PURE'][name][act][method] = sig_rPPG
                    except:
                        # print('Could not load ',method,' ',name,' ',act, '. Either gt or the rppg are missing')
                        pass
    if 'MR-NIRP' in datasets:
        fps['MR-NIRP'] = {'gt':60,'rPPG':30}
        names = ['Subject1','Subject2','Subject3','Subject4','Subject5','Subject6','Subject7','Subject8']
        activities = ['still','motion']
        root =  path+'\\MR-NIRP_indoor_npy_all\\'
        
        for name in names: 
            for act in activities:
                for method in methods:
                    try:
                        files = [ f.path for f in os.scandir(root) ]
                        dir = ''
                        for file in files:
                            if method+'_'+name+'_'+act in file:
                                dir = file
                        sig_rPPG = np.load(dir)
                        if method=='gt':
                            data['MR-NIRP'][name][act][method] = np.expand_dims(sig_rPPG,0)
                            length = data['MR-NIRP'][name][act]['gt'].shape[1]
                        else:
                            data['MR-NIRP'][name][act][method] = sig_rPPG
                            # Take only until len(gt)/2, because fps(gt)=60, and fps(rppg)=30
                            data['MR-NIRP'][name][act][method] = data['MR-NIRP'][name][act][method][:,:int(length/2)]
                    except:
                        # print('Could not load ',method,' ',name,' ',act, '. Either gt or the rppg are missing')
                        pass
                # data['MR-NIRP'][name][act]['gt'] = scipy.signal.resample(data['MR-NIRP'][name][act]['gt'],int(length/2),axis=1)

    print('loaded dataset ',datasets)
    return data,fps




''' Functions for normalization'''

def normalization(norm,data,dim=3):
    """
    Normalizes ND data along the last axis.
    
    Args:
        norm (str): Normalization type, one of {'min_max', 'robust', 'mean_normalization'}.
        data (numpy.ndarray): Input data to normalize.
        dim (int): The axis to normalize. Default is 3.
    
    Returns:
        numpy.ndarray: The normalized data.
    """
    if norm == 'min_max':
        data = (data-np.min(data,axis=dim-1,keepdims=True))/(np.max(data,axis=dim-1,keepdims=True)-np.min(data,axis=dim-1,keepdims=True))
    elif norm == 'robust':
        data = (data-np.mean(data,axis=dim-1,keepdims=True))/(np.percentile(data,75,axis=dim-1,keepdims=True)-np.percentile(data,25,axis=dim-1,keepdims=True))
    elif norm == 'mean_normalization':
        data = (data-np.mean(data,axis=dim-1,keepdims=True))/(np.max(data,axis=dim-1,keepdims=True)-np.min(data,axis=dim-1,keepdims=True))
    return data



def norm(data,mode='robust'):
    """
    Normalizes all 2D data in a nested dictionary using a specified normalization method.
    
    Args:
        data (dict): Nested dictionary containing 2D data to normalize.
        mode (str): Normalization type, one of {'min_max', 'robust', 'mean_normalization'}.
    
    Returns:
        dict: The normalized data.
    """
    for dataset in data.keys():
        for name in data[dataset].keys():
            for act in data[dataset][name].keys():
                for method in data[dataset][name][act].keys():
                    data[dataset][name][act][method] = normalization(mode,data[dataset][name][act][method])
    return data



def norm_windows(data,mode='min_max',dim=3):
    """
    Normalizes all 2D data in a nested dictionary along a specified axis using a specified normalization method.
    
    Args:
        data (dict): Nested dictionary containing 2D data to normalize.
        mode (str): Normalization type, one of {'min_max', 'robust', 'mean_normalization'}.
        dim (int): The axis to normalize. Default is 3.
    
    Returns:
        dict: The normalized data.
    """
    for dataset in data.keys():
        for name in data[dataset].keys():
            for act in data[dataset][name].keys():
                for method in data[dataset][name][act].keys():
                    data[dataset][name][act][method] = normalization(mode,data[dataset][name][act][method],dim=dim)
    return data

                    

''' Functions for resampling'''


def resample_gt(data): 
    """
    Resample the ground truth signals to have the same number of samples as the RPPG signals.
    
    Args:
        data (dict): A dictionary containing the data to be resampled.
        
    Returns:
        dict: A dictionary containing the resampled data.
    """
    for dataset in data.keys():
        for name in data[dataset].keys():
            for act in data[dataset][name].keys():
                data[dataset][name][act]['gt'] = signal.resample(data[dataset][name][act]['gt'], data[dataset][name][act]['cpu_GREEN'].shape[1],axis=1)
    return data


def resample_methods(data): 
    """
    Resample the RPPG signals to have the same number of samples as the ground truth signals.
    
    Args:
        data (dict): A dictionary containing the data to be resampled.
        
    Returns:
        dict: A dictionary containing the resampled data.
    """
    for dataset in data.keys():
        for name in data[dataset].keys():
            for act in data[dataset][name].keys():
                for method in data[dataset][name][act].keys():
                    if method != 'gt':
                        data[dataset][name][act][method] = signal.resample(data[dataset][name][act][method], data[dataset][name][act]['gt'].shape[1],axis=1)
    return data


def resample_windows(data,seconds,fps):
    """
    Resample the data to have a specified sampling rate and time window.
    
    Args:
        data (dict): A dictionary containing the data to be resampled.
        seconds (float): The time window duration in seconds.
        fps (float): The desired sampling rate (frames per second).
        
    Returns:
        dict: A dictionary containing the resampled data.
    """
    for dataset in data.keys():
        for name in data[dataset].keys():
            for act in data[dataset][name].keys():
                for method in data[dataset][name][act].keys():
                    data[dataset][name][act][method] = signal.resample(data[dataset][name][act][method],int(fps*seconds) ,axis=2)
    return data


def resample_fps(data,fps): 
    """
    Resample the data to have a specified sampling rate.
    
    Args:
        data (dict): A dictionary containing the data to be resampled.
        fps (float): The desired sampling rate (frames per second).
        
    Returns:
        dict: A dictionary containing the resampled data.
    """
    for dataset in data.keys():
        for name in data[dataset].keys():
            for act in data[dataset][name].keys():
                data[dataset][name][act]['gt'] = signal.resample(data[dataset][name][act]['gt'], data[dataset][name][act]['cpu_GREEN'].shape[1],axis=1)
    return data




''' SPLIT WINDOWS '''

def split_array(array, window, overlap=0):
    '''
    Splits an input array into chunks of a specified window size and overlap.

    Parameters:
        array (ndarray): The input array to be split.
        window (int): The size of the chunks to be created.
        overlap (int): The amount of overlap between consecutive chunks. Defaults to 0.

    Returns:
        chunks (list of ndarrays): The list of created chunks.
    '''
    chunks = []
    step = window - overlap
    for i in range(0, len(array) - window + 1, step):
        slice = array[i:i+window]
        if len(slice) == window:
            chunks.append(slice)
    return chunks

def split_dataset_windows(data,win_secs,fps,overlap=0):
    '''
    Splits the input dataset into windows of a specified length and overlap, using the given fps.

    Parameters:
        data (dict): The input dataset to be split.
        win_secs (float): The size of the window in seconds.
        fps (dict): A dictionary containing the fps for each dataset.

    Returns:
        data (dict): The updated dataset with windowed data.
    '''
    for dataset in data.keys():
        for name in data[dataset].keys():
            for act in data[dataset][name].keys():
                for method in data[dataset][name][act].keys():
                    windows = []
                    for i in range(data[dataset][name][act][method].shape[0]):
                        # split each row of data into windows
                        # if the dataset is not LGI-PPGI, and the act is not 'gym', then use the overlap, otherwise don't (in gym there are too many samples)
                        if dataset != 'LGI-PPGI' and act != 'gym':
                            ar = split_array(data[dataset][name][act][method][i,:], int(win_secs*fps[dataset]['rPPG']), overlap=overlap)
                        else:
                            ar = split_array(data[dataset][name][act][method][i,:], int(win_secs*fps[dataset]['rPPG']), overlap=0)
                        windows.append(ar)
                    data[dataset][name][act][method] = np.array(windows)
    # DATA SHAPE: dict(dataset,name,act,method) arr(landmark,n_windows,data), len(data)=100 for windows of 4 secs and fps 25
    return data

def split_dataset_windows_all_same_fps(data,win_secs,fps):
    '''
    Splits the input dataset into windows of a specified length and overlap, assuming all datasets have the same fps.

    Parameters:
        data (dict): The input dataset to be split.
        win_secs (float): The size of the window in seconds.
        fps (int): The fps to use for all datasets.

    Returns:
        data (dict): The updated dataset with windowed data.
    '''
    for dataset in data.keys():
        for name in data[dataset].keys():
            for act in data[dataset][name].keys():
                for method in data[dataset][name][act].keys():
                    windows = []
                    for i in range(data[dataset][name][act][method].shape[0]):
                        # split each row of data into windows
                        ar = split_array(data[dataset][name][act][method][i,:], int(win_secs*fps))
                        windows.append(ar)
                    data[dataset][name][act][method] = np.array(windows)
    # DATA SHAPE: dict(dataset,name,act,method) arr(landmark,n_windows,data), len(data)=100 for windows of 4 secs and fps 25
    return data




''' UTIL FUNCTIONS '''

def avg_cell(x):
    """
    Calculates the average value of a 2D numpy array.

    Parameters:
    x (numpy.ndarray): A 2D numpy array.

    Returns:
    float: The average value of the array.
    """
    return np.mean(x)

def nested_dict(n, type):
    """
    Creates a nested defaultdict with n levels.

    Parameters:
    n (int): The number of nested levels for the defaultdict.
    type: The default data type for the defaultdict.

    Returns:
    defaultdict: A nested defaultdict with n levels.
    """
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))

def select_landmarks(landmarks_type):
    """
    Returns a list of landmarks based on the landmarks_type parameter.

    Parameters:
    landmarks_type (str): The type of landmarks to be selected.

    Returns:
    list: A list of landmark IDs.
    """
    landmarks_forehead = [107, 66, 69, 109, 10, 338, 299, 296, 336, 9]
    landmarks_leftcheek = [118, 119, 100, 126, 209, 49, 129, 203, 205, 50]
    landmarks_rightcheek = [347, 348, 329, 355, 429, 279, 358, 423, 425, 280]
    high_prio_forehead = [10, 67, 69, 104, 108, 109, 151, 299, 337, 338]
    high_prio_nose = [3, 4, 5, 6, 45, 51, 115, 122, 131, 134, 142, 174, 195, 196, 197, 198,
                      209, 217, 220, 236, 248, 275, 277, 281, 360, 363, 399, 419, 420, 429, 437, 440]
    high_prio_left_cheek = [36, 47, 50, 100, 101, 116, 117,
                            118, 119, 123, 126, 147, 187, 203, 205, 206, 207, 216]
    high_prio_right_cheek = [266, 280, 329, 330, 346, 347,
                             347, 348, 355, 371, 411, 423, 425, 426, 427, 436]
    mid_prio_forehead = [8, 9, 21, 68, 103, 251,
                         284, 297, 298, 301, 332, 333, 372, 383]
    mid_prio_nose = [1, 44, 49, 114, 120, 121, 128, 168, 188, 351, 358, 412]
    mid_prio_left_cheek = [34, 111, 137, 156, 177, 192, 213, 227, 234]
    mid_prio_right_cheek = [340, 345, 352, 361, 454]
    landmarks = []
    if landmarks_type == 'forehead':
        landmarks = landmarks_forehead
    elif landmarks_type == 'combined':
        landmarks = landmarks_forehead+landmarks_leftcheek+landmarks_rightcheek
    elif landmarks_type == 'all':
        landmarks = np.arange(455)
    elif landmarks_type == 'prio_all':
        landmarks = high_prio_forehead + high_prio_nose + high_prio_left_cheek + high_prio_right_cheek + \
            mid_prio_forehead + mid_prio_nose + mid_prio_left_cheek + mid_prio_right_cheek
    else: 
        print('Name of landmarks not identified')
    return landmarks 


''' CLEAN DATA NEW '''

from skimage import exposure
def hist_equalize(data_eval):
    for dataset in data_eval.keys():
        for subject in data_eval[dataset].keys():
            for activity in data_eval[dataset][subject].keys():
                for method in data_eval[dataset][subject][activity].keys():
                    for i,landmark in enumerate(data_eval[dataset][subject][activity][method]):
                        # Apply histogram equalization, for every landmark, to the windows,frames as a 2d image
                        data_eval[dataset][subject][activity][method][i] = exposure.equalize_hist(landmark)
    return data_eval


def clean_signals(signals):
    '''
    Removes outliers from the signals. 
    First, it removes the landmarks with variance and mean in the upper 5% of the distribution, and the nans.
        If there are no landmarks that are not: nan, mean>95% or variance>95%, then:
        Second, if there are non nans, replace the nans with the mean of all the non nan landmarks.
            If there are no non nans, then:
            Third, Leave the signals as they are. 
    '''

    var_big = np.argwhere(np.var(signals,axis=1)>np.percentile(np.var(signals,axis=1),95)).flatten()
    var_small = np.argwhere(np.var(signals,axis=1)<1e-6).flatten()
    mean_big = np.argwhere(np.mean(signals,axis=1)>np.percentile(np.mean(signals,axis=1),95)).flatten()
    nan_landmarks = np.argwhere(np.isnan(np.mean(signals,axis=1))).flatten()
    outliers = np.unique(np.concatenate((var_big,var_small,mean_big,nan_landmarks)))

    # Find the landmarks with variance and mean in the lower 30% of the distribution
    common = np.argwhere((np.var(signals,axis=1)<np.percentile(np.var(signals,axis=1),30)) & (np.abs(np.mean(signals,axis=1))<np.percentile(np.abs(np.mean(signals,axis=1)),30))).flatten()
    if len(common)!=0:
        # Substitute the outliers landmarks by the averaged landmarks with variance in the lower 30% of the distribution and mean
        signals[outliers,:] = np.repeat(np.mean(signals[common,:],axis=0)[np.newaxis,:],outliers.shape[0],axis=0)
    else: 
        # print('no landmarks with good variance and mean')
        # Get the number of landmarks that are not nan
        n_non_nan = np.sum(~np.isnan(np.mean(signals,axis=1)))
        if n_non_nan>0:
            # Substitute the outliers landmarks by the averaged landmarks that are not nan
            signals[nan_landmarks,:] = np.repeat(np.mean(signals[~np.isnan(np.mean(signals,axis=1)),:],axis=0)[np.newaxis,:],nan_landmarks.shape[0],axis=0)
        else:
            print('no landmarks with no nan values')

    return signals


def clean_landmarks(data):
    for dataset in data.keys():
        for subject in data[dataset].keys():
            for activity in data[dataset][subject].keys():
                for method in data[dataset][subject][activity].keys():
                    if method != 'gt':
                        data[dataset][subject][activity][method] = clean_signals(data[dataset][subject][activity][method])
    return data

def clean_windows(data):
    ''' 
    Array with shape (landmarks, number of windows, frames per window)
    Does the same as clean_landmarks but for the windows. If there are no non nans, then the difference it substitutes the nans with 0s
    '''
    # Swap the dimensions to have (number of windows, landmarks, frames per window)
    data = np.swapaxes(data,0,1)

    for i,window in enumerate(data):
        var_small =  np.argwhere(np.var(window,axis=1)<0.005)
        mean_big =  np.argwhere(np.mean(window,axis=1)>0.95)
        nan_windows =  np.argwhere(np.isnan(np.mean(window,axis=1)))
        outliers = np.concatenate((var_small,mean_big,nan_windows))
        outliers = np.unique(outliers)

        # Now we have the indexes of the windows to substitute. Substitute the windows by the mean of the windows with variance lower than the threshold
        if len(outliers)>0:
            # print('outliers',outliers.shape, outliers)
            # Create an array with the (windows,landmarks) whose indexes that are not in outliers
            common = np.argwhere((np.var(window,axis=1)>0.005) & (np.abs(np.mean(window,axis=1))<0.95) & (np.isnan(np.mean(window,axis=1))==False))
            common = common.flatten()
            # print('common',common.shape)
            # print('-'*50)
            if len(common)!=0:
                # Substitute the outliers windows by the mean of the windows with variance lower than the threshold
                data[i,outliers,:] = np.repeat(np.mean(window[common,:],axis=0)[np.newaxis,:],outliers.shape[0],axis=0) 
            else: 
                print('no windows with good variance,mean and no nan values')
                # Get the number of windows that are non nan
                non_nan = np.argwhere(np.isnan(np.mean(window,axis=1))==False)
                if len(non_nan)!=0:
                    # Substitute the windows with nan values by the mean of the non nan windows
                    data[i,nan_windows,:] = np.repeat(np.mean(window[non_nan,:],axis=0)[np.newaxis,:],nan_windows.shape[0],axis=0)
                else: 
                    print('no windows with no nan values')
                    # Substitute the windows with nan values by 0
                    data[i,nan_windows,:] = np.repeat(np.zeros((1,window.shape[1])),nan_windows.shape[0],axis=0)
    # Swap the dimensions to have (landmarks, number of windows, frames per window)
    data = np.swapaxes(data,0,1)
    return data

def clean_windowed_dataset(data):
    for dataset in data.keys():
        for subject in data[dataset].keys():
            for activity in data[dataset][subject].keys():
                for method in data[dataset][subject][activity].keys():
                    if method != 'gt':
                        data[dataset][subject][activity][method] = clean_windows(data[dataset][subject][activity][method])
                    
    return data



''' CLEAN DATA OLD'''


def clean_data(data,print_info=False):
    '''
    Cleans the input data by filling in missing values and removing invalid values.

    Args:
    - data: a dictionary containing the input data
    - print_info: a boolean indicating whether to print information about invalid data (default: False)

    Returns:
    - the cleaned data as a dictionary
    '''
    for dataset in data.keys():
        for name in data[dataset].keys():
            for act in data[dataset][name].keys():
                data[dataset][name][act],invalid_methods,invalid_landmarks = fill_nans(data[dataset][name][act],show=False,plot=False)
                if len(invalid_methods)>0 and print_info:
                    unique_inv_land = get_invalid_landmarks(invalid_landmarks)
                    print('Invalid data in: ',dataset,name,act,'\nMethods:',invalid_methods,'N_landmarks: ',len(unique_inv_land),'\n')
    return data

def get_invalid_landmarks(invalid_landmarks):
    '''
    Given a dictionary with methods as keys and landmarks as values,
    returns the unique values of the landmarks.

    Args:
    - invalid_landmarks: a dictionary with methods as keys and
                         landmarks as values

    Returns:
    - a numpy array of the unique landmarks in invalid_landmarks
    '''
    all_invalid_land = []
    for method in invalid_landmarks.keys():
        all_invalid_land.extend(list(invalid_landmarks[method]))
    all_invalid_land = np.array(all_invalid_land)
    return np.unique(all_invalid_land)

def fill_nans(video_methods,show=False,plot=False):
    '''
    This function receives a dictionary with the data of different methods for a certain video and fills the nans of the methods that have them.
    If all the methods have nans, it replaces the landmarks of the method with other landmarks.

    Args:
    - video_methods: dictionary where the keys are strings with the name of the method and the values are numpy arrays with the data of the method
    - show: boolean indicating if additional information will be printed
    - plot: boolean indicating if a plot will be generated with the results

    Returns:
    - video_methods: updated dictionary with the same structure as the input but with nans filled
    - nan_methods: list with the names of the methods that had nans
    - nan_landmarks: nested dictionary where the keys are the names of the methods that had nans, and the values are lists with the landmarks 
                    responsible for the nans
    '''

    no_nan_methods,nan_methods,nan_landmarks = not_valid_methods_landmarks(video_methods,show=show)
    if len(nan_methods)!=0:
        video_methods = replace_with_other_landmarks(video_methods,nan_methods,nan_landmarks)
    return video_methods, nan_methods,nan_landmarks


def replace_with_other_landmarks(video_methods,nan_methods,nan_landmarks):
    '''
    For each method with nans, replaces the nan landmarks with the average of the landmarks from non-nan methods.

    Args:
    - video_methods: a dictionary with methods as keys and numpy arrays of landmarks as values
    - nan_methods: a list of methods that contain nans
    - nan_landmarks: a dictionary with methods as keys and a list of the landmark indices that contain nans as values

    Returns:
    - video_methods: the updated dictionary with methods as keys and numpy arrays of landmarks as values
    '''
    for nan_method in nan_methods:
        invalid_landmarks = nan_landmarks[nan_method]
        # Select landmarks that are not in nan_landmarks
        all_landmarks = list(np.arange(1,455))
        all_valid_landmarks = list(set(list(all_landmarks)) - set(list(invalid_landmarks)))
        # print('len all landmarks, and inv landmarks:',len(all_landmarks),len(invalid_landmarks))
        valid_landmarks = random.choices(all_valid_landmarks,k=len(invalid_landmarks))
        valid_landmarks = np.array(valid_landmarks)
        video_methods[nan_method][invalid_landmarks,:] = video_methods[nan_method][valid_landmarks,:]
    return video_methods


def replace_with_other_methods(video_methods,no_nan_methods,nan_methods,nan_landmarks,plot=False):    
    '''
    For each method with nans, replaces the nan landmarks with the average of the landmarks from non-nan methods.

    Args:
    - video_methods: a dictionary with methods as keys and numpy arrays of landmarks as values
    - no_nan_methods: a list of methods that do not contain nans
    - nan_methods: a list of methods that contain nans
    - nan_landmarks: a dictionary with methods as keys and a list of the landmark indices that contain nans as values
    - plot: a boolean indicating whether to plot the replacement process 
    '''
    # For each method with nans, replace the nan landmarks with the average  of the landmarks from non-nan methods
    for nan_method in nan_methods:
        land = nan_landmarks[nan_method]
        methods_avg = np.empty((0,) + video_methods[nan_method][land,:].shape) #Create array to append all the methods with no nans and later average
        for no_nan_method in no_nan_methods:
            methods_avg = np.append(methods_avg, np.expand_dims(video_methods[no_nan_method][land,:],axis=0),axis=0)

        video_methods[nan_method][land,:] = np.mean(methods_avg,axis=0)
    return video_methods

def not_valid_methods_landmarks(video_methods,show=False):
    '''
    This function receives a dictionary with the data of different methods for a certain video and returns the methods 
    that have nans, the landmarks responsible for the nans, and the variance threshold used to detect invalid landmarks.

    Args:
    - video_methods: dictionary where the keys are strings with the name of the method and the values are numpy arrays with the data of the method
    - show: boolean indicating if additional information will be printed

    Returns:
    - no_nan_methods: list with the names of the methods that don't have nans
    - nan_methods: list with the names of the methods that have nans
    - nan_landmarks: nested dictionary where the keys are the names of the methods that have nans, and the values are lists with the landmarks 
                    responsible for the nans
    '''
    rPPG_methods = ['cpu_GREEN','cpu_RED','cpu_BLUE','cpu_PCA','cpu_ICA','cpu_LGI','cpu_CHROM','cpu_POS', 'cpu_PBV', 'cpu_OMIT','GRGB']
    var_threshold = {'cpu_GREEN':0.01,'cpu_RED':0.01,'cpu_BLUE':0.01,'cpu_PCA':0.01,'cpu_ICA':0.01,'cpu_LGI':0.01,
                        'cpu_CHROM':0.01,'cpu_POS':0.0001, 'cpu_PBV':1e-6, 'cpu_OMIT':0.01,'GRGB':0.01}

    # For a certain video, Create a list with the methods that dont have nans, and a dictionary to store the landmarks responsible for the nans
    nan_methods = []
    nan_landmarks = nested_dict(1, list)

    # For every method, if there are nans, append that method to the nan_methods list, and take the landmarks with those nans
    for method in rPPG_methods: 
        var_thres = var_threshold[method]
        ind_mean = np.where(np.mean(video_methods[method],axis=1)>1)[0]
        ind_var = np.where((np.var(video_methods[method],axis=1)<0.000001) | (np.var(video_methods[method],axis=1)>1e3)  | (np.isnan(np.var(video_methods[method],axis=1))))[0]
        # ind_var = np.where((np.var(video_methods[method],axis=1)<var_thres) | (np.var(video_methods[method],axis=1)>1e3)  | (np.isnan(np.var(video_methods[method],axis=1))))[0]
        ind_nans = np.argwhere(np.isnan(video_methods[method]))[:,0] # [:,0] is [N_points, [landmark,point value]]
        nan_landmarks[method] = np.unique(np.concatenate((ind_nans,ind_var,ind_mean)) )
        if len(nan_landmarks[method])>0:
            nan_methods.append(method)
            # print('method:',method,' land:',nan_landmarks[method])
            # print('Len ind, Var of those: \n',method,len(nan_landmarks[method]),np.var(video_methods[method][nan_landmarks[method]],axis=1))
            # print('Len ind, mean of those: \n',method,len(nan_landmarks[method]),np.mean(video_methods[method][nan_landmarks[method]],axis=1))
            # print('len var',len(ind_var))
            # print('len ind_mean',len(ind_mean))
            # print('len ind_nans',len(ind_nans))

    # Select the methods that don't have nans
    no_nan_methods = [method for method in rPPG_methods if method not in nan_methods]
    
    return no_nan_methods,nan_methods,nan_landmarks
    

    

''' DATASET PREPARATION'''


def get_d( data,datasets=['LGI-PPGI','PURE','MR-NIRP'],
                methods = ['bpm','GRGR','gt','cpu_GREEN','cpu_RED','cpu_BLUE','cpu_PCA','cpu_ICA','cpu_LGI','cpu_CHROM','cpu_POS', 'cpu_PBV', 'cpu_OMIT'],
                names = {'LGI-PPGI':['alex','angelo','cpi','david','felix','harun'], 'PURE':['01','02','03','04','05','06','07','08','09','10']},
                activities={'LGI-PPGI':['resting','talk','gym','rotation'],'PURE':['01','02','03','04','05','06'],'MR-NIRP':['still','motion'] } ):

    ''' Given a dictionary with all the data, go through all keys. Within those, find the ones that are specified to be returned '''
    results = nested_dict(4, list)

    for dataset in datasets:
        if dataset in list(data.keys()):
            
            for name in names[dataset]:
                if name in list(data[dataset].keys()):

                    for act in activities[dataset]:
                        if act in list(data[dataset][name].keys()):

                            for method in methods:
                                if method in list(data[dataset][name][act].keys()):
                                    results[dataset][name][act][method] = data[dataset][name][act][method]
    return results
def get_inputs_and_labels(data,ground_truth='gt'):
    ''' Given a dictionary with data, return the specified datasets, metrics and so on as a: list-->(#videos,#methods) array-->[#landm,#windows/video,#samples] '''
    x = []
    y = []
    for dataset in data.keys():
        for name in data[dataset].keys():
            for act in data[dataset][name].keys():
                # If an activity is missing, or one method is missing, skip it
                try: 
                    rPPG_methods = []
                    gt = []
                    for method in data[dataset][name][act].keys():
                        # For gt I set an equal because all for every activity there's only one gt, but 7 methods (green,red,chrom...)
                        if method==ground_truth:
                            gt = [data[dataset][name][act][method]]
                        else: 
                            rPPG_methods.append(data[dataset][name][act][method])
                    x.append(rPPG_methods) 
                    y.append(gt)  
                except: 
                    print('missing ',dataset,name,act)    
                    continue              
    return x,y
    
def swap_dims(array,dims):
    ''' Input dims: 
            array: list (#videos,#methods) array[#landm,#windows,#samples]. 
            dims: tuple (0,samples,n_methods,n_landmarks)
        output dims: (videoxwindows, landmarks,methods,smamples)'''
    array_out = np.empty(dims)  # [video] (methods,landmarks,windows,samples)
    for i,video in enumerate(array):    
        array_i = np.moveaxis(np.array(video),[0,1,2,3],[2,3,0,1]) # (videoxwindows,samples,n_methods,n_landmarks)
        array_out = np.append(array_out,array_i,axis=0)
    return array_out

