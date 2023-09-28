import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from dtaidistance import dtw
from sklearn.metrics import mean_squared_error

from utils_model.data_helper import *
from utils_model.process_bpm import *

'''
In this python file, we define the functions to evaluate the performance of the methods
'''


''' FUNCTIONS TO EVALUATE THE METHODS'''

def eval_models(data,params,models,data_results,names_splits,datasets=None,split=None,isprint=True,data_bpm=None,fps=None):
    """
    Evaluate the performance of machine learning models on the input data.

    Args:
        data (dict): A dictionary containing the input data.
        params (dict): A dictionary containing the parameters needed for the evaluation.
        models (list): A list containing the machine learning models to evaluate.
        data_results (dict): A dictionary to store the evaluation results.
        names_splits (list): A list containing the train/test split information.
        datasets (list, optional): A list of datasets to evaluate. If None, all datasets in the input data will be evaluated. Defaults to None.
        split (int, optional): The index of the train/test split to evaluate. If None, all splits will be evaluated. Defaults to None.
        isprint (bool, optional): Whether to print progress messages. Defaults to True.
        data_bpm (dict, optional): A dictionary containing the heart rate information. Defaults to None.
        fps (int, optional): The frames per second of the input data. Defaults to None.

    Returns:
        data_results (dict): The updated evaluation results.
        data (dict): The input data with the predicted results.

    """
    if datasets!=None:
        datasets_eval = datasets
    else:
        datasets_eval = params['datasets']

    if split!=None:
        range_splits = [split]
    else:
        range_splits = np.arange(len(names_splits))

    for dataset in datasets_eval:#data.keys(): # Go through each dataset
        print('Dataset ',dataset)
        for name in data[dataset].keys():
            for act in data[dataset][name].keys():
                # print(dataset,name,act)
                gt = np.squeeze(data[dataset][name][act]['gt'])     

                for i in range_splits:
                    # si el dataset not esta en el names del test set, se evalua, Si esta, se cogen los nombres de ahi
                    if (dataset not in names_splits[i]['test'].keys()):
                        evaluate_name = True
                    elif name in names_splits[i]['test'][dataset]:
                        evaluate_name = True
                    else:
                        evaluate_name = False
                    if evaluate_name:
                        if isprint:
                            print(name,act,'. Split ',i)
                        # Choose the landmarks used in the model
                        landmarks_model = select_landmarks(params['landmarks'])
                        # Initialize the methods used in that particular model, with shape (n_methods,n_windows, n_samples)
                        features_model = np.zeros((0,data[dataset][name][act]['gt'].shape[1],data[dataset][name][act]['gt'].shape[2]))
                        # Get all the methods needed for the prediction
                        model_methods = params['methods_list'][1:]
                        for method in model_methods:
                            # Take the method with the landmarks that were used to train the model
                            features_model = np.concatenate((features_model, data[dataset][name][act][method][landmarks_model]) )

                        # Reshape to (n_windows,n_samples,n_methodsxlandmarks), which is the input for the model
                        feat_pred = np.moveaxis(features_model,[0,1,2],[2,0,1]) 
                        pred = models[i].predict(feat_pred,verbose=0)
                        data[dataset][name][act]['Model'] = pred
                        pred = np.squeeze(pred)
                        pred = normalization('min_max',pred,dim=2)
 

                        if (data_bpm is not None) and (dataset!='MR-NIRP'):
                            bpm = np.squeeze(data_bpm[dataset][name][act]['bpm'])
                            results = evaluate_signals(pred,gt,fps,fps,bpm_gt=bpm)
                        else: 
                            results = evaluate_signals(pred,gt,fps,fps)
                        # Average, for every subject, all the windows
                        for metric in results.keys():
                            data_results[dataset][name][act]['Model'][metric] = np.mean(np.array(results[metric])) 

    return data_results,data





def eval_methods(data, data_results, methods=['cpu_GREEN','cpu_RED','cpu_BLUE','cpu_PCA','cpu_ICA','cpu_LGI','cpu_CHROM','cpu_POS', 'cpu_PBV', 'cpu_OMIT','GRGB'], isprint=True, data_bpm=None, fps=None):
    """
    Evaluates the performance of various methods on physiological signals data.

    Args:
        data (dict): Dictionary containing the physiological signals data.
        data_results (dict): Dictionary to store the evaluation results.
        methods (list): List of methods to be evaluated. Default is a list of 10 methods.
        isprint (bool): Flag to print progress messages. Default is True.
        data_bpm (dict): Dictionary containing the BPM data. Default is None.
        fps (float): Frames per second of the physiological signals data. Default is None.

    Returns:
        dict: Dictionary containing the evaluation results.

    """
    # Landmarks 
    landmarks = select_landmarks('combined')
    # Iterate over all datasets, subjects, and activities
    for dataset in ['LGI-PPGI', 'MR-NIRP', 'PURE']:
        if isprint:
            print('Dataset:', dataset)

        for name in data[dataset].keys():
            for act in data[dataset][name].keys():
                if isprint:
                    print('Subject:', name, 'Activity:', act)

                # Initialize all the methods with shape (n_methods, n_windows, n_samples)
                features_all = np.zeros((0, data[dataset][name][act]['gt'].shape[1], data[dataset][name][act]['gt'].shape[2]))

                # Get the ground truth data
                gt = np.squeeze(data[dataset][name][act]['gt'])

                # Evaluate all the methods
                for method in methods:
                    # Append the method to the features array
                    features_all = np.concatenate((features_all, data[dataset][name][act][method][landmarks]))

                    # Take the average of the landmarks of the forehead to do the evaluation of the method
                    avg_land = np.mean(data[dataset][name][act][method][landmarks], axis=0)

                    # Evaluate the signals
                    if (data_bpm is not None) and (dataset != 'MR-NIRP'):
                        bpm = np.squeeze(data_bpm[dataset][name][act]['bpm'])
                        results = evaluate_signals(avg_land, gt, fps, fps, bpm_gt=bpm)
                    else:
                        results = evaluate_signals(avg_land, gt, fps, fps)

                    # Average, for every subject, all the windows
                    for metric in results.keys():
                        data_results[dataset][name][act][method][metric] = np.mean(np.array(results[metric]))

    return data_results





def evaluate_signals(rPPG,cPPG,fps_rPPG,fps_cPPG,print_info=False,bpm_gt=None,plot_windows=False):
    '''
    Calculates the metrics between two signals rPPG and cPPG, and returns them as a dictionary.

    Args:
        rPPG (ndarray): Signal a (rPPG).
        cPPG (ndarray): Signal b (cPPG).
        fps_rPPG (int): Sampling frequency of rPPG.
        fps_cPPG (int): Sampling frequency of cPPG.
        print_info (bool): Whether to print the last value of each metric for each window and the average across windows.
        bpm_gt (ndarray): Ground truth BPM with one value per window.
        plot_windows (bool): Whether to plot each window of the signals.

    Returns:
        A dictionary with the following keys:
        - "DTW": Dynamic Time Warping distance.
        - "r": Pearson correlation coefficient.
        - "RMSE": Root Mean Square Error.
        - "bpm_PPG_fft": Absolute difference between BPM estimated from PPG signal using FFT and ground truth BPM.
        - "bpm_PPG_acorr": Absolute difference between BPM estimated from PPG signal using autocorrelation and ground truth BPM.
        - "bpm_PPG_welch": Absolute difference between BPM estimated from PPG signal using Welch's method and ground truth BPM.
        - "bpm_rPPG_fft": Absolute difference between BPM estimated from rPPG signal using FFT and ground truth BPM.
        - "bpm_rPPG_acorr": Absolute difference between BPM estimated from rPPG signal using autocorrelation and ground truth BPM.
        - "bpm_rPPG_welch": Absolute difference between BPM estimated from rPPG signal using Welch's method and ground truth BPM.
        - "PPG_rPPG_fft": Absolute difference between BPM estimated from PPG signal using FFT and BPM estimated from rPPG signal using FFT.
        - "PPG_rPPG_acorr": Absolute difference between BPM estimated from PPG signal using autocorrelation and BPM estimated from rPPG signal using autocorrelation.
        - "PPG_rPPG_welch": Absolute difference between BPM estimated from PPG signal using Welch's method and BPM estimated from rPPG signal using Welch's method.
    '''
    # calculate metrics for each window
    results = nested_dict(1, list)
    if len(rPPG)!=len(cPPG):
        print('ERROR: signals have different length')
    # For every window calculate the metrics and append them to a lists
    for i in range(len(rPPG)):
        # Calculate the metrics
        DTW_ = dtw.distance(rPPG[i],cPPG[i])
        # DDTW_ = ddtw_distance(rPPG[i],cPPG[i]) 
        results['DTW'].append(DTW_)
        # results['DDTW'].append(DDTW_)
        r = stats.pearsonr(rPPG[i],cPPG[i])[0]
        results['r'].append(np.abs(r))
        RMSE = mean_squared_error(rPPG[i],cPPG[i], squared=False)
        results['RMSE'].append(RMSE)

        # Take the average of the BPM for every window and compare with the average GT BPM for that window
        bpm_PPG_fft ,bpm_PPG_acorr ,bpm_PPG_welch = get_BPM(cPPG[i],fps_cPPG)
        bpm_rPPG_fft,bpm_rPPG_acorr,bpm_rPPG_welch = get_BPM(rPPG[i],fps_rPPG)
        results['PPG_rPPG_fft'].append(  np.abs(bpm_PPG_fft -   bpm_rPPG_fft) )
        results['PPG_rPPG_acorr'].append(np.abs(bpm_PPG_acorr - bpm_rPPG_acorr) )
        results['PPG_rPPG_welch'].append(np.abs(bpm_PPG_welch - bpm_rPPG_welch) )
        
        if bpm_gt is not None:
            bpm_gt_win = np.mean(bpm_gt[i])
            results['bpm_PPG_fft'].append(   np.abs(bpm_PPG_fft -   bpm_gt_win)  ) 
            results['bpm_PPG_acorr'].append( np.abs(bpm_PPG_acorr - bpm_gt_win) )
            results['bpm_PPG_welch'].append( np.abs(bpm_PPG_welch - bpm_gt_win) )
            results['bpm_rPPG_fft'].append(  np.abs(bpm_rPPG_fft -  bpm_gt_win) )
            results['bpm_rPPG_acorr'].append(np.abs(bpm_rPPG_acorr -bpm_gt_win) )
            results['bpm_rPPG_welch'].append(np.abs(bpm_rPPG_welch -bpm_gt_win) )
        else: 
            results['bpm_PPG_fft'].append(   1e-5  ) 
            results['bpm_PPG_acorr'].append( 1e-5 )
            results['bpm_PPG_welch'].append( 1e-5 )
            results['bpm_rPPG_fft'].append(  1e-5 )
            results['bpm_rPPG_acorr'].append(1e-5)
            results['bpm_rPPG_welch'].append(1e-5)

        if plot_windows:
            fig, axs = plt.subplots(2, figsize=(10, 3))
            axs[0].plot(cPPG[i],label='rPPG')
            axs[1].plot(rPPG[i],color='orange',label='GT')
            plt.legend()
            plt.show()

        if print_info:
            for key in results.keys():
                print(key,': ',np.round(results[key][-1],2),end=' ')

    if print_info:
        for key in results.keys():
            print(key,',',end=' ')
        print(': ',end=' ')
    return results



