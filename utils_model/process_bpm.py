import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.fft import rfft, rfftfreq
from scipy import signal
import json
import xml.etree.ElementTree as ET
import statsmodels.api as sm

from utils_model.data_helper import *

'''
In this python file, we define functions to process the BPM data.
'''

''' FUNCTIONS TO PROCESS BPM DATA '''

def split_bpm_windows(data: dict, win_secs: float, fps: dict, overlap = 0) -> dict:
    """
    Splits a dataset of rPPG signals into windows of a specified length and returns the result.

    Args:
        data (dict): A dictionary containing the rPPG signal dataset to split.
        win_secs (float): The window length in seconds.
        fps (dict): A dictionary containing the frame rate per second (fps) for each dataset.

    Returns:
        dict: A dictionary containing the split rPPG signal dataset, where each signal is divided into windows
        of the specified length.

    Example Usage:
        >>> data = {'dataset1': {'name1': {'act1': {'method1': np.random.rand(1000, 3)}}}}
        >>> fps = {'dataset1': {'rPPG': 25}}
        >>> split_bpm_windows(data, 4, fps)
    """
    
    # Iterate through each level of the nested dictionary to split each rPPG signal dataset into windows
    for dataset in data.keys():
        for name in data[dataset].keys():
            for act in data[dataset][name].keys():
                for method in data[dataset][name][act].keys():
                    windows = []                  
                    # # Split the rPPG signal dataset into windows using the specified window length and fps
                    # if the dataset is not LGI-PPGI, and the act is not 'gym', then use the overlap, otherwise don't (in gym there are too many samples)
                    if dataset != 'LGI-PPGI' and act != 'gym':
                        ar = split_array(data[dataset][name][act][method], int(win_secs * fps[dataset]['rPPG']),overlap=overlap)
                    else:
                        ar = split_array(data[dataset][name][act][method], int(win_secs * fps[dataset]['rPPG']),overlap=0)
                    windows.append(ar)
                    
                    # Convert the list of windows into a numpy array and update the dataset with the split rPPG signal
                    data[dataset][name][act][method] = np.array(windows)
    
    # Return the split rPPG signal dataset
    return data



def resample_bpm(data_bpm, data):
    """
    Resamples the BPM signal to have the same number of samples as the corresponding video data.
    
    Args:
    - data_bpm (dict): A dictionary containing the BPM signal data for each dataset, name, and activity.
                       The data should be in the format of {'dataset': {'name': {'act': {'bpm': data}}}}.
    - data (dict): A dictionary containing the video data for each dataset, name, and activity.
                   The data should be in the format of {'dataset': {'name': {'act': {'cpu_GREEN': data}}}}.
    
    Returns:
    - data_bpm (dict): The resampled BPM signal data in the same format as the input data_bpm.
    """
    for dataset in data_bpm.keys():
        for name in data_bpm[dataset].keys():
            for act in data_bpm[dataset][name].keys():
                # Resample the BPM signal to have the same number of samples as the corresponding video data
                data_bpm[dataset][name][act]['bpm'] = signal.resample(data_bpm[dataset][name][act]['bpm'],
                                                                       data[dataset][name][act]['cpu_GREEN'].shape[1])
    return data_bpm



def load_bpm(data_eval):
    """
    Load the ground truth heart rate data for each subject in the evaluation dataset.
    
    Args:
        data_eval (dict): The evaluation dataset.
        
    Returns:
        dict: A nested dictionary containing the ground truth heart rate data for each subject.
    """
    data_bpm = nested_dict(6, list)
    root_pure = '.\\pyVHR\\datasets\\PURE'
    root_lgi = '.\\pyVHR\\datasets\\lgi_ppgi'
    name_file_lgi = 'cms50_stream_handler.xml'

    # Load BPM data for PURE dataset
    for dataset in ['PURE']:
        for name in data_eval[dataset]:
            for act in data_eval[dataset][name]:
                path = root_pure+'\\'+name+'-'+act+'\\'+name+'-'+act+'.json'
                Gt = {"ppg": np.array([]), "bpm": np.array([])}
                data = json.load(open(path))
                data = data["/FullPackage"]
                for row in data:
                    Gt["ppg"] = np.append(Gt["ppg"], float(row["Value"]["waveform"]))
                    Gt["bpm"] = np.append(Gt["bpm"], float(row["Value"]["pulseRate"]))
                data_bpm[dataset][name][act]['bpm'] = Gt["bpm"]
    
    # Load BPM data for LGI-PPGI dataset
    for dataset in ['LGI-PPGI']:
        for name in data_eval[dataset]:
            for act in data_eval[dataset][name]:
                path_lgi= root_lgi+'\\'+name+'\\'+name+'_'+act+'\\'+name_file_lgi
                Gt = {"ppg": np.array([]), "bpm": np.array([]), "info": {} }
                tree = ET.parse(path_lgi)           
                root = tree.getroot()
                for record in root:
                    if len(record) > 2:
                        Gt["ppg"] = np.append(Gt["ppg"], float(record[2].text))
                        Gt["bpm"] = np.append(Gt["bpm"], float(record[1].text))
                    elif len(record) <= 2:
                        print('! data value from BPM gt missing')
                        Gt["ppg"] = np.append(Gt["ppg"], float(record[2].text))
                        Gt["bpm"] = np.append(Gt["bpm"], float(record[1].text))
                data_bpm[dataset][name][act]['bpm'] = Gt["bpm"]  
    return data_bpm

def get_BPM(sig, fps, plot=False):
    """Calculates the heart rate in beats per minute (BPM) using different methods.

    Args:
        sig (array): The input signal.
        fps (int): The sampling rate of the signal, in Hz.
        plot (bool): Whether to plot the power spectrum or not.

    Returns:
        A tuple with the heart rate in BPM, calculated using three methods:
        - BPM calculated with rFFT
        - BPM calculated with the autocorrelation of the signal
        - BPM calculated with Welch's method of the signal

    """

    # Set figure size for plots
    plt.rcParams["figure.figsize"] = (7,2)

    # Compute frequency band of interest
    SAMPLE_RATE = fps
    N = len(sig)
    F = rfftfreq(N, 1 / SAMPLE_RATE)
    band = np.argwhere((F > .65) & (F < 4)).flatten()
    Pfreqs = 60 * F[band]  # Convert frequencies to BPM

    # Calculate the BPM with rFFT
    P = rfft(sig)
    Power = np.abs(P[band])
    Pmax = np.argmax(Power)
    bpm_fft = Pfreqs[Pmax]

    # Plot power spectrum, if requested
    if plot:
        plt.plot(Pfreqs, np.abs(Power), linewidth=1)
        plt.title('Power Spectrum')
        plt.xlabel('BPM')
        plt.ylabel('Power')
        plt.show()

    # Calculate the BPM with the autocorrelation of the signal
    acorr = sm.tsa.acf(sig, nlags=len(sig))
    P = rfft(acorr)
    Power = np.abs(P[band])
    Pmax = np.argmax(Power)
    bpm_acorr = Pfreqs[Pmax]

    # Calculate the BPM with Welch's method of the signal
    bvps = np.expand_dims(sig, 0)
    Pfreqs, Power = Welch(bvps, fps, 0.65, 4, 2048)
    Pmax = np.argmax(Power, axis=1)
    bpm_welch = Pfreqs[Pmax.squeeze()]

    return bpm_fft, bpm_acorr, bpm_welch



def Welch(bvps, fps, minHz=0.65, maxHz=4.0, nfft=2048):
    """
    This function computes Welch'method for spectral density estimation.

    Args:
        bvps(flaot32 numpy.ndarray): BVP signal as float32 Numpy.ndarray with shape [num_estimators, num_frames].
        fps (float): frames per seconds.
        minHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        nfft (int): number of DFT points, specified as a positive integer.
    Returns:
        Sample frequencies as float32 numpy.ndarray, and Power spectral density or power spectrum as float32 numpy.ndarray.
    """
    _, n = bvps.shape
    if n < 256:
        seglength = n
        overlap = int(0.8*n)  # fixed overlapping
    else:
        seglength = 256
        overlap = 200
    # -- periodogram by Welch
    F, P = welch(bvps, nperseg=seglength, noverlap=overlap, fs=fps, nfft=nfft)
    F = F.astype(np.float32)
    P = P.astype(np.float32)
    # -- freq subband (0.65 Hz - 4.0 Hz)
    band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
    Pfreqs = 60*F[band]
    Power = P[:, band]
    return Pfreqs, Power

