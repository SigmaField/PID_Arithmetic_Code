import mne
import pandas as pd
import numpy  as np

def load_subjet_data(path: str) -> pd.DataFrame:
    """
    Loads an eeg or meg data set using the package mne and returns the data in a pandas DataFrame.
    Parameters:
        path: path to a single subject data set
    Returns:
        pandas DataFrame with full data, including all channels.   
    """

    format = path[-3:]
    if format == 'edf':
        return mne.io.read_raw_edf(path).copy().resample(sfreq=128).to_data_frame(verbose=False)
    elif format == 'set':
        return mne.io.read_raw_eeglab(path).to_data_frame(verbose=False)
    return

def only_useful_data(raw_data: pd.DataFrame, dataset:str, stage:str) -> pd.DataFrame:
    """
    Selects non-ground channels and appropriate time intervals of data where needed.
    Currently taking into consideration data sets of the experiments
    Arithmetic Data Tasks (dataset='ari') and Healthy Brain Network (dataset='hbn').
    
    Parameters: 
        raw_data: raw data as read from the unprocessed data sets in pandas DataFrame format, may include reference channels or non-EEG data.
        dataset:  string with id of the data set stored in 'data'. It can one of the following: ari, hbn.
    Returns: 
        pandas DataFrame without ground channel and if needed, a subset of time considered adequate.
    """
  
    if dataset == 'ari':
        if stage == 'rest':
            raw_data = raw_data[(120<=raw_data.time) & (raw_data.time<=180)] # only last minute when using resting state data
        return raw_data[raw_data.columns[1:-2]]                              # 1st column: time, last 2 columns: potential zero and heart data
    if dataset == 'hbn':
        return raw_data[raw_data.columns[:-1]]                               # last column: potential zero

def binarize_data(data:pd.DataFrame) -> pd.DataFrame:
    """
    Discretizes each EEG channel data by assigning 1 to each observation if it's above the median and 0 if it's below.
       Parameters:
           data: data frame with EEG channels data
        Returns:
            binary_data: Dataframe with binarized  EEG channels data.
    """
    binary_data = data.copy(deep=True)
    for col in data.columns:
        binary_data[col] = (binary_data[col] <= np.median(binary_data[col])).astype(int)
    return binary_data