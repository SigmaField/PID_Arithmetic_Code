import dit
import numpy  as np
import pandas as pd
from typing import Dict,NoReturn
from scipy.linalg import cholesky

def symbolize(data:pd.DataFrame,pctls:list[int]) -> pd.DataFrame:
    """
    Discretizes each EEG channel data by binning the continuous values according to a given set of percentile values that function as catagories
       Parameters:
           data: data frame with EEG channels data
        Returns:
            symbolic_data: Dataframe with discretized EEG channels data.
    """
    
    symbolic_data = data.copy(deep=True)
    for col in data.columns:
        boundaries         = [np.percentile(data[col],q) for q in pctls]
        symbolic_data[col] = np.digitize(data[col],boundaries)
    return symbolic_data

def joint_symbols_to_string(symbols):
    return np.array2string(symbols,separator='')[1:-1]

def partial_info_decomp(r1,r2,r3):
    joint_symbols,freqs = np.unique(list(zip(r1,r2,r3)), axis=0, return_counts=True)
    symbol_strings      = list(map(joint_symbols_to_string,joint_symbols))
    distribution        = dit.Distribution(symbol_strings, freqs/sum(freqs))
    pid                 = dit.pid.PID_WB(distribution)
    return read_pid_table(str(pid))

# terminar type hints de función "boolean_joint_distribution"
def boolean_joint_distribution(*random_variates):
    """
    Gets the joint distribution functioin of n binary random variables in the format required by the package 
    for discrete information theory 'dit'.
    Parameters: 
        variables: collection of lists of observations of binary random variables.
    Returns:
        distribution: a 'dit' Distribution object representing the joint distribution of 'variables'
    """
    def joint_symbols_to_string(bits:np.ndarray[np.int32])->str:
        return np.array2string(bits,separator='')[1:-1]
    joint_symbols,frequencies = np.unique(list(zip(*random_variates)), axis=0, return_counts=True)
    symbol_strings            = list(map(joint_symbols_to_string,joint_symbols))
    distribution              = dit.Distribution(symbol_strings, frequencies/sum(frequencies))
    return distribution

def read_pid_table(pid_table: str) -> Dict[str,float]: 
    """
    Extracts the results of the partial information decomposition performed by the package 'dit' into a dictionary.
    Parameters:
        pid_table: table of pid results in string format
    Returns: 
        pid_dict: dictionary with each atom of the pid decomposition
    """
    pid_dict = {}
    pid_rows = pid_table.split('\n')[3:-1]
    for row in pid_rows:
        contents = row.split('|')
        pid_dict[contents[1].strip()] = float(contents[3])
    return pid_dict

def triplet_pid_to_csv(data: pd.DataFrame, triplet:frozenset, savepath:str) -> NoReturn:
    channel_names = list(triplet)
    channel0_data, channel1_data, channel2_data = data[channel_names].to_numpy().T 
   
    distro1      = boolean_joint_distribution(channel0_data,channel1_data,channel2_data)
    permutation1 = channel_names[0] + "," + channel_names[1] + "-" + channel_names[2]
    distro2      = boolean_joint_distribution(channel0_data,channel2_data,channel1_data)
    permutation2 = channel_names[0] + "," + channel_names[2] + "-" + channel_names[1]
    distro3      = boolean_joint_distribution(channel1_data,channel2_data,channel0_data)
    permutation3 = channel_names[1] + "," + channel_names[2] + "-" + channel_names[0]

    with open(savepath+'_'+permutation1+".txt", "w") as text_file:
        text_file.write(dit.pid.PID_WB(distro1).to_string())
    with open(savepath+'_'+permutation2+".txt", "w") as text_file:
        text_file.write(dit.pid.PID_WB(distro2).to_string())
    with open(savepath+'_'+permutation3+".txt", "w") as text_file:
        text_file.write(dit.pid.PID_WB(distro3).to_string())
    return

def pid_row(data: pd.DataFrame, triplet:list, results:Dict[str,list]) -> NoReturn:
    rvs      = data[triplet].to_numpy().T 
    def fill_row(i,j,k):
        distro = boolean_joint_distribution(rvs[i],rvs[j],rvs[k])
        pid    = read_pid_table(dit.pid.PID_WB(distro).to_string())
        results['source1'].append(triplet[i])
        results['source2'].append(triplet[j])
        results['target'].append(triplet[k])
        results['sinergy'].append(pid['{0:1}'])
        results['unique1'].append(pid['{1}'])
        results['unique2'].append(pid['{0}'])
        results['redundancy'].append(pid['{0}{1}'])
    fill_row(0,1,2)
    fill_row(0,2,1)
    fill_row(1,2,0)
    return 
    
def pid_row_nonbinary(data: pd.DataFrame, triplet:list, results:Dict[str,list]) -> NoReturn:
    rvs = data[triplet].to_numpy().T 
    def fill_row(i,j,k):
        pid    = partial_info_decomp(rvs[i],rvs[j],rvs[k])
        results['source1'].append(triplet[i])
        results['source2'].append(triplet[j])
        results['target'].append(triplet[k])
        results['sinergy'].append(pid['{0:1}'])
        results['unique1'].append(pid['{1}'])
        results['unique2'].append(pid['{0}'])
        results['redundancy'].append(pid['{0}{1}'])
    fill_row(0,1,2)
    fill_row(0,2,1)
    fill_row(1,2,0)
    return 

def generate_surrogates(data_df:pd.DataFrame) -> pd.DataFrame:
    """
    Creates a sample of surrogate data via Cholesky decomposition.
    Parameters:
        data_df: a dataframe containing only the useful columns of the EEG data
    Returns:
        surrogate_df: a dataframe containing the Cholesky surrogates
    """
    def cholesky_surrogates(series):
        s = np.copy(series)

        for i in range(len(s)):
            s[i] = np.random.permutation(s[i])
            s[i] = (s[i]-np.mean(s[i]))/np.std(s[i])
        pearson_corr  = np.corrcoef(series)
        x             = s
        c             = cholesky(pearson_corr, lower=True)
        choles        = np.dot(c, x)
        return choles
    def gaussian_cholesky_surrogates(series):
        pearson_corr  = np.corrcoef(series)
        x             = np.random.normal(size=series.shape)
        c             = cholesky(pearson_corr, lower=True)
        choles        = np.dot(c, x)
        return choles   
    def random_permutation_surrogates(series):
        rng   = np.random.default_rng()
        return rng.permuted(series, axis=1)
    data_array      = data_df.to_numpy()
    surrogate_array = gaussian_cholesky_surrogates(data_array.T)
#    surrogate_array = random_permutation_surrogates(data_array.T)
    surrogate_df    = pd.DataFrame(surrogate_array.T,columns=data_df.columns)
    return surrogate_df


    
    


