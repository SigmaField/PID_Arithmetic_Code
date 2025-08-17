import dit
import numpy  as np
import pandas as pd
from typing import Dict,NoReturn
from scipy.linalg import cholesky
from mutual_info import compute_mi as mi
from mutual_info import compute_cmi as cmi


def PID_continuous(s1,s2,t):
    I1  = mi(s1,t,n_neighbors=10)
    I2  = mi(s2,t,n_neighbors=10)
    I12 = cmi(t,s1,s2,n_neighbors=10) + mi(t,s2,n_neighbors=10) 
    r   = np.min([I1,I2])
    u1  = I1-r
    u2  = I2-r
    s   = I12-u1-u2-r
    results = {'u1':u1,'u2':u2,'r':r,'s':s}
    return results

def pid_row_continuous(data: pd.DataFrame, triplet:list, results:Dict[str,list]) -> NoReturn:
    rvs = data[triplet].to_numpy().T 
    def fill_row(i,j,k):
        pid    = PID_continuous(rvs[i],rvs[j],rvs[k])
        results['source1'].append(triplet[i])
        results['source2'].append(triplet[j])
        results['target'].append(triplet[k])
        results['sinergy'].append(pid['s'])
        results['unique1'].append(pid['u1'])
        results['unique2'].append(pid['u2'])
        results['redundancy'].append(pid['r'])
    fill_row(0,1,2)
    fill_row(0,2,1)
    fill_row(1,2,0)
    return 

def generate_surrogates(data_df:pd.DataFrame) -> pd.DataFrame:
    """
    Creates a sample of surrogate data via Cholesky decomposition.
    Parameters:
        data_df: a dataframe containing the empirical EEG data
    Returns:
        surrogate_df: a dataframe containing the surrogate EEG data
    """

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



