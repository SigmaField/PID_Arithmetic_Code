import numpy  as np
import pandas as pd
from typing import Dict
from scipy.linalg import cholesky
from mutual_info import compute_mi as mi
from mutual_info import compute_cmi as cmi
from scipy.stats import pearsonr

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

def pid_row_continuous(data: pd.DataFrame, triplet:list, results:Dict[str,list]) -> None:
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

def pid_analytical(data: pd.DataFrame, triplet:list[str],  MIs:dict[frozenset[str],float], results:dict[str,list])->None:
    triplet_permutations = [(0,1,2),(1,0,2),(2,1,0)]
    for idx_t, idx_s1, idx_s2 in triplet_permutations:
        #locate source with minimal mutual information with respect to the target
        target          = triplet[idx_t]
        sources         = [triplet[idx_s1],triplet[idx_s2]]
        I1              = MIs[frozenset((target,sources[0]))]
        I2              = MIs[frozenset((target,sources[1]))]
        idx_min,idx_max = np.argsort([I1,I2])
        source_min      = sources[idx_min]
        source_max      = sources[idx_max]
        #assign correlation coefficient 'a' to pair with minimal MI, 'b' to remaining pair and 'c' between sources.
        a               = pearsonr(data[target],data[source_min]).statistic     #correlation of pair (target,source) with minimal MI
        c               = pearsonr(data[target],data[source_max]).statistic     #correlation of remaining (target,source) pair
        b               = pearsonr(data[source_min],data[source_max]).statistic #correlation between sources
        a2, b2, c2      = np.square(a),np.square(b),np.square(c)
        #compute PID assuming Gaussian random variables using as correlation matrix the sample correlation matrix
        redundancy      = 0.5*np.log(1/(1-a2))
        synergy         = 0.5*np.log(((1-b2)*(1-c2))/(1-(a2+b2+c2)+2*a*b*c)) 
        results['source1'].append(sources[0])
        results['source2'].append(sources[1])
        results['target'].append(target)
        results['sinergy'].append(synergy)
        results['redundancy'].append(redundancy)
    return


def compute_and_store_analytical_results(data_df:pd.DataFrame, left_triplets:list, right_triplets:list, left_MIs:Dict[frozenset[str],float], right_MIs:Dict[frozenset[str],float], results:dict, path:str)-> None:
    print("Computing PID on LEFT hemisphere...")
    print("")
    for triplet in left_triplets:
       pid_analytical(data_df, triplet, left_MIs,results)
    print("Computing PID on RIGHT hemisphere...")
    print("")
    for triplet in right_triplets:
       pid_analytical(data_df, triplet, right_MIs,results)    
    pd.DataFrame(results).to_csv(path)    


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
    surrogate_df    = pd.DataFrame(surrogate_array.T,columns=data_df.columns)
    return surrogate_df
