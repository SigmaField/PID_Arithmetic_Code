#------------------------------------REPL SANDBOX--------------------------------------

import pandas as pd
import numpy  as np
from scipy.stats import pearsonr
from mutual_info import compute_mi as mi

def pid_analytical(data: pd.DataFrame, triplet:list[str],  MIs:dict[frozenset[str],float], results:dict[str,list])->None:
    triplet_permutations = [(0,1,2),(1,0,2),(2,1,0)]
    for idx_t, idx_s1, idx_s2 in triplet_permutations:
        target          = triplet[idx_t]
        sources         = [triplet[idx_s1],triplet[idx_s2]]
        I1              = MIs[frozenset((target,sources[0]))]
        I2              = MIs[frozenset((target,sources[1]))]
        idx_min,idx_max = np.argsort([I1,I2])
        source_min      = sources[idx_min]
        source_max      = sources[idx_max]
        a               = pearsonr(data[target],data[source_min]).statistic     #correlation corresponding to pair (target,source) that attains minimal mutual information
        c               = pearsonr(data[target],data[source_max]).statistic     #correlation corresponding to pair (target,source) that didn't attain minimal mutual information
        b               = pearsonr(data[source_min],data[source_max]).statistic #correlation between sources
        a2, b2, c2      = np.square(a),np.square(b),np.square(c) 
        redundancy      = 0.5*np.log(1/(1-a2))
        synergy         = 0.5*np.log(((1-b2)*(1-c2))/(1-(a2+b2+c2)+2*a*b*c)) 
        results['source1'].append(sources[0])
        results['source2'].append(sources[1])
        results['target'].append(target)
        results['sinergy'].append(synergy)
        results['redundancy'].append(redundancy)
    return


a,b,c      = -0.1,0.6,0.5
a2, b2, c2 = np.square(a),np.square(b),np.square(c)
cov        = np.array([[1,a,c], [a,1,b],[c,b,1]])
x,y,z      = np.random.multivariate_normal(np.zeros(3),cov,size=15000).T
data_df    = pd.DataFrame({'x':x,'y':y,'z':z})
triplet    = ['x','y','z']
MIs        = {frozenset(('x','y')):mi(x,y),frozenset(('x','z')):mi(x,z),frozenset(('y','z')):mi(y,z)}
results    = {'source1':[],'source2':[],'target' :[],'sinergy':[],'redundancy':[]}
pid_analytical(data_df,triplet,MIs,results)
results_df = pd.DataFrame(results)


_a_   = pearsonr(x,y).statistic
_b_   = pearsonr(y,z).statistic
_c_   = pearsonr(x,z).statistic
_a2_  = np.square(_a_)
_b2_  = np.square(_b_)
_c2_  = np.square(_c_)

_redundancy_yzx = 0.5*np.log(1/(1-_a2_))
_synergy_yzx    = 0.5*np.log(((1-_b2_)*(1-_c2_))/(1-(_a2_+_b2_+_c2_)+2*_a_*_b_*_c_)) 



_a_   = pearsonr(x,y).statistic
_b_   = pearsonr(x,z).statistic
_c_   = pearsonr(y,z).statistic
_a2_  = np.square(_a_)
_b2_  = np.square(_b_)
_c2_  = np.square(_c_)

_redundancy_xzy = 0.5*np.log(1/(1-_a2_))
_synergy_xzy    = 0.5*np.log(((1-_b2_)*(1-_c2_))/(1-(_a2_+_b2_+_c2_)+2*_a_*_b_*_c_)) 



_a_   = pearsonr(x,z).statistic
_b_   = pearsonr(y,x).statistic
_c_   = pearsonr(y,z).statistic
_a2_  = np.square(_a_)
_b2_  = np.square(_b_)
_c2_  = np.square(_c_)

_redundancy_yxz = 0.5*np.log(1/(1-_a2_))
_synergy_yxz    = 0.5*np.log(((1-_b2_)*(1-_c2_))/(1-(_a2_+_b2_+_c2_)+2*_a_*_b_*_c_)) 


print(results_df)
print(_synergy_yzx,_redundancy_yzx,'\n',_synergy_xzy,_redundancy_xzy,'\n',_synergy_yxz,_redundancy_yxz)