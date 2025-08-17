from pathlib           import Path
from itertools         import combinations
from pid_auxiliaries   import pid_row_continuous,generate_surrogates

from brain_data_reader import load_subjet_data, only_useful_data
from typing            import Dict,NoReturn
from tqdm              import tqdm
import pandas          as     pd
import argparse
import time

left_channels  = ['EEG Fp1', 'EEG F3', 'EEG F7', 'EEG T3', 'EEG C3', 'EEG T5', 'EEG P3', 'EEG O1', 'EEG Fz', 'EEG Cz', 'EEG Pz']
right_channels = ['EEG Fp2', 'EEG F4', 'EEG F8', 'EEG T4', 'EEG C4', 'EEG T6', 'EEG P4', 'EEG O2', 'EEG Fz', 'EEG Cz', 'EEG Pz']
    
left_triplets  = [list(triplet) for triplet in combinations(left_channels,3)]
right_triplets = [list(triplet) for triplet in combinations(right_channels,3)]

def str_to_int_list(input_string):
    return [int(item) for item in input_string.split(',')]

def analyze_and_store_results(continuous_data_df:pd.DataFrame, results:Dict, path:str)-> NoReturn:
    print("Computing PID on LEFT hemisphere...")
    print("")
    for triplet in tqdm(left_triplets):
       pid_row_continuous(continuous_data_df, triplet, results)
    print("Computing PID on RIGHT hemisphere...")
    print("")
    for triplet in tqdm(right_triplets):
       pid_row_continuous(continuous_data_df, triplet, results)    
    pd.DataFrame(results).to_csv(path)    

def main():
    parser     = argparse.ArgumentParser()
    parser.add_argument('-subjectids', type=str_to_int_list, help='List of integers separated by commas')
    parser.add_argument("-stageid",help='stage of the experiment, can be 1 for rest or 2 for task',choices=["1","2"])
    parser.add_argument("-surrogates",help='wether to do the PID analysis on original data or on Cholesky surrogates',choices=["yes","no"])    
    parser.add_argument("-samplesize",help='number of surrogate samples')
    parser.add_argument('-datafolder',    default="G:\\My Drive\\data_research\\NeuNet\\PID_Arithmetic\\data\\", help='path of raw data')
    parser.add_argument('-resultsfolder', default="G:\\My Drive\\data_research\\NeuNet\\PID_Arithmetic\\resultsPID\\", help='path for PID analysis results')
    args  = parser.parse_args()
    stage =  "rest" if args.stageid  == "1" else "task"
    Path(args.resultsfolder+"\\rest").mkdir(exist_ok=True)
    Path(args.resultsfolder+"\\task").mkdir(exist_ok=True)

    subject_IDs    = args.subjectids
    
    start = time.time()
    for ID in subject_IDs:
        print('\nCurrently processing test subject', str(ID),'...',end='\n\n')
        data_path       = args.datafolder + "Subject" + str(ID) + "_" + args.stageid + ".edf"
        raw_data        = load_subjet_data(data_path)
        useful_raw_data = only_useful_data(raw_data,dataset='ari',stage=stage)

        if args.surrogates == 'yes':
            samplesize     = int(args.samplesize)
            for n in range(samplesize):
                results         = {'source1':[],'source2':[],'target' :[],'sinergy':[],'unique1':[],'unique2':[],'redundancy':[]}
                print("\nCurrently processing surrogate #"+str(n))
                surrogates_path = args.resultsfolder  + stage + '_surrogates' +'\\Subject' + str(ID) + "_PID_" + stage + "_surrogate"+str(n)+"_128Hz_continuous.csv"
                analyze_and_store_results(generate_surrogates(useful_raw_data), results, surrogates_path)
        else:
            results         = {'source1':[],'source2':[],'target' :[],'sinergy':[],'unique1':[],'unique2':[],'redundancy':[]}
            result_path     = args.resultsfolder  + stage + '\\Subject' + str(ID) + "_PID_" + stage + "_128Hz_continuous.csv"
            analyze_and_store_results(useful_raw_data, results, result_path)

        print('\nFinished!\n\n')
    end = time.time()
    print("ELAPSED TIME:",(end-start)/60)
if __name__ == "__main__":
    main()


