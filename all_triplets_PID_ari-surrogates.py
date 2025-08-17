from pathlib           import Path
from itertools         import combinations
from dit_auxiliaries   import pid_row,generate_surrogates
from brain_data_reader import load_subjet_data, only_useful_data, binarize_data
import argparse
import pandas as pd

def main():
    parser     = argparse.ArgumentParser()
    parser.add_argument("-stageid",help='stage of the experiment, can be 1 or 2 for rest and task, accordingly',choices=["1","2"])
    parser.add_argument('-datafolder', default="C:\\Users\\rober\\NeuNet\\ArithmeticTasks\\data\\", help='path of raw data')
    parser.add_argument('-resultsfolder', default="C:\\Users\\rober\\NeuNet\\ArithmeticTasks\\resultsPID\\", help='path for PID analysis results')
    args  = parser.parse_args()
    stage =  "rest" if args.stageid  == "1" else "task"

    Path(args.resultsfolder+"\\rest").mkdir(exist_ok=True)
    Path(args.resultsfolder+"\\task").mkdir(exist_ok=True)

    left_channels  = ['EEG Fp1', 'EEG F3', 'EEG F7', 'EEG T3', 'EEG C3', 'EEG T5', 'EEG P3', 'EEG O1', 'EEG Fz', 'EEG Cz', 'EEG Pz']
    right_channels = ['EEG Fp2', 'EEG F4', 'EEG F8', 'EEG T4', 'EEG C4', 'EEG T6', 'EEG P4', 'EEG O2', 'EEG Fz', 'EEG Cz', 'EEG Pz']
    
    left_triplets  = [list(triplet) for triplet in combinations(left_channels,3)]
    right_triplets = [list(triplet) for triplet in combinations(right_channels,3)]
    subject_IDs    = [1]

    for ID in subject_IDs:
        print('Currently processing test subject', str(ID),'...',end='\n\n')

        results         = {'source1':[],'source2':[],'target' :[],'sinergy':[],'unique1':[],'unique2':[],'redundancy':[]}
        data_path       = args.datafolder + "Subject" + str(ID) + "_" + args.stageid + ".edf"
        
        raw_data        = load_subjet_data(data_path)
        useful_raw_data = only_useful_data(raw_data,dataset='ari',stage=stage)

        result_path     = args.resultsfolder  + stage + '\\Subject' + str(ID) + "_PID_" + stage + ".csv"
        binary_data     = binarize_data(useful_raw_data)
    
        for triplet in left_triplets:
            pid_row(binary_data, triplet, results)
        for triplet in right_triplets:
            pid_row(binary_data, triplet, results)    
        pd.DataFrame(results).to_csv(result_path)

        print('\nFinished!\n\n')

if __name__ == "__main__":
    main()