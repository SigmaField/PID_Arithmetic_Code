import os
import time
import argparse
from mutual_info       import compute_mi as mi
from pathlib           import Path
from configparser      import ConfigParser
from itertools         import combinations
from pid_auxiliaries   import compute_and_store_analytical_results
from brain_data_reader import load_subjet_data, only_useful_data

left_channels  = ['EEG Fp1', 'EEG F3', 'EEG F7', 'EEG T3', 'EEG C3', 'EEG T5', 'EEG P3', 'EEG O1', 'EEG Fz', 'EEG Cz', 'EEG Pz']
right_channels = ['EEG Fp2', 'EEG F4', 'EEG F8', 'EEG T4', 'EEG C4', 'EEG T6', 'EEG P4', 'EEG O2', 'EEG Fz', 'EEG Cz', 'EEG Pz']
left_triplets  = [list(triplet) for triplet in combinations(left_channels,3)]
right_triplets = [list(triplet) for triplet in combinations(right_channels,3) if triplet!=('EEG Fz', 'EEG Cz', 'EEG Pz')] #triplet (EEG Fz,EEG Cz,EEG Pz) already in left channels, avoid repeat in right channels
left_pairs     = list(combinations(left_channels,2))
right_pairs    = list(combinations(right_channels,2))

def get_file_paths_from_config():
  current_directory = os.path.dirname(os.path.abspath(__file__))
  config_file       = Path(current_directory) / 'config.ini'
  config            = ConfigParser()
  config.read(config_file)
  data_path    = config.get('DEFAULT', 'data_path')
  results_path = config.get('DEFAULT', 'results_path')
  return data_path,results_path

def str_to_int_list(input_string):
    return [int(item) for item in input_string.split(',')]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-subjectids', type=str_to_int_list, help='List of integers separated by commas')
    parser.add_argument("-stageid",help='stage of the experiment, can be 1 for rest or 2 for task',choices=["1","2"])
    args  = parser.parse_args()

    data_folder, rslts_folder =  get_file_paths_from_config()
   
    Path(rslts_folder+"\\rest").mkdir(exist_ok=True)
    Path(rslts_folder+"\\task").mkdir(exist_ok=True)
   
    subject_IDs = args.subjectids
    stageid     = args.stageid
    stage       = "rest" if stageid == "1" else "task"
    start = time.time()
    for ID in subject_IDs:
        print('\nCurrently processing test subject', str(ID),'...',end='\n\n')
        data_path_full   = data_folder + "Subject" + str(ID) + "_" + stageid + ".edf"
        result_path_full = rslts_folder + stage + '\\Subject' + str(ID) + "_PID_" + stage + "_128Hz_continuous_analytical.csv"
        raw_data         = load_subjet_data(data_path_full)
        useful_raw_data  = only_useful_data(raw_data,dataset='ari',stage=stage)
        left_MIs         = {frozenset(pair):mi(useful_raw_data[pair[0]],useful_raw_data[pair[1]]) for pair in left_pairs} 
        right_MIs        = {frozenset(pair):mi(useful_raw_data[pair[0]],useful_raw_data[pair[1]]) for pair in right_pairs} 
        results          = {'source1':[],'source2':[],'target' :[],'sinergy':[],'redundancy':[]}
        compute_and_store_analytical_results(useful_raw_data, left_triplets, right_triplets,left_MIs,right_MIs, results, result_path_full)
        print('\nFinished!\n\n')
    end = time.time()
    print("ELAPSED TIME:",(end-start)/60)
if __name__ == "__main__":
    main()