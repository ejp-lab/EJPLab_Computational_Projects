import os 
import sys
import glob
import numpy as np
import pySAR
from pySAR.descriptors import *
from tqdm import tqdm
import multiprocessing as mp

# Compute_Pysar_Features_MP_Wrapper.py - Handles the multiprocessing for generating the PySAR features  
# To run: run Compute_Pysar_MP_Wrapper.py

def ComputeFeatures_MP(tmp_list):

    outname_flag = str(tmp_list[0])
    start_index = str(tmp_list[1])
    end_index = str(tmp_list[2])

    os.system('python 2_Compute_PYSAR_Features_MP.py ' + outname_flag + ' ' + start_index + ' ' + end_index)

    return None

def main():

    total_sequences = 64000000

    index_array = np.arange(total_sequences)
    index_chunks = np.array_split(index_array, 64)

    inds_to_mp = [[idx, chunk[0], chunk[-1]] for idx, chunk in enumerate(index_chunks)]

    pool = mp.Pool(32)
    pool.map(ComputeFeatures_MP, inds_to_mp)
    pool.close()
    pool.join()

    os.system('cat PeptideManifoldWithPysarAutocorrelationFeatures_*.csv > PeptideManifoldWithPysarAutocorrelationFeatures.csv')
    
    files_to_remove = glob.glob('PeptideManifoldWithPysarAutocorrelationFeatures_*.csv')
    for fil in files_to_remove:
        os.remove(fil)

    return None

if __name__ == "__main__":

    main()
