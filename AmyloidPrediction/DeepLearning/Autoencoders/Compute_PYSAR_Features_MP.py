import numpy as np
import pySAR
from pySAR.descriptors import *
from tqdm import tqdm
import multiprocessing as mp

# Compute_Pysar_Features.py - Generates PySAR features for 6-mer sequences. Excuted with the wrapper file. (DL_Pytoch)  

'''
Globals
''' 

test_peptide = 'AAAAAC'
ac = pySAR.PyBioMed.PyBioMed.PyProtein.Autocorrelation
ctd = pySAR.PyBioMed.PyBioMed.PyProtein.CTD

feature_name_dict = {**ac.CalculateGearyAutoFreeEnergy(test_peptide), **ctd.CalculateCompositionHydrophobicity(test_peptide)}

feature_name_list = list(feature_name_dict.keys())
out_file_name = 'PeptideManifoldWithPysarAutocorrelationFeatures.csv'

with open(out_file_name, 'w') as h:

    h.write('Sequence' + ',')
    for name_idx, name in enumerate(feature_name_list):
        if name_idx != len(feature_name_list) - 1:
            h.write(name + ',')
        else:
            h.write(name + '\n')

def worker(arg, q):

    '''runs long running process'''

    try:
        features = list({**ac.CalculateGearyAutoFreeEnergy(arg), **ctd.CalculateCompositionHydrophobicity(arg)}.values())
        features = ','.join([arg] + [str(i) for i in features]) + '\n'

    except:

        features = list({**ctd.CalculateCompositionHydrophobicity(arg)}.values())
        features = ','.join([arg] + ['0.0' for i in range(30)] + [str(i) for i in features]) + '\n'

    q.put(features)

    return features

def listener(q):

    '''listens for messages on the q, writes to file. '''

    with open(out_file_name, 'a') as f:
        while 1:
            m = q.get()
            if m == 'kill':
                f.write('killed')
                break
            f.write(str(m))
            f.flush()

def main():

    #must use Manager queue here, or will not work
    manager = mp.Manager()
    q = manager.Queue()    
    pool = mp.Pool(mp.cpu_count() - 2)

    #put listener to work first
    watcher = pool.apply_async(listener, (q,))

    f = open('PeptideManifold.csv', 'r')
    sequences = f.readlines()
    f.close()

    #fire off workers
    jobs = []
    for seq in tqdm(sequences):
        job = pool.apply_async(worker, (seq, q))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs: 
        job.get()

    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()

    return None

if __name__ == "__main__":

    main()