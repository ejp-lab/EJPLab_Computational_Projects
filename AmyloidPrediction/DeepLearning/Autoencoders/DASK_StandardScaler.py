import sys
import numpy as np
import dask_cudf as dd
from dask import array as da
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# DASK_StandardScaler.py - Scales data generated from previous two scripts. (DL_Pytoch)  
# To run: python DASK_StandardScaler.py [pysar_set]  

if __name__ == "__main__":

    in_data_file = sys.argv[1]
    out_data_file = in_data_file.split('.csv')[0] + '_Scaled.csv'

    dask_df = dd.read_csv(in_data_file, blocksize="1GB") # The blocksize argument will make sure chunks dont exceed a certain size so you can never run out of memory

    header = ','.join(list(dask_df.columns)) + '\n'

    scaler = StandardScaler()
    for chunk in dask_df.partitions:
        chunk = chunk.drop(chunk.columns[0], axis=1)
        scaler.partial_fit(chunk)

    with open(out_data_file, 'w') as h:
        h.write(header)

        for idx, chunk in enumerate(tqdm(dask_df.partitions)):

            chunk_sequences = chunk[chunk.columns[0]].compute().to_list()
            chunk = chunk.drop(chunk.columns[0], axis=1)

            tmp = scaler.transform(chunk)
            tmp = np.around(tmp, decimals=3).astype(str)

            for idx, row in enumerate(tmp):
                h.write(chunk_sequences[idx] + ',' + ','.join(list(row)) + '\n')
