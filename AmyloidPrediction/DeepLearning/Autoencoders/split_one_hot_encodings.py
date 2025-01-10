import dask_cudf
import glob
import cudf
from tqdm import tqdm
import os
import pandas as pd
import sys
import dask.dataframe as dd

# Splits the sequence datasets into the same order as the pySAR datasets. (DL_Pytoch)
# To run: python split_one_hot_encodings.py [sequence_set] [pysar_set] [new_dir]  

if __name__ == "__main__":

    sequence_features = sys.argv[1]
    pysar_features = sys.argv[2]
    new_dir = sys.argv[3]

    seq_df = dask_cudf.read_csv(sequence_features, blocksize="40MB")
    pysar_parts = glob.glob(f"{pysar_features}/*")

    files_done = [i.split("/")[-1] for i in glob.glob(f"{new_dir}/*")]

    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)

    for file in tqdm(pysar_parts):
        
        file_part = file.split("/")[-1]

        if file_part in files_done:
            print("Skipping")
            continue
    
        df = pd.read_csv(file)

        seq_part = seq_df.loc[seq_df['Sequence'].isin(df['Sequence'])]

        seq_part.to_csv(f"{new_dir}/{file_part}", single_file=True)
