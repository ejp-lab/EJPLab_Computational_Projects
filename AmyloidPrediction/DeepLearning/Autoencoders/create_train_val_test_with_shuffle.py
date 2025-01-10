import pandas as pd
import dask.dataframe as dd
from dask.array.slicing import shuffle_slice
import numpy as np
import sys

#create_train_val_test_with_shuffle.py - Splits the array into training, validation, and testing using dask for pySAR features. WARNING: Parameter for parition size may need to be adjusted based on computer specifications. (DL_Pytoch) NEEDS FIXING  
# To run: python create_train_val_test_with_shuffle.py [pysar_set]  

np.random.seed(42)

if __name__ == "__main__":

    df = dd.read_csv(f"{sys.argv[1]}/*", blocksize="100MB").reset_index()

    mutants_in_test = pd.read_csv("Dataset.csv")['name'].to_list()

    mutant_test = df.loc[df['Sequence'].isin(pd.Series(mutants_in_test))].compute()
    df = df.loc[~df['Sequence'].isin(mutants_in_test)]

    df_10 = df.sample(frac=0.1)
    df = df.loc[~df['Sequence'].isin(df_10['Sequence'].compute())]

    d_arr = df.to_dask_array(True)
    df_len = len(df)
    index = np.random.choice(df_len, df_len, replace=False)
    d_arr = shuffle_slice(d_arr, index)
    df = d_arr.to_dask_dataframe(df.columns)

    df_10.to_csv("Validation_Set",header=None, index=None)
    df.to_csv("Training_Set", header=None, index=None)
    mutant_test.to_csv("Testing_Set", header=None, index=None)
