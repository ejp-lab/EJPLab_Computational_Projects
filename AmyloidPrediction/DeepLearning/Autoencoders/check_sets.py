import cudf
import glob
from tqdm import tqdm
import sys

# Used to debug. Checks if the Sequence set and pySAR set are of same length. (DL_Pytoch) 
# To run: python check_sets.py [pysar_set] [sequence_set]
# Prints any missing data in the sequence set from the pysar set

if __name__ == "__main__":

    pysar_dir = sys.argv[1]
    seq_dir = sys.argv[2]

    pysar_files = glob.glob(f"{pysar_dir}/*")
    seq_files = glob.glob(f"{seq_dir}/*")

    for idx, seq_file in enumerate(tqdm(seq_files)):
        
        seq_df = cudf.read_csv(seq_file)
        pysar_df = cudf.read_csv(pysar_files[idx])

        len_seq = seq_df.__len__()
        len_pysar = pysar_df.__len__()

        if len_seq != len_pysar:
            print(seq_file)

            missing_data = pysar_df.loc[~pysar_df['Sequence'].isin(seq_df['Sequence'])]
            print(missing_data)
    