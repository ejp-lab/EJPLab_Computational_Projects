import numpy as np

# Compute_OneHot_Features.py - Generates one-hot encodings for the 6-mer sequences. (DL_Pytoch)  

'''
Globals
'''

d = {'C' : 0, 'D' : 1, 'S' : 2, 'Q' : 3, 'K' : 4,
     'I' : 5, 'P' : 6, 'T' : 7, 'F' : 8, 'N' : 9, 
     'G': 10, 'H': 11, 'L': 12, 'R' : 13,'W': 14,
     'A': 15, 'V': 16, 'E': 17, 'Y': 18, 'M': 19}

aa_list = list(d.keys())
sequence_length = 6
n_aas = len(aa_list)
feature_name_list = [str(i) for i in range(sequence_length * n_aas)]

with open('PeptideManifoldWithOneHotFeatures.csv', 'w') as h:

    h.write('Sequence' + ',')
    for name_idx, name in enumerate(feature_name_list):
        if name_idx != len(feature_name_list) - 1:
            h.write(name + ',')
        else:
            h.write(name + '\n')

def GetOneHotFeatures(pocket_string):

    onehot_features = []
    split_pocket = list(pocket_string)[0:-1]

    for pocket_res in split_pocket:

        tmp_feature_list = [0 for _ in range(n_aas)]
        tmp_feature_list[d[pocket_res]] = 1

        onehot_features += tmp_feature_list

    onehot_features = ','.join([pocket_string.strip('\n')] + [str(i) for i in onehot_features] + ['\n'])

    return onehot_features

if __name__ == "__main__":

    with open('PeptideManifoldWithOneHotFeatures.csv', 'a') as h:
        with open('PeptideManifold.csv', 'r') as f:

            for line_idx, line in enumerate(f):

                out_string = GetOneHotFeatures(line)
                h.write(out_string)

                print(str((((line_idx + 1) / 64000000) * 100)) + '%')