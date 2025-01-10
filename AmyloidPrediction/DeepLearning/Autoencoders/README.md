# Autoencoders  

This folder contains the nescessary files for training the autoencoders. The datasets used for training these autoencoders are 10's of GB in size.

## Scripts for Generating Dataset
The pysar dataset was created first followed by the sequence dataset. If you want to create them locally, use the following scripts:  

Compute_OneHot_Features.py - Generates one-hot encodings for the 6-mer sequences. (DL)  
To run: `python Compute_OneHot_Features.py`  

Compute_Pysar_Features_MP.py - Generates PySAR features for 6-mer sequences. Excuted with the wrapper file Compute_PYSAR_Features_MP_Wrapper.py. (DL)  
Do not run directly.  

Compute_Pysar_Features_MP_Wrapper.py - Handles the multiprocessing for generating the PySAR features  
To run: `python Compute_Pysar_MP_Wrapper.py`

DASK_StandardScaler.py - Scales data generated from previous two scripts. (DL)  
To run: `python DASK_StandardScaler.py [pysar_set]`  

RemoveRedundantColumns.py - Remove columns with 0 variance (all 0's). (DL)  
To run: `python RemoveRedundantColumns.py [pysar_set]`

create_train_val_test_with_shuffle.py - Splits the array into training, validation, and testing using dask for pySAR features. WARNING: Parameter for parition size may need to be adjusted based on computer specifications. (DL)
To run: `python create_train_val_test_with_shuffle.py [pysar_set]`  

split_one_hot_encodings.py - Splits the sequence datasets into the same order as the pySAR datasets. (DL)  
To run: `python split_one_hot_encodings.py [sequence_set] [pysar_set] [new_dir]`  

check_sets.py - Checks if the Sequence set and pySAR set are of same length. (DL)  
To run: `python check_sets.py [pysar_set] [sequence_set]`  
Prints any missing data in the sequence set from the pysar set  

The following peptide was removed: "NAAAAA" for incompatibility with pandas.

## Scripts for Tuning Autoencoders
Autoencoders were trained using a custom generate_arrays_from_file function. This function reads the DASK generated files into memory and yields numpy arrays for training.  
To train the autoencoders, use the following scripts:  

Peptide_Autoencoder_HT_Seq.py - Trains a sequence autoencoder on peptide manifold. (DL_TFGPU)  
To run: `python Peptide_Autoecncoder_HT_Seq.py`  

Peptide_Autoencoder_Pysar_HT.py - Trains a pySAR autoencoder on peptide manifold. (DL_TFGPU)  
To run: `python Peptide_Autoencoder_Pysar_HT.py`  

## Scripts for Visualization  
Cluster_Seq_pysar_Manifold_DL_Kmeans.py - Combines the sequence and pySAR manifolds and performs kmeans clustering. Generates manifold figure. Splits datasets into testing and training. (DL)  
To run: `python Cluster_Seq_Pysar_Manfiolds_DL_Kmeans.py`  

Make_Embedding_Figures.py - Makes UMAP embeddings for sequence and pySAR manifolds separately. (DL)  
To Run: `python Make_Embedding_Figures.py`  

Cluster_Membership.py - Shows top 5 motifs in clusters. (DL)  
To run: `python Cluster_Membership.py [num_clusters] [length_of_motif]`  

## Optuna Studies  
Optuna study objects are provided. They hold information on optimization history and best parameters. They are held in the following files:  
Pysar_Study.pickle  
Seq_Study.pickle  

