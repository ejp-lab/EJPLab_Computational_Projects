# Machine Learning Prediction of Multiple Distinct High-Affinity Chemotypes for α-Synuclein Fibrils
This repository contains the data and code associated with our study:
“Machine Learning Prediction of Multiple Distinct High-Affinity Chemotypes for α-Synuclein Fibrils” by Xinning Li et al. (University of Pennsylvania, Washington University in St. Louis, and Sentauri Inc.)

## Overview
We developed a machine learning framework to predict new high-affinity ligands that bind to α-synuclein fibrils that is the key pathological features of Parkinson’s disease and related synucleinopathies.
Despite being trained on fewer than 300 experimentally measured binding affinities, the model robustly generalized across chemical space and successfully identified five new sub-10 nM binders from a 140 million-compound virtual library.


## Repository Structure


- **📂 Data** – Datasets used for model training, testing, and prospective screening
  
  - **Model development data:**
      - `tran.csv` – training set of experimentally measured ligands
      - `test.csv` – held-out test set for model evaluation
        
  - **Prospective screening data:**
      - `prospective_data.csv` – compounds extracted from the full Mcule library for new binder prediction
        
  - **Mordred descriptors (for model reproduction):**
      - `train_mordred.csv` – descriptor matrix for training compounds
      - `test_mordred.csv` – descriptor matrix for test compounds
      - `prospective_mordred.csv` – descriptor matrix for prospective screening compounds

- **📂 Model** – Files and scripts for reproducing the best-performing machine learning model
  
  - **Model components:**
    - `cv_selected_features.joblib` – selected Morgan fingerprint and Mordred descriptor features used in the final model
    - `cv_selected_logistic_regression.joblib` – optimized logistic regression model and hyperparameters
      
  - **Reproducing predictions:**
  - To reproduce or run inference on new data, execute:
    ```
    python predict.py \
      cv_selected_logistic_regression.joblib \   # trained logistic regression model
      cv_selected_features.joblib \              # selected feature set (fingerprints + Mordred)
      <YOUR_MORDRED_DATA.csv> \                  # Mordred descriptor file for your input molecules
      <YOUR_INPUT_DATA.csv> \                    # raw input file containing SMILES strings for prediction
      data_predictions.csv \                     # output file with model predictions
      SMILES                                     # column name in your input CSV containing SMILES strings
    ```
  - This command runs the trained logistic regression model to predict α-synuclein fibril binding.
  - `data_predictions.csv` — output file containing predicted binding labels and probabilities
  - `SMILES` — name of the column in your input file that contains SMILES strings
  
 
- **📂 Mcule Screening** - This section describes how to calculate chemical similarity across the full Mcule library (~140 million compounds) and visualize scaffold diversity relative to the training, test, and prospective datasets.
  
  - **Calculate Tanimoto Similarity for the Full Mcule Library**:
    - To perform multi-processing Tanimoto calculations against the three parental ligands (BV-21, TZ61-84, M503-1619), run:
      ```
      python 0_Tanimoto_Mcule.py \
        --smi mcule_db.smi \          # input Mcule database (.smi)
        --outdir ./mcule_fp_out \     # directory for fingerprint outputs
        --nprocs 64 \                 # number of CPU cores to use
        --count_lines                 # display progress

      ```
    - This command parallelizes computation of Morgan fingerprint (radius = 3, 1024 bits) Tanimoto similarities between each Mcule compound and the three parental scaffolds.
    - **📄 Output** `./mcule_fp_out/mcule_fp_label.parquet` — contains computed fingerprints and similarity scores.

  - **Calculate Tanimoto Similarity for Training, Test, and Prospective Datasets**:
    - Compute Morgan fingerprint-based (radius = 3, 1024 bits) Tanimoto similarities for train+test/prospective datasets:
      ```
      python 1_FP_Tanimoto_cal_for_sets.py INPUT_CSV OUTPUT_CSV
      ```
  - **Visualize Chemical-Space Similarity**
    - Overlay similarity distributions of training + test vs prospective compounds against the background Mcule library::
      ```
      python 2_visualize.py TRAIN_TEST_CSV PROSPECTIVE_CSV PARQUET_DATASET
      ```
    - This script generates histograms showing scaffold coverage and diversity, reproducing Fig. 2 from the manuscript.
  
- **📄 environment.yml**
  - Configuration file for environment settings


