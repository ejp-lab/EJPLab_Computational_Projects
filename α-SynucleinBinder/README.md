# Machine Learning Prediction of Multiple Distinct High-Affinity Chemotypes for α-Synuclein Fibrils
This repository contains the data and code associated with our study:
“Machine Learning Prediction of Multiple Distinct High-Affinity Chemotypes for α-Synuclein Fibrils” by Xinning Li et al. (University of Pennsylvania, Washington University in St. Louis, and Sentauri Inc.)

##🧠 Overview
We developed a machine learning framework to predict new high-affinity ligands that bind to α-synuclein fibrils that is the key pathological features of Parkinson’s disease and related synucleinopathies.
Despite being trained on fewer than 300 experimentally measured binding affinities, the model robustly generalized across chemical space and successfully identified five new sub-10 nM binders from a 140 million-compound virtual library.


## Repository Structure
- **📂 Data **
  - Contains data for model development: tran.csv, test.csv
  - Contains data extracted from the full Mcule library to identify new binders: prospective_data.csv
  - Cont
  - Helps in understanding data distributions and key features
 
- **📂 GFP**
  - **train.joblib**: the dataset for GFP model training
  - **val.joblib**: the dataset for GFP model validation
  - **test.joblib**: the dataset for GFP model testing
  - **persequence_classification_t5_finetuning_GFP.py**: GFP model training code
  - **models.py**: model architecture
  - **util.py**: util functions
 
- **📂 dG**
  - **train.joblib**: the dataset for dG model training
  - **val.joblib**: the dataset for dG model validation
  - **test.joblib**: the dataset for dG model testing
  - **persequence_regression_t5_finetuning_Tsuboyama.py**: dG model training code
  - **models.py**: model architecture
  - **util.py**: util functions
 
- **📂 ddG**
  - **skempi_train.joblib**: the dataset for ddG model training
  - **skempi_val.joblib**: the dataset for ddG model validation
  - **skempi_test.joblib**: the dataset for ddG model testing
  - **persequence_regression_t5_finetuning_SKEMPI.py**: ddG model training code
  - **models.py**: model architecture
  - **util.py**: util functions
  
- **📄 environment.yml**
  - Configuration file for environment settings
 
- **💡 Training Instructions**
  - Directly run `persequence_<BACKBONE>_t5_finetuning_<DATASET>.py` in each folder 

  - ``` python persequence_<BACKBONE>_t5_finetuning_<DATASET>.py ```

