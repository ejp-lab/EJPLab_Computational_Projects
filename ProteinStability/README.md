# Protein Stability Prediction Models
## Overview
Predicting how amino‑acid substitutions affect protein stability is a fundamental problem that impacts enzyme engineering, drug discovery and our basic understanding of folding thermodynamics.  This repository collects end‑to‑end workflows—from data curation to model training and evaluation—for three complementary stability prediction tasks.

| Task | Experimental proxy for stability | Folder |
|------|----------------------------------|--------|
| **GFP** | Fluorescence intensity of Green Fluorescent Protein variants  | `GFP/` |
| **dG** | Absolute folding free energy (ΔG) in kcal mol⁻¹ for single‑chain proteins upon mutation| `dG/` |
| **ddG** | Change in binding free energy (ΔΔG) in kcal mol⁻¹of protein–protein complexes upon mutation | `ddG/` |


## Repository Structure
- **📂 EDA**
  - Contains the initial data exploration of dG datasets
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

  ``` python persequence_<BACKBONE>_t5_finetuning_<DATASET>.py ```
