# Protein Stability Prediction Models
## Overview
This repository contains code and data for training models to predict protein stabilities upon mutations. The models are organized into specific tasks focusing on GFP fluorescence (heavily influenced by the stability of GFP), dG (protein tertiary stability), and ddG (protein quaternary stability) predictions.

## Repository Structure
- **📂 EDA/**
  - Contains the initial data exploration of dG datasets
  - Helps in understanding data distributions and key features
 
- **📂 GFP, dG, ddG**
  - Each folder contains:
    - **train.joblib**: the dataset for model training
    - **val.joblib**: the dataset for model validation
    - **test.joblib**: the dataset for model testing
    - **persequence_classfication_t5_finetuning.py**: training code
