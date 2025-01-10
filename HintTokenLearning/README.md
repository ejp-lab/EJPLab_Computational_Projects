# Hint Token Learning Repository
## Overview

This repository contains the data, training code, and best model parameters after hyper-parameter tuning in the hint token learning project.

## Repository Structure

- **📂 Best_Model_Params/**
    -   **KCNE1/**: Parameters that obtain the best-performing models fine-tuned from ProtBERT, ESM_650M, ProtT5, and ESM_15B on the KCNE1 dataset
    -   **PTEN/**: Parameters that obtain the best-performing models fine-tuned from ProtBERT, ESM_650M, ProtT5, and ESM_15B on the PTEN dataset
    -   **RecA/**: Parameters that obtain the best-performing models fine-tuned from ProtBERT, ESM_650M, ProtT5, and ESM_15B on the RecA dataset
    -   **TPMT/**: Parameters that obtain the best-performing models fine-tuned from ProtBERT, ESM_650M, ProtT5, and ESM_15B on the TPMT dataset

- **📂 Data/**
    -   **KCNE1/**: Original KCNE1 datasets used for training, validation, and testing
    -   **PTEN/**: Original PTEN datasets used for training, validation, and testing
    -   **RecA/**: Original RecA datasets used for training, validation, and testing
    -   **TPMT/**: Original TPMT datasets used for training, validation, and testing

- **📂 Model_Training/**
    -   **KCNE1/**: Original training code to train models from ProtBERT, ESM_650M, ProtT5, and ESM_15B on the KCNE1 dataset with and without the hint token learning strategy
    -   **PTEN/**: Original training code to train models from ProtBERT, ESM_650M, ProtT5, and ESM_15B on the PTEN dataset with and without the hint token learning strategy
    -   **RecA/**: Original training code to train models from ProtBERT, ESM_650M, ProtT5, and ESM_15B on the RecA dataset with and without the hint token learning strategy
    -   **TPMT/**: Original training code to train models from ProtBERT, ESM_650M, ProtT5, and ESM_15B on the TPMT dataset with and without the hint token learning strategy

- **📄  environment.yml**
    - Configuration file for environment settings
