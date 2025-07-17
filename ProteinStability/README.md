# Protein Stability Prediction Models
## Overview
Predicting how amino‑acid substitutions affect protein stability is a fundamental problem that impacts enzyme engineering, drug discovery and our basic understanding of folding thermodynamics.  This repository collects end‑to‑end workflows—from data curation to model training and evaluation—for three complementary stability prediction tasks.

| Task | Experimental proxy for stability | Folder |
|------|----------------------------------|--------|
| **GFP** | Fluorescence intensity of Green Fluorescent Protein variants (higher fluorescence ≈ greater fold stability) | `GFP/` |
| **dG** | Absolute folding free energy (ΔG) in kcal mol⁻¹ for soluble, single‑chain proteins | `dG/` |
| **ddG** | Change in binding free energy (ΔΔG) of protein–protein complexes upon mutation | `ddG/` |


## Repository Structure
- **📂 EDA**
  - Contains the initial data exploration of dG datasets
  - Helps in understanding data distributions and key features
 
- **📂 GFP, dG, ddG**
  - Each folder contains:
    - **train.joblib**: the dataset for model training
    - **val.joblib**: the dataset for model validation
    - **test.joblib**: the dataset for model testing
    - **persequence_classfication_t5_finetuning.py**: training code
