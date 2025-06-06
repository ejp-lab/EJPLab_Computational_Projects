# Model training instruction

All model‐training code is grouped under the top‐level `Model_Training/` folder. Inside, you’ll find one subfolder per protein dataset—e.g., `KCNE1/`, `PTEN/`, `RecA/`, and `TPMT/`. Each dataset folder follows an identical layout:

```
Model_Training/
├── KCNE1/ ← example shown; PTEN, RecA, TPMT follow the same pattern
│   ├── ESM_15B/ ← backbone A
│   │   ├── HTL/ ← Hint Token Learning experiments
│   │   │   ├── README.txt
│   │   │   ├── models.py
│   │   │   ├── utils.py
│   │   │   └── persequence_ESM15_finetuning_KCNE1_HTL.py
│   │   ├── NO_HTL/ ← control (no hint tokens)
│   │   │   ├── README.txt
│   │   │   ├── models.py
│   │   │   ├── utils.py
│   │   │   └── persequence_ESM15_finetuning_KCNE1_NO_HTL.py
│   ├── ESM_650M/ ← backbone B
│   │   ├── HTL/
│   │   │   ├── README.txt
│   │   │   ├── models.py
│   │   │   ├── utils.py
│   │   │   └── persequence_ESM650_finetuning_KCNE1_HTL.py
│   │   ├── NO_HTL/
│   │   │   ├── README.txt
│   │   │   ├── models.py
│   │   │   ├── utils.py
│   │   │   └── persequence_ESM650_finetuning_KCNE1_NO_HTL.py
│   ├── ProtBERT/ ← backbone C
│   │   ├── HTL/
│   │   │   ├── README.txt
│   │   │   ├── models.py
│   │   │   ├── utils.py
│   │   │   └── persequence_ProtBERT_finetuning_KCNE1_HTL.py
│   │   ├── NO_HTL/
│   │   │   ├── README.txt
│   │   │   ├── models.py
│   │   │   ├── utils.py
│   │   │   └── persequence_ProtBERT_finetuning_KCNE1_NO_HTL.py
│   ├── ProtT5/ ← backbone D
│   │   ├── HTL/
│   │   │   ├── README.txt
│   │   │   ├── models.py
│   │   │   ├── utils.py
│   │   │   └── persequence_ProtT5_finetuning_KCNE1_HTL.py
│   │   ├── NO_HTL/
│   │   │   ├── README.txt
│   │   │   ├── models.py
│   │   │   ├── utils.py
│   │   │   └── persequence_ProtT5_finetuning_KCNE1_NO_HTL.py

```

### Explanations

- **Dataset folders (e.g., `KCNE1/`, `PTEN/`, `RecA/`, `TPMT/`):**  
  Each corresponds to a different protein or experimental dataset. All four follow the same sub‐folder pattern.

- **Backbone directories (`ESM_15B/`, `ESM_650M/`, `ProtBERT/`, `ProtT5/`):**  
  These are the pretrained model families or “backbones” used for fine‐tuning. By separating them at this level, you can run identical pipelines across a variety of architectures.

- **`HTL/` vs. `NO_HTL/`:**  
  - `HTL/` contains code for experiments where **Hint Token Learning** is enabled.  
  - `NO_HTL/` contains the corresponding “control” experiments (same data + architecture, but without hint tokens).  
  Inside each of these, you’ll always find:  
  1. **`README.txt`** 
  2. **`models.py`** – model definitions or network architectures.  
  3. **`utils.py`** – helper functions (data loaders, metrics, etc.).  
  4. **`persequence_<BACKBONE>_finetuning_<DATASET>_<HTL/NO_HTL>.py`** – the actual script you invoke to train/fine‐tune on that dataset under the chosen configuration.

This structure is mirrored exactly for `KCNE1/`, `PTEN/`, `RecA/`, and `TPMT/`, so once you’re familiar with one directory tree, you know where to look for any other dataset’s training code.

### Training Instructions

- Directly run `persequence_<BACKBONE>_finetuning_<DATASET>_<HTL/NO_HTL>.py` in each folder after retrieve the corresponding data

  ``` python persequence_<BACKBONE>_finetuning_<DATASET>_<HTL/NO_HTL>.py ```

