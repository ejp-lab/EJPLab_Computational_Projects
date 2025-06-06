# Model training instruction

All model‐training code is grouped under the top‐level `Model_Training/` folder. Inside, you’ll find one subfolder per protein dataset—e.g., `KCNE1/`, `PTEN/`, `RecA/`, and `TPMT/`. Each dataset folder follows an identical layout:

```text
Model_Training/
├── KCNE1/← example shown; PTEN, RecA, TPMT follow the same pattern
│   ├── ESM_15B/
│   │   ├── HTL/ ← **“Hint Token Learning” experiments**
│   │   │   ├── README.txt
│   │   │   ├── models.py
│   │   │   ├── utils.py
│   │   │   └── persequence_ESM15_finetuning_KCNE1_HTL.py
│   │   ├── NO_HTL/
│   │   │   ├── README.txt
│   │   │   ├── models.py
│   │   │   ├── utils.py
│   │   │   └── persequence_ESM15_finetuning_KCNE1_NO_HTL.py
│   ├── ESM_650M/
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
│   ├── ProtBERT/
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
│   ├── ProtT5/
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
