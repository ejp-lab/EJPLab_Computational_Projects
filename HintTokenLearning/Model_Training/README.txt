Model_Training/
├── KCNE1/
│   ├── raw/                         # Raw input files for each dataset
│   │   ├── recA.csv
│   │   ├── pten.csv
│   │   ├── tpmt.csv
│   │   └── kcne1.csv
│   ├── processed/                   # Preprocessed & tokenized splits
│   │   ├── RecA/
│   │   │   ├── train.tsv
│   │   │   ├── val.tsv
│   │   │   └── test.tsv
│   │   ├── PTEN/ …                  # Same structure for other datasets
│   │   └── KCNE1/
│   └── splits/                      # Scripts for stratified splitting
│       ├── split_recA.py
│       ├── split_pten_tpmt.py
│       └── split_kcne1.py
├── data/
