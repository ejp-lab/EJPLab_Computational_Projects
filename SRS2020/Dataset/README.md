# Datasets
## Dataset ID and PDB Numbering
The datasets are id based on complexes that have been renumbered to be from 1 to length of the pdb and that have been rechained to be from A, B, C, D, etc. This is common when doing rosetta simulations. However, we understand if others utilize other naming schemes for pdbs. To translate this, we provide a csv that indicates the mapping from the original pdb numbering to Skempi clean pdb numbering and our pose (Rosetta) numbering.

Note that these csvs are seperated by a semicolon (;) as the original SKEMPI database v2 were. Officelibre and excel have options to view semicolon delimited data. For pythonic way to view data sets, simply use pandas.
## Training and Testing Sets
The training and testing datasets used for SRS2020ddGPredictor
