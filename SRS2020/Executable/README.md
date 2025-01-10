# Binary to Run SRS2020

__This Page is outdated look at the Demo Folder for Most Updated Files and Versions__
## Accessing executable
Due to size limit restriction of github, the executable is available for freely at the given google drive link.
https://bit.ly/2Qv5iEq

## To run the executable
It is important that the naming scheme for the resfile is followed. The resfile name must start with the pdb name then followed by underscore with original amino acid, its position, and the mutated amino acid. 

For example, the mutation of residue E 287 on pdb 1A4Y to A 287. The resfile will be named __1A4Y_E287A.resfile__. If one wishes to do multiple mutation, then string the mutations with underscore seperating each other. The resfile will be named __1A4Y_K500G_Y434A_Y437A.resfile__.
```
.\SRS2020_ddG_Predictor -r PDB_OriginalaminoacidResiduenumberMutatationaminoacid.resfile -p pdb_relaxed.pdb -db PATHTOROSETTADATABASE
```
## Executable Options
1. -r Resfile (required)
2. -p Pdb (required)
3. -db Database Path (required)
4. -m Model (optional)
5. -scal Scaler (optional)
