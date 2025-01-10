# Analysis Scripts for FastFloppyTail Structural Ensembles
These scripts allow various parameters to be computed from structural ensembles generated from FastFloppyTail.

## Script List
 
#### Distance_Extraction.py

_Requires pdb_assembler.py and polymer_analysis runs._ 

This script takes an interresidue average/standard deviation distance maps and isolated the values
for residues of interest (example INPUT_RESIDUE_PAIR_SET.cst found in Data_Compiled/asyn_distance.cst)

```run Distance_Extraction.py INPUT_RESIDUE_PAIR_SET.cst DISTANCES.out COMPILED.dm COMPILED.std```

#### dssp_output_analysis.py

_Requires Process_DSSP.py, pdb_assembler.py and polymer_analysis runs._

This script takes a set of DSSP outputs from _Process_DSSP.py_ and compiles the data into a single file

```run dssp_output_analysis.py COMPILED.nu DSSP_OUTPUT_*.txt```

#### EFRETs_from_Ensembles.py

This script takes a set of outputs from FastFloppyTail and computes the average FRET for given FRET pairs (example FRET_SET.txt found in Data_Compiled/asyn_efret.txt)

```run EFRETs_from_Ensembles.py FRET_SET.txt FASTFLOPPYTAIL_OUTPUTS_*.pdb```

#### Full_IDP_Analysis_Script_FastFloppyTail.py

This script is capable of running all analyses herein in a parallelized fashion for use on a CPU-based cluster. Requires manual rewrite of FastFloppyTail output
file names in script to run. 

```run Full_IDP_Analysis_Script_FastFloppyTail.py```

#### J_Couplings.py

_Requires Process_DSSP.py to run._

This script takes a set of DSSP outputs and computes J-coupling values based on input data (found in lines 90-94 and Data_Compiled)

```run J_Couplings.py DSSP_OUTPUT_*.txt```

#### make_PALES_input.py

This script creates the necessary input file required for comparative PALES analysis of FastFloppyTail structures from an input fasta.

```run make_PALES_input.py PROTEIN_SEQUENCE.fasta```

#### PALES_Analysis_Parallel_bestFit.py

_Requires Process_PALES_bestFit.py run_

This script take a directory that contains the outputs from PALES (Residual Dipolar Coupling, RDC) analysis of a set of FastFloppyTail
outputs and compiles the results into an Average RDC file and a file containing the RMSD and Q-value compared to experimental data

```run PALES_Analysis_Parallel_bestFit.py PALES_OUTPUT_DIRECTORY/```

#### pdb_assembler.py

This script takes a set of PBDs (generally with the same base name) and compiles them into a single PDB file

```run pdb_assembler.py OUTPUT_COMPILED.pdb INPUT_PDB_NAME_*.pdb```

#### polymer_analysis.py

This scripts takes an input compiled PDB files (obtained from _pdb_assembler.py_) and generates an interresidue contact probability map(.cm, cutoff distance specified in line 12, Default 25 Angstroms), 
an interresidue average distance map (.dm) and associated standard deviations (.std), a polymer scaling plot (.nu) and associated standard deviation (.nus) and computes the radius of gyration (.rg) for 
the set of structures.

```run polymer_analysis.py COMPILED.pdb```

#### PRE_Data_Comparison.py

This script takes the output from _PREs_from_Ensembles.py_ and compares the simulated values to input experiments (example PRE_DATA.pre found in Data_Compiled/asyn.pre).

```run PRE_Data_Comparison.py PRE_DATA.pre```

#### PREs_from_Ensembles.py

This script takes a set of outputs from FastFloppyTail and computes the average I/Io for given PRE probe sites (example PRE_DATA.pre found in Data_Compiled/asyn.pre).

```run PREs_from_Ensembles.py PRE_DATA.pre FASTFLOPPYTAIL_OUTPUTS_*.pdb```

#### Process_DSSP.py

This script takes a set of outputs from FastFloppyTail and computes various parameters through DSSP

```run Process_DSSP.py FASTFLOPPYTAIL_OUTPUTS_*.pdb```

#### Process_PALES_bestFit.py

This script takes a set of outputs from FastFloppyTail and predicts RDC values using the [PALES](https://www3.mpibpc.mpg.de/groups/zweckstetter/_links/software_pales.htm#HuP) software based on a PALES input file 
(example PALES.in can be found in Data_Compiled/asyn_PALES.in) which can be written with experimental data and _make_PALES_input.py_. 

```run Process_PALES_bestFit.py PALES.in FASTFLOPPYTAIL_OUTPUTS_*.pdb```

_Process_PALES_bestFit_Parallel.py performs same compute in parallelized fashion._

#### Process_SPARTA.py

This script takes a set of outputs from FastFloppyTail and predicts chemical shifts using the [SPARTA+](https://spin.niddk.nih.gov/bax/software/SPARTA+/) software

```run Process_SPARTA.py FASTFLOPPYTAIL_OUTPUTS_*.pdb```

_Process_Sparta_Parallel.py performs same compute in parallelized fashion._

#### Sparta_Analysis_Post_Process_2.py

_Requires Process_SPARTA_Parallel.py run_

This script takes as a set of SPARTA+ chemical shift predictions from a set of FastFloppyTail structurs and delivers a average chemical shifts and RMSD values from experiment
CHEMICAL_SHIFTS data (examples of CHEMICAL_SHIFTS found in Data_Compiled/asyn_chemical_shifts.nmr)

```run Sparta_Analysis_Post_Process_2.py CHEMICAL_SHIFTS.nmr SPARTA_PRED_OUTPUTS_*.txt```
