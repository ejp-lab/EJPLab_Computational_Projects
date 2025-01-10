# Demo: How to Generate the Complex for Classification

## Preperation of the System
The following demo assumes that the following operations are done in the conda environment gila.
See __Anaconda folder__ for instruction of installing Anaconda and recieving appropriate conda environment.

## Structure Preperation
Rosetta comes with set of tools to prepare structures that we recommend. To access these tools, get a license for rosetta and pyrosetta. They can be found at tools/protein_tools/scripts/. 
### Clean PDB structures
Non-proteanacious atoms must be removed before Rosetta Simulations. 
```
python clean_pdb.py <pdb> <chain id>
```
### Renumber and Rechain PDB structures
Write a simple python script to edit the pdb files such that residues start from 1 and rechain all of the protease atoms to chain A.

### Docking
If the protease you are interested in has a structural complex with a peptide deposited in the PDB, simply trim and mutate the sequence of the experimental peptide to your sequence of interest. If an experimental complex does not exist, use PyRosetta to pose_from_sequence the peptide you are interested in. You can then relax this peptide. Manually place the peptide into the active site of the cleaned protease. At this point, for either method, FlexPepDock can be used to optimze the binding interaction.

```
'/Rosetta/main/source/bin/FlexPepDocking.static.linuxgccrelease -in:file:s ManualDockedComplex.pdb -pre_pack -pep_refine -nstruct 100 -ex1 -ex2aro -out:pdb'
```
For more on FlexPepDock https://www.rosettacommons.org/docs/latest/application_documentation/docking/flex-pep-dock
Select the lowest enerfy complex from FlexPepDock to be relaxed with constraints.

### Relaxing the Structure
The structure must be relaxed with the following options. __Note there is no gurantee that ThioClass will be efficacious if these exact relax protocol options are not used__. The simple relax protocol is performed with constraints, beta16_cart score function, minimizer option of lbfgs_armijo_nonmonotone, minimize bond angles off and dualspace off, and the following fold tree. 

```
foldtree_array = [(active_site_position, 1, -1), 
(active_site_position, protease_end_position, -1), 
(active_site_position, amide_carbon_position, 1),
(amide_carbon_position, ligand_end_position, -1),
(amide_carbon_position, ligand_start_position, -1)]

Serine Proteases:
AtomPair  OG ActiveSiteNum C CleavedSiteNum FLAT_HARMONIC 2.75 1.0 1.0	

Cysteine Proteases:
AtomPair  SG ActiveSiteNum C CleavedSiteNum FLAT_HARMONIC 3.0 1.0 1.0
```

## Running Thio_Class Predictor

In order to run Thio_Class, you will unfortunately have to have an additional version of pyrosetta to the one within the Thio_Class virtual env. Or at minimum, you must have a pyrosetta database where you can edit the following files to contain the TS atom.

Path=’~/pyrosetta/database/chemical/atom_type_sets/fa_standard_genpot/’
* Path + atom_properties.txt
* Path + extras/ + /soft_rep_params.txt
* Path + extras/ + gen_born_params.txt
* Path + extras/ + gen_kirkwood_params.txt
* Path + extras/ + sasa_radii_legacy.txt
* Path + extras/ + NACCESS_sasa_radii.txt
* Path + extras/ + reduce_sasa_radii.txt
* Path + extras/ + memb_fa_params.txt
* Path + extras/ + std_charges.txt
* Path + extras/ + atom_orbital_hybridization.txt

Note the patch files and binary must all be in the same directory. If using a NCAA, the params and rotlib must also be in the same directory.

```
-pdb Your complex of interest, required.
-data Your posiitons of interest, required. See example for exact formatting.
-as_res Active Site residue number, required.
-cleave_res THe celaved residue number, required.
-m Model, required.
-s Scaler, required.
-d Path to pyrosetta database, require.
-params NCAA params file, NOT REQUIRED.
-ncaa_3 3 letter code for NCAA, NOT REQUIRED
```
Sample command
```
./Thio_Class_MasterScript -pdb input.pdb -data SampleData.csv -as_res some_number -cleave_res some_other_number -m Thio_Class_Training.sav -s Thio_Class_Scaled_Training.sav -d ~/pyrosetta/database/ -params mcm.params -ncaa_3 MCM
```

