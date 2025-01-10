# AbInitioVO and FastFloppyTail
 AbInitio Variable Order and FastFloppyTail Algorithms for Protein Structure Prediction from Sequence

A manuscript detailing the methods and demonstrating the utility of each of these algorithms can be found on [bioRxiv](https://doi.org/10.1101/2020.01.30.925636).  
## Algorithm Descritions
### AbInitioVO
The AbInitio Variable Order (AbInitioVO) PyRosetta algorithm is an improvement of the [AbInitio-Relax](https://www.rosettacommons.org/docs/latest/application_documentation/structure_prediction/abinitio-relax)
algorithm in Rosetta which features changes to the fragment selection process and structure scoring. Disordered probability predictions based on protein sequence are used as an additional inputs alongside secondary
structure predictions to influence 9-mer and 3-mer [fragment picking](https://www.rosettacommons.org/docs/latest/application_documentation/utilities/app-fragment-picker). Furthermore, by making improvements to the 
[centroid score terms](https://www.rosettacommons.org/docs/latest/rosetta_basics/scoring/centroid-score-terms), specifically the ""rg"" term, to accomadate disordered proteins this algorithm is able to accurately 
predict the structures of ordered domains, of both fully ordered and partially ordered proteins, as well as identifying disordered regions.

### FastFloppyTail
FastFloppyTail is an improved variant of the [FloppyTail](https://www.rosettacommons.org/docs/latest/application_documentation/structure_prediction/floppy-tail) Rosetta algorithm written for PyRosetta. The algorithm features 
a reduction in sidechain packing moves along with an increase in the fequency with which sampled structures are returned to their minima. Overall this algorithm boasts both a marked increase in accuracy compared to experimental 
data along with a 10-fold reduction in compute time compared to FloppyTail.

## Installation Guide
__Operating System:__ Linux (64-bit)

__Programming Langauge:__ Python
This code was specifically written and tested in Python3.6 (python3.6.8)
	
__Required Python Packages:__
- PyRosetta
	- This was specifically written and tested with PyRosetta 2019.17+release.2cb3f3a py36_0. Additional information can be found on the [PyRosetta](http://www.pyrosetta.org/) website. This specific PyRosetta build can be downloaded [here](http://www.pyrosetta.org/dow) after obtaining a [license](https://els.comotion.uw.edu/express_license_technologies/pyrosetta)
- biopython (1.73)
- numpy (1.14.5)
- scipy (1.1.0)

__Anaconda Environment:__
An anaconda environment containing all necessary packages can be found in the anaconda folder. Build time for this Anaconda environment takes on the order of mintues to hours depending on the number of processors used and is largely dependent on the PyRosetta build time. On a normal computer this is expected to take ~30 minutes. With this file you can generate a local version of this environment using the command:

```conda env create -f lion.yml```

__Additional Reccommended Software Packages:__
- Blast (blast-2.2.26)
- [Rosetta](https://www.rosettacommons.org/software) (code herein was tested with rosetta3.9)
- [DSSP](https://swift.cmbi.umcn.nl/gv/dssp/) (installed via [conda](https://anaconda.org/salilab/dssp)) 
- [Sparta+](https://spin.niddk.nih.gov/bax/software/SPARTA+/)
- [PALES](https://www3.mpibpc.mpg.de/groups/zweckstetter/_links/software_pales.htm#HuP)

## Simulation Times
All times reported herein were from 140-residue alpha-Synuclein on a normal computer. The previously reported FloppyTail simulation method took ~30 minutes per structure whereas the FastFloppyTail simulation method took 
~3 minutes per structure. The AbInitio and AbInitioVO algorithms both required ~45 minutes per structure and Relaxes took ~3 minutes a structure. The "Generalized Simulations" can be found in the "Testing Rosetta Parameters" folder and 
on average, Full-atom Generalized Simulations required ~3 hours per structure while simulations using exclusively Centroid coarse-graining required 
~30 minutes per structure. Simulations using both coarse-grained and all-atom molecular representations (SimAnn) took ~1.5 hours per structure. Unless otherwise noted, all other simulation/analyses develop herein take less than 5 minutes to run.

## Running AbInitioVO
AbInitioVO is run using the command-line, where inputs are specified using an arguement parser. An example run which can also be found in the demo section is:
```
run AbInitioVO.py -in asyn.fasta.txt -abinitiovo - abnstruct 5 -diso asyn.diso -t_frag asyn_dcor_frags.200.3mers -n_frag asyn_dcor_frags.200.9mers
```
Each of the parser flags are described below:
```
-in  --Input_FASTA_File  Name of the text file containing the FASTA sequence of the protein of interest. Carot should not be in same line as sequence, UniProt format preferred.
-abnstruct  --Number_of_AbInitio_Structures  Number of structures to sample during Centroid AbInitio portion. Default = 1000
-diso  --Input_DisoPred_File  Name of the file containing the per residue disordered probability prediction generated by RaptorX
-t_frag  --Three_Mer_Frag_Library  Name of the file containing the three-mer fragment library generated by disorder corrected method
-n_frag  --Nine_Mer_Frag_Library  Name of the file containing the nine-mer fragment library generated by disorder corrected method
-rsd_wt_helix  --Residue_Weight_Helix  Reweights the env, pair and cbeta scores for helical residues by specificied factor. Default: 0.5
-rsd_wt_loop  --Residue_Weight_Loop  Reweights the env, pair and cbeta scores for loop residues by specificied factor. Default: 0.5
-rsd_wt_sheet  --Residue_Weight_Sheet  Reweights the env, pair and cbeta scores for sheet residues by specificied factor. Default: 1.0
-rg_weight  --Rg_Weight  Reweights the weight of rg in each step by specificied factor. Default: 0.5
-cycles  --AbInitio_Cycles  Number of sampling cycles within each stage of AbInitio, identical to increase_cycles flag in C++ ClassicAbInitio and AbRelax. Default 10
-abinitiovo  --AbInitioVO  Boolean for running protocol with AbInitioVO score function, runs with flag
-refinesubset  --Refine_Subset  Only subjects the lowest X% of structures to Relax refinement where X is specified as input following flag. Default 100
-relnstruct  --Number_of_Relax_Structures  Number of independent full-atom Relax sampling trajectories from a single AbInitio structure. Default 50
```

### Acquiring the Necessary Inputs
All of the files (FASTA, Disordered Probability Prediction and Fragment Libraries) can all be obtained from the protein sequence as detailed below
#### Disordered Probability Prediction
Disordered probability predictions can be obtained using [RaptorX Property](http://raptorx.uchicago.edu/StructurePropertyPred/predict/), [PsiPred DISOPRED3](http://bioinf.cs.ucl.ac.uk/psipred/) or other servers of your choice.
#### Fragment Library Construction
##### Secondary Structure Prediction
Secondary structure predictions can be obtained using [RaptorX Property](http://raptorx.uchicago.edu/StructurePropertyPred/predict/), [PsiPred PSIPRED 4.0](http://bioinf.cs.ucl.ac.uk/psipred/) or other servers of your choice.
After obtaining these prediction files, the script _Diso_SS2_Reweight_Opt_FFT.py_ can be used to generate secondary structure predictions files that are reweighted based on disordered probabilities and vice-versa.

- To generate a reweighted secondary structure prediction based on a disordered probability prediction:

	```run Diso_SS2_Reweight_Opt_FFT.py -ssin INPUT_SS2_PREDICTION_FILE -sstype 4_LETTER_CODE -disoin INPUT_DISORDER_PREDICTION_FILE -out REWEIGHTED_SS2_PREDICTION_OUTPUT_FILE```
	
- To generate a reweighted disordered probability based on the secondary structure prediction, add the following flag to the above run command:

	```-disoout REWEIGHTED_DISORDER_PREDICTION_OUTPUT_FILE```
	
Currently this code only supports inputs from disodered probability predictions from the servers named above (which take on the same format) and secondary structure predictions from those named above and [Jufo](http://www.meilerlab.org/index.php/servers/show?s_id=5),
which all vary in format. The currently recognized 4-letter codes to specify the input file format are rapx=RaptorX, ppred=PsiPred and jufo=Jufo. All secondary structure outputs follow the PSIPRED format rquired by the Rosetta [FragmentPicker](https://www.rosettacommons.org/docs/latest/application_documentation/utilities/app-fragment-picker) application. Lastly, by supplying an appropriately 
formatted disordered prediction file with the probability of each residue set to zero, this script can also be used to non-PsiPred secondary structures prediction in the the appropriate formats recognized by Rosetta.

###### Reweighting Methods
Currently, there have been two approaches used to reweight secondary structure predictions. The first, termed _best-reweighting_, uses exclusively the disordered probability 
prediction file from [RaptorX Property](http://raptorx.uchicago.edu/StructurePropertyPred/predict/) to reweight secondary structure predictions from all other servers. RaptorX was 
selected because it was the sole server to prediction that the alpha-synuclein protein was fully-disordered. Alternatively, one can employ the _self-reweighting_ scheme where all 
servers secondary structure predictions are reweighted based on disordered predictions from those available on the same server software. Both of these methods have shown efficacy in 
under different condtions.

##### The Rosetta Fragment Picker Application
The [Fragment Picker](https://www.rosettacommons.org/docs/latest/application_documentation/utilities/app-fragment-picker) application is available within the [Rosetta Modeling Suite](https://www.rosettacommons.org/software) and is used to generate 
3-mer and 9-mer fragment libraries from secondary structure prediction files. We recommend using the _best_ protocol for selecting fragments for AbInitioVO and the _quota_ protocol for selecting fragments for 
FastFloppyTail. Respectively, these produce the most likely fragments and a broad array of possible fragments, which accurately services these two algorithms. Examples of input files necessary for each of these can be found in the demo folder. Note 
the location of various files or the vall fragment library may need to be changed to the correct locations on your system. Once an appropriate flags file containing the neccessary inputs for the Fragment Picker has been generated, the application can be run 
using the command:

``` /YOUR_ROSETTA_LOCATION/rosetta/rosetta3.9/main/source/bin/fragment_picker.static.linuxgccrelease @asyn.best.flags```

###### Sequence Profile from PSI-Blast
An optional input for the Fragment Picker application is a sequence profile generate with PSI-BLAST. This requires that you download a local version of a sequence database (the [non-redundant Blast database](https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=Download) is preferred) 
along with the appropriate version of BLAST specified above. Lastly, the PERL script found in the _BLAST_ directory within the _Fragment_Picker_ directory can be used to generate the sequence profile .chk file used by the Fragment Picker, however you will likely need to update line 33 which specifies 
location of your protein sequence directory:

```./Blast_for_Checkpoint_File.pl PROTEIN_SEQUENCE.fasta```

### Analysis
The analysis script _AbInitio_Analysis.py_ has been provided which performs very basic analysis of the output structures and information and can be run via the command:

```run AbInitio_Analysis.py SINGLE_LETTER_CODE START_RESIDUE END_RESIDUE PDB_FOR_COMPARISON.pdb OUTPUT_LOCATION(S)_*```

An example for the files found in the demo is:

```run AbInitio_Analysis.py D 1 140 ../asyn_STD.pdb Abasyn_*/```

The one letter codes (O=Ordered, P=Partially Ordered and D=Disordered) specify whether the RMSD is computed with respect to the specified pdb (O), with respect to a segment of the specified pdb and the corresponding segment of the predicted structures (P) or 
with respect to the lowest energy structure (D).

## Running FastFloppyTail
FastFloppyTail is run using the command-line, where inputs are specified using an arguement parser. An example run which can also be found in the demo section is:

```run FastFloppyTail.py -in asyn.fasta.txt -ftnstruct 500 -t_frag asyn_dcor_frags.200.3mers```

Note that the same inputs described above are used by the FastFloppyTail algorithm. Each of the parser flags is described below:
```
-in  --Input_FASTA_File'  Name of the text file containing the FASTA sequence of the protein of interest. Carot should not be in same line as sequence, UniProt format preferred.
-ftnstruct  --Number_of_FloppyTail_Structures  Number of structures to sample during FloppyTail portion. Default = 400
-t_frag  --Three_Mer_Frag_Library  Name of the file containing the three-mer fragment library generated by disorder corrected method
-cycles  --FloppyTail_Cycles  Number of sampling cycles within each stage of FloppyTail, identical to increase_cycles flag in C++ ClassicAbInitio and AbRelax. Default 0
-refinesubset  --Refine_Subset  Only subjects the lowest X% of structures to Relax refinement where X is specified as input following flag. Default 0
-relnstruct  --Number_of_Relax_Structures  Number of independent full-atom Relax sampling trajectories from a single AbInitio structure. Default 1
-diso  --Disorder_Probability_Prediction_File  File containing per residue disorder probability prediction in RaptorX format. Generally acquired from RaptorX prediciton
-inpdb  --Input_PDB_File  Name of the text file containing the PDB structure of the protein of interest. All residues are required, missing residues are not constructed
-code  --Order_Code  Single letter code specifying O = ordered, D = disordered, P = partially ordered. If D is supplied, fasta is used, if P is supplied PDB is used. NOTE THIS HAS NOT BEEN EXTENSIVELY TESTED.
```
After the initial structure generation is performed using _FastFloppyTail.py_, the loweset energy outputs can be selected using the _FloppyTail_Analysis.py_ script, which automatically selects the 1000 lowest energy conformers.

```run FloppyTail_Analysis.py D 1 140 ../asyn_STD.pdb FFT_asyn_*/```

The input structure mirrors that of _AbInitio_Analysis.py_ described above. After selecting the 1000 lowest energy structures, they can be further refined using the [FastRelax](https://www.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Movers/movers_pages/FastRelaxMover) algorithm and can be run via the command:

```run FloppyTail_Relax.py```

### Analysis
Within the _Analysis_Scripts_ directory there are a number of scripts that are capable of compuing a variety of parameters for comparison to experiments. The _README.md_ found inside that directory 
contains information about what is required and generated for each analysis. 

## Testing Rosetta Parameters
The directory _Testing_Rosetta_Parameters_ contains a set of Python/PyRosetta scripts which were used to test which parameters of Rosetta hindered accurate prediction of disordered proteins.
All of these are pre-programmed to simulated alpha-synuclein and can be run via 
```run SIMULATION_NAME.py```

## Cluster Submit Scripts
All of the scripts placed in the main directory are designed to run on a CPU-based cluster for parallel computing. The _Cluster_Submit_Scirpts_Directory_ contains sample submit scripts for most of the code contained in this project.
