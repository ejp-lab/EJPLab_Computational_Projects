# Demo for AbInitioVO and FastFloppyTail
 AbInitio Variable Order and FastFloppyTail Algorithms for Protein Structure Prediction from Sequence

## Contents
This directory contains versions of each script that can run on a single node and can produce structural predictions from 
AbInitioVO and structural ensembles from FastFloppyTail. Alpha-Synuclein is used as an example where re-weighted secondary 
structure prediction files have already been prepared.

## Run Commands
### AbInitioVO
```
run AbInitioVO.py -in asyn.fasta.txt -abinitiovo - abnstruct 5 -diso asyn.diso -t_frag asyn_dcor_frags.200.3mers -n_frag asyn_dcor_frags.200.9mers
```

### FastFloppyTail

```run FastFloppyTail.py -in asyn.fasta.txt -ftnstruct 500 -t_frag asyn_dcor_frags.200.3mers```

```run FloppyTail_Analysis.py D 1 140 ./asyn_STD.pdb ./```

```run FloppyTail_Relax.py```
