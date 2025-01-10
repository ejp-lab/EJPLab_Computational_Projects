# A Rosetta Machine Learning Classification Model for Predicting Sites of Tolerability to Acd: RML_ACD

## About Us
__Authors__: Sam Giannakoulias, Sumant Shringari, John Ferrie, E. James Petersson

Contact us at ejpetersson@sas.upenn.edu

## Set Up
Predict positions which will be tolerant to mutation to Acd 

Operating System: Linux (64-bit)

Programming Langauge: Python This code was specifically written and tested in Python3.7 (python3.7.6)

__Required Python Packages__:

    PyRosetta:    http://www.pyrosetta.org/dow v. 2020.08+release.cb1caba
    biopython:    pip or conda install v. 1.76.1
    numpy:        pip or conda install v. 1.18.1
    scipy:        pip or conda install v. 1.4.1
    scikit-learn: pip or conda install v. 0.22.1

Anaconda Environment: An anaconda environment yml file containing all necessary packages can be found in the anaconda folder. All code was sepcifically tested using packages with versions exactly provided in the yml provided.  

Anaconda installation should take approximately 20 minutes including PyRosetta

## Instruction for Use
Refer to RCSF, ESF, and Machine Learning folders for further instruction

## Datasets
The training and hold-out datasets can be found in Dataset folder. 

## Simulation Runtime
This program is built off of averaged trials of local relaxes in PyRosetta. Each position of interest can be simulated on a basic desktop or laptop computer in approximately 30 minutes
