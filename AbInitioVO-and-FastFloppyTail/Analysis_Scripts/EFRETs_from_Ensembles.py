# Script which takes native structure and output .pdb files from 
# Syntax: run Avg_RMSD.py Spin_Label_Position input_structures.pdb 

import math
from math import exp, log, pi, sqrt
from random import random as rnd
from scipy import *
import sys
from Bio.PDB import *
import numpy as np

# Import PDB functionalities from Biopython
parser = PDBParser()
io = PDBIO()
sup = Superimposer()

# Inputs for calculation of EFRET values


# Iteratively import structures from the ensemble, extract amide N protons, measure distance to spin position c-alpha, log in array
output_file = 'Calculated_EFRETs.txt'
output_file_2 = 'Calculated_EFRETs_STD.txt'
constraints = sys.argv[1]
cst_file = open(constraints).readlines()

residue_a_matrix = []
residue_b_matrix = []
R0_matrix = []
exp_efret_matix = []

for line in cst_file:
	if len(line) > 0:
		words = line.split()
		residue_a = int(words[0])
		residue_a_matrix.append(residue_a)
		residue_b = int(words[1])
		residue_b_matrix.append(residue_b)
		R0_val = float(words[2])
		R0_matrix.append(R0_val)
		exp_val = float(words[3])
		exp_efret_matix.append(exp_val)
		
nstruct = len(sys.argv)
n_FRET_pairs = len(residue_a_matrix)
full_EFRET_matrix = np.zeros([n_FRET_pairs, nstruct])
full_SUM_matrix = np.empty([n_FRET_pairs,1])
full_SUMS_matrix = np.empty([n_FRET_pairs,1])


for k in range(len(residue_a_matrix)):		
	for i in range(2,nstruct,1):
		ensemble_struct = parser.get_structure('s', sys.argv[i])
		ensemble_model = ensemble_struct[0]
		ensemble_acceptor_ca = ensemble_model['A'][residue_a_matrix[k]]['CA']
		ensemble_donor_ca = ensemble_model['A'][residue_b_matrix[k]]['CA']
		distance = sqrt((ensemble_acceptor_ca-ensemble_donor_ca)**2)
		print(distance)
		R0_exp = R0_matrix[k]
		EFRET = 1/(1+(distance/R0_exp)**6)
		print(EFRET)
		full_EFRET_matrix[k][i]= EFRET
		
full_AVG_matrix = np.average(full_EFRET_matrix, axis=1)

full_STD_matrix = np.std(full_EFRET_matrix, axis=1)

np.savetxt(output_file, full_AVG_matrix, fmt='%s', delimiter=' ', newline='\n')
np.savetxt(output_file_2, full_STD_matrix, fmt='%s', delimiter=' ', newline='\n')

# Compute RMSDs
for fret_line_idx, fret_line_item in enumerate(full_AVG_matrix):
	full_AVG_matrix[fret_line_idx] = (fret_line_item - exp_efret_matix[fret_line_idx])**2

full_rmsd = [np.sqrt(np.average(full_AVG_matrix))]
np.savetxt('EFRET_RMSD.txt', full_rmsd, fmt='%s')