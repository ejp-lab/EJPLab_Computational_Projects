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

# Inputs for calculation of PRE values
big_k = 1.23*10**(-32)       ## Constant for nitroxide radical
tc = 4*10**(-9)              ## correlation time for the electron-nuclear interaction vector
omega = 2*pi*700*10**8       ## 2 pi times larmour frequency of amide proton
R2r = 4                     ## intrinsic transverse relaxation rate estimated from reduced spectra
time = 4*10**(-3)           ## total INEPT evolution time of the HSQC experiment

# Iteratively import structures from the ensemble, extract amide N protons, measure distance to spin position c-alpha, log in array

output_file = 'Calculated_PREs.txt'
nstruct = len(sys.argv) -1
n_aa_struct = parser.get_structure('s', sys.argv[2])
n_aa = len(n_aa_struct[0]['A'])
#PRE_res_list = [20, 24, 42, 62, 85, 87, 103, 120]
PRE_res_list = np.genfromtxt(sys.argv[1], delimiter=',', dtype=str)
for pre_res_idx,pre_res_item in enumerate(PRE_res_list):
	PRE_res_list[pre_res_idx] = pre_res_item.split('.')[0].split('_')[len(pre_res_item.split('.')[0].split('_'))-1]
complete_PRE_matrix = np.empty([n_aa, 2*len(PRE_res_list)])
for PRE_res_num,PRE_res_id in enumerate(PRE_res_list):
#	output_file = 'Calculated_PREs_' + str(PRE_res_id) + '.txt'
	full_PRE_matrix = np.empty([n_aa, nstruct])
	for i in range(2,nstruct,1):
		ensemble_struct = parser.get_structure('s', sys.argv[i])
		ensemble_model = ensemble_struct[0]
		ensemble_n_protons = []
		ensemble_spin_ca = ensemble_model['A'][int(PRE_res_id)]['CA']
		for chain in ensemble_model.get_list():
			for residue in chain.get_list():
				string1 = residue.id
				string2 = residue.resname
				int1 = string1[1]
				if int1 < 2:
					if '1H' in residue:
						ensemble_n_protons.append(residue['1H']) # 1H
					else:
						ensemble_n_protons.append(residue['H']) # H
				elif string2 == 'PRO':
					if 'HA' in residue:
						ensemble_n_protons.append(residue['HA']) # HA
					else:
						ensemble_n_protons.append(residue['N']) # N
				elif int1 >= 2:
					ensemble_n_protons.append(residue['H']) # H
		for j in range(len(ensemble_n_protons)):
			nh = ensemble_n_protons[j]
			distance = np.sqrt(np.square(np.float(nh-ensemble_spin_ca)))
			if distance > 100.0:
				Io_Ir = 1.0
			elif distance < 0.1:
				Io_Ir = 0.0
			else:	
				distance_cm = distance*10**(-8)
				gamma = (big_k/distance_cm**6)*(4*tc+((3*tc)/(1+(omega**2)*(tc**2))))
				Io_Ir = R2r*np.exp(-gamma*time)/(R2r+gamma)
				if np.isinf(Io_Ir) == True or Io_Ir > 1.0:
					Io_Ir = 1.0
			full_PRE_matrix[j][i] = Io_Ir
			if np.isnan(full_PRE_matrix[j][i]) == True:
				print(full_PRE_matrix[j][i])
	for i_idx,i_item in enumerate(full_PRE_matrix):
		for j_idx,j_item in enumerate(i_item):
			if j_item > 1.0:
				full_PRE_matrix[i_idx][j_idx] = 1.0
			if j_item < 0.0:
				full_PRE_matrix[i_idx][j_idx] = 0.0
	average_PRE_matrix = np.average(full_PRE_matrix, axis=1)
	std_PRE_matrix = np.std(full_PRE_matrix, axis=1)
#	np.savetxt(output_file, average_PRE_matrix, fmt='%s', delimiter=' ', newline='\n')
	for k in range(len(average_PRE_matrix)):
		complete_PRE_matrix[k][PRE_res_num*2] = average_PRE_matrix[k]
		complete_PRE_matrix[k][PRE_res_num*2+1] = std_PRE_matrix[k]
np.savetxt(output_file, complete_PRE_matrix, fmt='%s', delimiter=' ', newline='\n')		