##Syntax: run dssp_output_analysis.py length_of_protein dssp_output*.txt

import sys
from numpy import genfromtxt
import numpy as np
import os
from shutil import copy

phi_psi_outfile = 'output_phi_phi.txt'
tco_outfile = 'output_tco.txt'
racc_outfile = 'output_racc.txt'
hbond_outfile = 'output_hbond.txt'
hbond_total_outfile = 'output_hbondtotal.txt'
acc_total_outfile = 'output_acc_total.txt'
phi_psi_2his_outfile = 'output_phi_psi_2his.txt'
phi_psi_2his_no_GLY_outfile = 'output_phi_psi_no_GLY_2his.txt'

import_for_length = genfromtxt(sys.argv[1], delimiter='\t', dtype=float)
length = len(import_for_length)

#Creating Keys for computing relative solvent accessible surface area
#Values obtained from Wilke: Tien et al. 2013 http://dx.doi.org/10.1371/journal.pone.0080635 
aa_acc_max   = { \
        	   'A': 129.0, 'R': 274.0, 'N': 195.0, 'D': 193.0,\
        	   'C': 167.0, 'Q': 225.0, 'E': 223.0, 'G': 104.0,\
        	   'H': 224.0, 'I': 197.0, 'L': 201.0, 'K': 236.0,\
        	   'M': 224.0, 'F': 240.0, 'P': 159.0, 'S': 155.0,\
 	           'T': 172.0, 'W': 285.0, 'Y': 263.0, 'V': 174.0}

#Creating Key for linking each amino acid to a Phi-Psi matrix
ALA = []
ARG = []
ASN = []
ASP = []
CYS = []
GLN = []
GLU = []
GLY = []
HIS = []
ILE = []
LEU = []
LYS = []
MET = []
PHE = []
PRO = []
SER = []
THR = []
TRP = []
TYR = []
VAL = []

aa_phi_mat   = { \
        	   'A': ALA, 'R': ARG, 'N': ASN, 'D': ASP,\
        	   'C': CYS, 'Q': GLN, 'E': GLU, 'G': GLY,\
        	   'H': HIS, 'I': ILE, 'L': LEU, 'K': LYS,\
        	   'M': MET, 'F': PHE, 'P': PRO, 'S': SER,\
 	           'T': THR, 'W': TRP, 'Y': TYR, 'V': VAL}

ALA_2 = []
ARG_2 = []
ASN_2 = []
ASP_2 = []
CYS_2 = []
GLN_2 = []
GLU_2 = []
GLY_2 = []
HIS_2 = []
ILE_2 = []
LEU_2 = []
LYS_2 = []
MET_2 = []
PHE_2 = []
PRO_2 = []
SER_2 = []
THR_2 = []
TRP_2 = []
TYR_2 = []
VAL_2 = []
Full_phi_psi_matrix = [ALA, ALA_2, ARG, ARG_2, ASN, ASN_2, ASP, ASP_2, CYS, CYS_2, GLN, GLN_2, GLU, GLU_2, GLY, GLY_2, HIS, HIS_2, ILE, ILE_2, LEU, LEU_2, LYS, LYS_2, MET, MET_2, PHE, PHE_2, PRO, PRO_2, SER, SER_2, THR, THR_2, TRP, TRP_2, TYR, TYR_2, VAL, VAL_2]
aa_psi_mat   = { \
        	   'A': ALA_2, 'R': ARG_2, 'N': ASN_2, 'D': ASP_2,\
        	   'C': CYS_2, 'Q': GLN_2, 'E': GLU_2, 'G': GLY_2,\
        	   'H': HIS_2, 'I': ILE_2, 'L': LEU_2, 'K': LYS_2,\
        	   'M': MET_2, 'F': PHE_2, 'P': PRO_2, 'S': SER_2,\
 	           'T': THR_2, 'W': TRP_2, 'Y': TYR_2, 'V': VAL_2}
 	           
#Building Matricies for Holding/Analyzing Data
racc_matrix = np.empty([len(sys.argv), int(length)])
tco_matrix = np.empty([len(sys.argv), int(length)])
full_hbonding_matrix = np.empty([len(sys.argv), 14])
total_acc_matrix = []
total_hbond_matrix = []
percent_data_array = np.zeros([length, 3]) # Helix, Sheet, Loop

for fnu,fna in enumerate(sys.argv[2:]):
	lines = open(fna).readlines()
	total_acc_matrix.append(float(lines[7][1:8]))
	total_hbond_matrix.append(float(lines[8][2:6]))
	for idx,item in enumerate(lines[8:22]):
		full_hbonding_matrix[fnu][idx] = int(item[2:6])
	for idx,item in enumerate(lines[28:]):
		res_num = int(item[6:10])
		res_aa = item[13]
		if res_aa == 'X':
			res_aa = 'Y'
		max_for_rel = aa_acc_max[res_aa]
		res_ss = item[16]
		res_acc = float(int(item[35:38]))
		res_rel_acc = res_acc/max_for_rel
		racc_matrix[fnu][idx] = res_rel_acc
		res_tco = float(item[85:92])
		#if res_tco > 0.75:
		#	res_ss = 'H'
		#if res_tco < -0.75:
		#	res_ss = 'E'
		if res_ss == 'E' or res_ss == 'B':
			percent_data_array[idx][1] += 1
		elif res_ss == 'H' or res_ss == 'G' or res_ss == 'I':
			percent_data_array[idx][0] += 1
		else:
			percent_data_array[idx][2] += 1
		tco_matrix[fnu][idx] = res_tco
		res_phi = float(item[103:109])
		aa_phi_mat[res_aa].append(res_phi)
		res_psi = float(item[109:115])
		aa_psi_mat[res_aa].append(res_psi)

#Full_phi_psi_matrix_map = map(None, *Full_phi_psi_matrix)
#pp_out = open(phi_psi_outfile, 'w')
#for i in range(len(Full_phi_psi_matrix_map)):
#	for j in range(len(Full_phi_psi_matrix_map[0])):
#		pp_out.write("%s\t" % Full_phi_psi_matrix_map[i][j])
#	pp_out.write("\n")
#pp_out.close()

full_phi_list = np.empty((0,0))
full_phi_list = np.append(full_phi_list, ALA)
full_phi_list = np.append(full_phi_list, ARG)
full_phi_list = np.append(full_phi_list, ASN)
full_phi_list = np.append(full_phi_list, ASP)
full_phi_list = np.append(full_phi_list, CYS)
full_phi_list = np.append(full_phi_list, GLN)
full_phi_list = np.append(full_phi_list, GLU)
full_phi_list = np.append(full_phi_list, GLY)
full_phi_list = np.append(full_phi_list, HIS)
full_phi_list = np.append(full_phi_list, ILE)
full_phi_list = np.append(full_phi_list, LEU)
full_phi_list = np.append(full_phi_list, LYS)
full_phi_list = np.append(full_phi_list, MET)
full_phi_list = np.append(full_phi_list, PHE)
full_phi_list = np.append(full_phi_list, PRO)
full_phi_list = np.append(full_phi_list, SER)
full_phi_list = np.append(full_phi_list, THR)
full_phi_list = np.append(full_phi_list, TRP)
full_phi_list = np.append(full_phi_list, TYR)
full_phi_list = np.append(full_phi_list, VAL)

full_phi_list_no_GLY = []
full_phi_list_no_GLY = np.append(full_phi_list_no_GLY, ALA)
full_phi_list_no_GLY = np.append(full_phi_list_no_GLY, ARG)
full_phi_list_no_GLY = np.append(full_phi_list_no_GLY, ASN)
full_phi_list_no_GLY = np.append(full_phi_list_no_GLY, ASP)
full_phi_list_no_GLY = np.append(full_phi_list_no_GLY, CYS)
full_phi_list_no_GLY = np.append(full_phi_list_no_GLY, GLN)
full_phi_list_no_GLY = np.append(full_phi_list_no_GLY, GLU)
full_phi_list_no_GLY = np.append(full_phi_list_no_GLY, HIS)
full_phi_list_no_GLY = np.append(full_phi_list_no_GLY, ILE)
full_phi_list_no_GLY = np.append(full_phi_list_no_GLY, LEU)
full_phi_list_no_GLY = np.append(full_phi_list_no_GLY, LYS)
full_phi_list_no_GLY = np.append(full_phi_list_no_GLY, MET)
full_phi_list_no_GLY = np.append(full_phi_list_no_GLY, PHE)
full_phi_list_no_GLY = np.append(full_phi_list_no_GLY, PRO)
full_phi_list_no_GLY = np.append(full_phi_list_no_GLY, SER)
full_phi_list_no_GLY = np.append(full_phi_list_no_GLY, THR)
full_phi_list_no_GLY = np.append(full_phi_list_no_GLY, TRP)
full_phi_list_no_GLY = np.append(full_phi_list_no_GLY, TYR)
full_phi_list_no_GLY = np.append(full_phi_list_no_GLY, VAL)

full_psi_list = []
full_psi_list = np.append(full_psi_list, ALA_2)
full_psi_list = np.append(full_psi_list, ARG_2)
full_psi_list = np.append(full_psi_list, ASN_2)
full_psi_list = np.append(full_psi_list, ASP_2)
full_psi_list = np.append(full_psi_list, CYS_2)
full_psi_list = np.append(full_psi_list, GLN_2)
full_psi_list = np.append(full_psi_list, GLU_2)
full_psi_list = np.append(full_psi_list, GLY_2)
full_psi_list = np.append(full_psi_list, HIS_2)
full_psi_list = np.append(full_psi_list, ILE_2)
full_psi_list = np.append(full_psi_list, LEU_2)
full_psi_list = np.append(full_psi_list, LYS_2)
full_psi_list = np.append(full_psi_list, MET_2)
full_psi_list = np.append(full_psi_list, PHE_2)
full_psi_list = np.append(full_psi_list, PRO_2)
full_psi_list = np.append(full_psi_list, SER_2)
full_psi_list = np.append(full_psi_list, THR_2)
full_psi_list = np.append(full_psi_list, TRP_2)
full_psi_list = np.append(full_psi_list, TYR_2)
full_psi_list = np.append(full_psi_list, VAL_2)

full_psi_list_no_GLY = []
full_psi_list_no_GLY = np.append(full_psi_list_no_GLY, ALA_2)
full_psi_list_no_GLY = np.append(full_psi_list_no_GLY, ARG_2)
full_psi_list_no_GLY = np.append(full_psi_list_no_GLY, ASN_2)
full_psi_list_no_GLY = np.append(full_psi_list_no_GLY, ASP_2)
full_psi_list_no_GLY = np.append(full_psi_list_no_GLY, CYS_2)
full_psi_list_no_GLY = np.append(full_psi_list_no_GLY, GLN_2)
full_psi_list_no_GLY = np.append(full_psi_list_no_GLY, GLU_2)
full_psi_list_no_GLY = np.append(full_psi_list_no_GLY, HIS_2)
full_psi_list_no_GLY = np.append(full_psi_list_no_GLY, ILE_2)
full_psi_list_no_GLY = np.append(full_psi_list_no_GLY, LEU_2)
full_psi_list_no_GLY = np.append(full_psi_list_no_GLY, LYS_2)
full_psi_list_no_GLY = np.append(full_psi_list_no_GLY, MET_2)
full_psi_list_no_GLY = np.append(full_psi_list_no_GLY, PHE_2)
full_psi_list_no_GLY = np.append(full_psi_list_no_GLY, PRO_2)
full_psi_list_no_GLY = np.append(full_psi_list_no_GLY, SER_2)
full_psi_list_no_GLY = np.append(full_psi_list_no_GLY, THR_2)
full_psi_list_no_GLY = np.append(full_psi_list_no_GLY, TRP_2)
full_psi_list_no_GLY = np.append(full_psi_list_no_GLY, TYR_2)
full_psi_list_no_GLY = np.append(full_psi_list_no_GLY, VAL_2)

phi_psi_2his_1, phi_psi_2his_2, phi_psi_2his_3 = np.histogram2d(full_phi_list, full_psi_list, bins=121, range=[[-180,180], [-180,180]])
phi_psi_2his_no_GLY_1, phi_psi_2his_no_GLY_2, phi_psi_2his_no_GLY_3 = np.histogram2d(full_phi_list_no_GLY, full_psi_list_no_GLY, bins=121, range=[[-180,0], [-180,180]])

tam_out = open(acc_total_outfile, 'w')
for i in range(len(total_acc_matrix)):
	tam_out.write("%s\n" % total_acc_matrix[i])
tam_out.close()

thm_out = open(hbond_total_outfile, 'w')
for i in range(len(total_hbond_matrix)):
	thm_out.write("%s\n" % total_hbond_matrix[i])
thm_out.close()

#percent_helix = percent_helix/len(sys.argv[2:])
#percent_sheet = percent_sheet/len(sys.argv[2:])
#percent_loop = percent_loop/len(sys.argv[2:])
#percent_array = [('% Helix --> ', percent_helix), ('% Sheet --> ', percent_sheet), ('% Loop --> ', percent_loop)]
percent_data_array = percent_data_array/len(sys.argv[2:])
np.savetxt('Percent_HEL.txt', percent_data_array, fmt='%s', delimiter=' ', newline='\n')

avg_hbonding_matrix = np.average(full_hbonding_matrix, axis=0)
avg_tco_matrix = np.average(tco_matrix, axis=0)
avg_racc_matrix = np.average(racc_matrix, axis=0)
std_hbonding_matrix = np.std(full_hbonding_matrix, axis=0)
std_tco_matrix = np.std(tco_matrix, axis=0)
std_racc_matrix = np.std(racc_matrix, axis=0)
comb_tco_matrix = np.column_stack((avg_tco_matrix, std_tco_matrix))
comb_racc_matrix = np.column_stack((avg_racc_matrix, std_racc_matrix))
comb_hbonding_matrix = np.column_stack((avg_hbonding_matrix, std_hbonding_matrix))

np.savetxt(tco_outfile, comb_tco_matrix, fmt='%s', delimiter=' ', newline='\n')
np.savetxt(racc_outfile, comb_racc_matrix, fmt='%s', delimiter=' ', newline='\n')
np.savetxt(hbond_outfile, comb_hbonding_matrix, fmt='%s', delimiter=' ', newline='\n')
np.savetxt(phi_psi_2his_outfile, phi_psi_2his_1, fmt='%s', delimiter=' ', newline='\n')
np.savetxt(phi_psi_2his_no_GLY_outfile, phi_psi_2his_no_GLY_1, fmt='%s', delimiter=' ', newline='\n')