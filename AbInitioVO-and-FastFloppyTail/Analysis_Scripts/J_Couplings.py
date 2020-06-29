## Syntax of code: run J_Coupling_Calculations.py input_dssp_outputs_*.txt
## Importing various Functionalities
import sys
from numpy import genfromtxt
import numpy as np
import os
from shutil import copy

## Making Arrays for holding data
num_of_inputs = len(sys.argv[1:])
J3HNHA_coupling_array = np.zeros([num_of_inputs,140])
J1CaHa_coupling_array = np.zeros([num_of_inputs,140])
J1CaN_coupling_array = np.zeros([num_of_inputs,140])
J2CaN_coupling_array = np.zeros([num_of_inputs,140])
J3CC_coupling_array = np.zeros([num_of_inputs,140])
J_coupling_array = np.zeros([140,8])

## Computing 3JHN-HA
#J3HNHA = 7.97*((np.cos(np.radians(phi-60)))**2)-1.26*np.cos(np.radians(phi-60))+0.63
## Computing 1JCaHa
#J1CaHa = 140.3+1.4*np.sin(np.radians(psi+138))-4.1*((np.cos(np.radians(psi+138)))**2)+2.0*((np.cos(np.radians(phi+30)))**2)
## Computing 1JCaN
#J1CaN = 2.8484*((np.cos(np.radians(psi)))**2)-1.2129*np.cos(np.radians(psi))+8.6453
## Computing 2JCaN
#J2CaN = -0.66*((np.cos(np.radians(psi)))**2)-1.51*np.cos(np.radians(psi))+Cconstant
#Cconstant = 7.65 # Val,Thr,Ile,Ser
#Cconstant = 8.15 # All other amino acids
aa_Aconstant_val   = { \
        	   'A': 142.0, 'R': 141.5, 'N': 141.5, 'D': 142.5,\
        	   'C': 140.3, 'Q': 141.1, 'E': 141.9, 'G': 104.0,\
        	   'H': 143.8, 'I': 143.8, 'L': 141.1, 'K': 141.5,\
        	   'M': 142.2, 'F': 142.9, 'P': 148.4, 'S': 142.1,\
 	           'T': 141.4, 'W': 143.0, 'Y': 143.0, 'V': 141.3}
 	           
psi_1 = 0
phi_1 = 0
## Extracting Phi/Psi and Computing Couplings
for fnu,fna in enumerate(sys.argv[1:]):
	lines = open(fna).readlines()
	for idx,item in enumerate(lines[28:]):
		res_aa = item[13]
		if res_aa == 'X':
			res_aa = 'Y'
		phi = float(item[103:109])
		psi = float(item[109:115])
		J3HNHA = 7.97*((np.cos(np.radians(phi-60)))**2)-1.26*np.cos(np.radians(phi-60))+0.63
		Aconstant = aa_Aconstant_val[res_aa]
		J1CaHa = Aconstant+(1.4*np.sin(np.radians(psi+138)))-(4.1*((np.cos(2*np.radians(psi+138)))))+(1.7*((np.cos(2*np.radians(phi+30)))))
		J1CaN = 1.7040*((np.cos(np.radians(psi)))**2)-0.9799*np.cos(np.radians(psi))+9.5098
		if res_aa is 'V' or 'T' or 'I' or 'S':
			Cconstant = 7.65
		else:
			Cconstant = 8.15
		J2CaN = -(0.2047*((np.cos(np.radians(psi_1)))**2))-(1.5176*np.cos(np.radians(psi_1)))+Cconstant
		if phi < 0.0:
			J3CC = 1.78*((np.cos(np.radians(phi)))**2)-(0.95*np.cos(np.radians(phi)))+0.46
		else:
			J3CC = -100
		J3HNHA_coupling_array[fnu][idx] = J3HNHA
		J1CaHa_coupling_array[fnu][idx] = J1CaHa
		J1CaN_coupling_array[fnu][idx] =  J1CaN
		J2CaN_coupling_array[fnu][idx] = J2CaN
		J3CC_coupling_array[fnu][idx] = J3CC
		psi_1 = psi
		phi_1 = phi
		
## Removing Bad J3CC values
J3CC_coupling_avg = np.average(J3CC_coupling_array, axis=0)
J3CC_coupling_std = np.std(J3CC_coupling_array, axis=0)

for J3_res_idx in range(len(J3CC_coupling_array[0])):
	temp_J3_array = []
	for J3_pdb_idx in range(len(J3CC_coupling_array)):
		if J3CC_coupling_array[J3_pdb_idx][J3_res_idx] != -100:
				temp_J3_array.append(J3CC_coupling_array[J3_pdb_idx][J3_res_idx])
	J3CC_coupling_avg[J3_res_idx] = np.average(temp_J3_array)		
	J3CC_coupling_std[J3_res_idx] = np.std(temp_J3_array)
			
## Computing Averages and Standard Deviations
J3HNHA_coupling_avg = np.average(J3HNHA_coupling_array, axis=0)
J1CaHa_coupling_avg = np.average(J1CaHa_coupling_array, axis=0)
J1CaN_coupling_avg = np.average(J1CaN_coupling_array, axis=0)
J2CaN_coupling_avg = np.average(J2CaN_coupling_array, axis=0)

J3HNHA_coupling_std = np.std(J3HNHA_coupling_array, axis=0)
J1CaHa_coupling_std = np.std(J1CaHa_coupling_array, axis=0)
J1CaN_coupling_std = np.std(J1CaN_coupling_array, axis=0)
J2CaN_coupling_std = np.std(J2CaN_coupling_array, axis=0)

Exp_J3HNHA = genfromtxt('Data_Compiled/Bax_3JHNHa.csv', delimiter=',')
Exp_J1CaHa = genfromtxt('Data_Compiled/Bax_1JHaCa.csv', delimiter=',')
Exp_J1CaN = genfromtxt('Data_Compiled/Bax_1JNCa.csv', delimiter=',')
Exp_J2CaN = genfromtxt('Data_Compiled/Bax_2JCaiNi.csv', delimiter=',')
Exp_J3CC = genfromtxt('Data_Compiled/Bax_3JCC.csv', delimiter=',')

Sim_J3HNHA = np.zeros([len(Exp_J3HNHA),3])
Sim_J1CaHa = np.zeros([len(Exp_J1CaHa),3])
Sim_J1CaN = np.zeros([len(Exp_J1CaN),3])
Sim_J2CaN = np.zeros([len(Exp_J2CaN),3])
Sim_J3CC = np.zeros([len(Exp_J3CC),3])

couplings_avg_list = [J3HNHA_coupling_avg,J1CaHa_coupling_avg,J1CaN_coupling_avg,J2CaN_coupling_avg,J3CC_coupling_avg]
couplings_std_list = [J3HNHA_coupling_std,J1CaHa_coupling_std,J1CaN_coupling_std,J2CaN_coupling_std,J3CC_coupling_std]
compiled_exp_list = [Exp_J3HNHA, Exp_J1CaHa, Exp_J1CaN, Exp_J2CaN, Exp_J3CC]
compiled_sim_list = [Sim_J3HNHA, Sim_J1CaHa, Sim_J1CaN, Sim_J2CaN, Sim_J3CC]

for com_idx, com_item in enumerate(compiled_exp_list):
	for res_idx, res_jc in enumerate(com_item):
		compiled_sim_list[com_idx][res_idx][0] = com_item[res_idx][0]
		compiled_sim_list[com_idx][res_idx][1] = couplings_avg_list[com_idx][int(com_item[res_idx][0])-1]
		compiled_sim_list[com_idx][res_idx][2] = couplings_std_list[com_idx][int(com_item[res_idx][0])-1]

R_Exp_J3HNHA = np.zeros([len(Exp_J3HNHA),1])
R_Exp_J1CaHa = np.zeros([len(Exp_J1CaHa),1])
R_Exp_J1CaN = np.zeros([len(Exp_J1CaN),1])
R_Exp_J2CaN = np.zeros([len(Exp_J2CaN),1])
R_Exp_J3CC = np.zeros([len(Exp_J3CC),1])

R_Sim_J3HNHA = np.zeros([len(Exp_J3HNHA),1])
R_Sim_J1CaHa = np.zeros([len(Exp_J1CaHa),1])
R_Sim_J1CaN = np.zeros([len(Exp_J1CaN),1])
R_Sim_J2CaN = np.zeros([len(Exp_J2CaN),1])
R_Sim_J3CC = np.zeros([len(Exp_J3CC),1])

D_list = [Exp_J3HNHA,Exp_J1CaHa,Exp_J1CaN,Exp_J2CaN,Exp_J3CC,Sim_J3HNHA,Sim_J1CaHa,Sim_J1CaN,Sim_J2CaN,Sim_J3CC]
R_list = [R_Exp_J3HNHA,R_Exp_J1CaHa,R_Exp_J1CaN,R_Exp_J2CaN,R_Exp_J3CC,R_Sim_J3HNHA,R_Sim_J1CaHa,R_Sim_J1CaN,R_Sim_J2CaN,R_Sim_J3CC]
for d_idx, d_item in enumerate(D_list):
	for d_ar_idx, d_ar_item in enumerate(d_item):
		R_list[d_idx][d_ar_idx] = d_ar_item[1]

rmsd_J3HNHA = np.average((R_Exp_J3HNHA-R_Sim_J3HNHA)**2)
print('J3HNHA -> ' + str(np.sqrt(rmsd_J3HNHA)))
final_J3HNHA = np.sqrt(rmsd_J3HNHA)
rmsd_J1CaHa = np.average((R_Exp_J1CaHa-R_Sim_J1CaHa)**2)
print('J1CaHa -> ' + str(np.sqrt(rmsd_J1CaHa)))
final_J1CaHa = np.sqrt(rmsd_J1CaHa)
rmsd_J1CaN = np.average((R_Exp_J1CaN-R_Sim_J1CaN)**2)
print('J1CaN -> ' + str(np.sqrt(rmsd_J1CaN)))
final_J1CaN = np.sqrt(rmsd_J1CaN)
rmsd_J2CaN = np.average((R_Exp_J2CaN-R_Sim_J2CaN)**2)
print('J2CaN -> ' + str(np.sqrt(rmsd_J2CaN)))
final_J2CaN = np.sqrt(rmsd_J2CaN)
rmsd_J3CC = np.average((R_Exp_J3CC-R_Sim_J3CC)**2)
print('J3CC -> ' + str(np.sqrt(rmsd_J3CC)))
final_J3CC = np.sqrt(rmsd_J3CC)
rmsd_final = np.sqrt(np.average([rmsd_J3HNHA,rmsd_J1CaHa,rmsd_J1CaN,rmsd_J2CaN,rmsd_J3CC]))

print(rmsd_final)

rmsd_full_list = [final_J3HNHA,final_J1CaHa,final_J1CaN,final_J2CaN,final_J3CC,rmsd_final]

np.savetxt('J-Coupling_RMSD.txt', rmsd_full_list, fmt='%s', delimiter=' ', newline='\n')

name_set = ['J3HNHA', 'J1CaHa', 'J1CaN', 'J2CaN', 'J3CC']
for outfileset_idx,outfileset in enumerate(compiled_sim_list):
	np.savetxt('Simulated_J-Couplings_' + str(name_set[outfileset_idx]) + '.txt', outfileset, fmt='%s', delimiter=' ', newline='\n')