## To Run: run Sparta_Analysis primary_seq_len pred_files_*
import numpy as np
from numpy import genfromtxt
import sys
import os

current_file = genfromtxt(sys.argv[2], skip_header=29, usecols=[0,1,2,3,4,5,6,7,8], dtype=None, encoding=None)
total_res_num = current_file[len(current_file) - 1][0]

# Arrays for storing differnt atom shift data
shifts_N = np.zeros((int(total_res_num),int(len(sys.argv)-2)))
shifts_HA = np.zeros((int(total_res_num),int(len(sys.argv)-2)))
shifts_C = np.zeros((int(total_res_num),int(len(sys.argv)-2)))
shifts_CA = np.zeros((int(total_res_num),int(len(sys.argv)-2)))
shifts_CB = np.zeros((int(total_res_num),int(len(sys.argv)-2)))
shifts_HN = np.zeros((int(total_res_num),int(len(sys.argv)-2)))

rc_shifts_N = np.zeros((int(total_res_num),int(len(sys.argv)-2)))
rc_shifts_HA = np.zeros((int(total_res_num),int(len(sys.argv)-2)))
rc_shifts_C = np.zeros((int(total_res_num),int(len(sys.argv)-2)))
rc_shifts_CA = np.zeros((int(total_res_num),int(len(sys.argv)-2)))
rc_shifts_CB = np.zeros((int(total_res_num),int(len(sys.argv)-2)))
rc_shifts_HN = np.zeros((int(total_res_num),int(len(sys.argv)-2)))

print(len(sys.argv))
for file_num,file_name in enumerate(sys.argv[2:]):
	current_file = genfromtxt(str(file_name), skip_header=29, usecols=[0,1,2,3,4,5,6,7,8], dtype=None, encoding=None)
	for line in range(0, len(current_file)):
		if current_file[line][2] == 'N':
			shifts_N[(int(current_file[line][0])-1)][file_num] = current_file[line][4]
			rc_shifts_N[(int(current_file[line][0])-1)][file_num] = current_file[line][5]
		if current_file[line][2] == 'HA':
			shifts_HA[(int(current_file[line][0])-1)][file_num] = current_file[line][4]
			rc_shifts_HA[(int(current_file[line][0])-1)][file_num] = current_file[line][5]
		if current_file[line][2] == 'C':
			shifts_C[(int(current_file[line][0])-1)][file_num] = current_file[line][4]
			rc_shifts_C[(int(current_file[line][0])-1)][file_num] = current_file[line][5]
		if current_file[line][2] == 'CA':
			shifts_CA[(int(current_file[line][0])-1)][file_num] = current_file[line][4]
			rc_shifts_CA[(int(current_file[line][0])-1)][file_num] = current_file[line][5]
		if current_file[line][2] == 'CB':
			shifts_CB[(int(current_file[line][0])-1)][file_num] = current_file[line][4]
			rc_shifts_CB[(int(current_file[line][0])-1)][file_num] = current_file[line][5]
		if current_file[line][2] == 'HN':
			shifts_HN[(int(current_file[line][0])-1)][file_num] = current_file[line][4]
			rc_shifts_HN[(int(current_file[line][0])-1)][file_num] = current_file[line][5]

# Making Average and StdDev arrays
final_N = np.zeros((int(total_res_num),2))
final_HA = np.zeros((int(total_res_num),2))
final_C = np.zeros((int(total_res_num),2))
final_CA = np.zeros((int(total_res_num),2))
final_CB = np.zeros((int(total_res_num),2))
final_HN = np.zeros((int(total_res_num),2))

rc_final_N = np.zeros((int(total_res_num),2))
rc_final_HA = np.zeros((int(total_res_num),2))
rc_final_C = np.zeros((int(total_res_num),2))
rc_final_CA = np.zeros((int(total_res_num),2))
rc_final_CB = np.zeros((int(total_res_num),2))
rc_final_HN = np.zeros((int(total_res_num),2))

N_average = np.average(shifts_N, axis=1)
N_std = np.std(shifts_N, axis=1)
HA_average = np.average(shifts_HA, axis=1)
HA_std = np.std(shifts_HA, axis=1)
C_average = np.average(shifts_C, axis=1)
C_std = np.std(shifts_C, axis=1)
CA_average = np.average(shifts_CA, axis=1)
CA_std = np.std(shifts_CA, axis=1)
CB_average = np.average(shifts_CB, axis=1)
CB_std = np.std(shifts_CB, axis=1)
HN_average = np.average(shifts_HN, axis=1)
HN_std = np.std(shifts_HN, axis=1)

rc_N_average = np.average(rc_shifts_N, axis=1)
rc_N_std = np.std(rc_shifts_N, axis=1)
rc_HA_average = np.average(rc_shifts_HA, axis=1)
rc_HA_std = np.std(rc_shifts_HA, axis=1)
rc_C_average = np.average(rc_shifts_C, axis=1)
rc_C_std = np.std(rc_shifts_C, axis=1)
rc_CA_average = np.average(rc_shifts_CA, axis=1)
rc_CA_std = np.std(rc_shifts_CA, axis=1)
rc_CB_average = np.average(rc_shifts_CB, axis=1)
rc_CB_std = np.std(rc_shifts_CB, axis=1)
rc_HN_average = np.average(rc_shifts_HN, axis=1)
rc_HN_std = np.std(rc_shifts_HN, axis=1)

for row_num in range(0, int(total_res_num)):
	final_N[row_num][0] = N_average[row_num]
	final_N[row_num][1] = N_std[row_num]
	final_HA[row_num][0] = HA_average[row_num]
	final_HA[row_num][1] = HA_std[row_num]
	final_C[row_num][0] = C_average[row_num]
	final_C[row_num][1] = C_std[row_num]
	final_CA[row_num][0] = CA_average[row_num]
	final_CA[row_num][1] = CA_std[row_num]
	final_CB[row_num][0] = CB_average[row_num]
	final_CB[row_num][1] = CB_std[row_num]
	final_HN[row_num][0] = HN_average[row_num]
	final_HN[row_num][1] = HN_std[row_num]
	# Now for the Random Coil Values
	rc_final_N[row_num][0] = rc_N_average[row_num]
	rc_final_N[row_num][1] = rc_N_std[row_num]
	rc_final_HA[row_num][0] = rc_HA_average[row_num]
	rc_final_HA[row_num][1] = rc_HA_std[row_num]
	rc_final_C[row_num][0] = rc_C_average[row_num]
	rc_final_C[row_num][1] = rc_C_std[row_num]
	rc_final_CA[row_num][0] = rc_CA_average[row_num]
	rc_final_CA[row_num][1] = rc_CA_std[row_num]
	rc_final_CB[row_num][0] = rc_CB_average[row_num]
	rc_final_CB[row_num][1] = rc_CB_std[row_num]
	rc_final_HN[row_num][0] = rc_HN_average[row_num]
	rc_final_HN[row_num][1] = rc_HN_std[row_num]	
	

## Importing Experimental and Random Coil Data
### Random Coil Data C CB CA N H HN
#random_coil_set = np.genfromtxt(sys.argv[2], delimiter='\t', dtype=None, encoding=None)
#randcoil_chemshift_set = np.zeros([len(final_N),12])
#for rand_data_idx,rand_data_item in enumerate(random_coil_set):
#	for atom_idx,atom_item in enumerate(rand_data_item):
#		if atom_idx > 1 and atom_idx < 8:
#			new_idx = atom_idx - 2
#			randcoil_chemshift_set[rand_data_idx+1][2*new_idx] = atom_item

# Compile all Computed Results
comp_data_list = [final_C, final_CB, final_CA, final_N, final_HN, final_HA]
computed_chemshift_set = np.zeros([len(final_N),12])
for comp_data_idx,comp_data_item in enumerate(comp_data_list):
	for residue_idx,residue_item in enumerate(comp_data_item):
		computed_chemshift_set[residue_idx][2*comp_data_idx] = residue_item[0]
		computed_chemshift_set[residue_idx][2*comp_data_idx+1] = residue_item[1]
		
# Compile all Random Coil Results
rc_data_set = [rc_final_C, rc_final_CB, rc_final_CA, rc_final_N, rc_final_HN, rc_final_HA]
randcoil_chemshift_set = np.zeros([len(rc_final_N),12])
for comp_data_idx,comp_data_item in enumerate(rc_data_set):
	for residue_idx,residue_item in enumerate(comp_data_item):
		randcoil_chemshift_set[residue_idx][2*comp_data_idx] = residue_item[0]
		randcoil_chemshift_set[residue_idx][2*comp_data_idx+1] = residue_item[1]
		
# Compile all Experimental Results
exp_data_file_list = np.genfromtxt(sys.argv[1], dtype=str)
exp_data_set = []
for exp_data_file in exp_data_file_list:
	exp_data_set.append(genfromtxt(exp_data_file, delimiter=' '))
experimental_chemshift_set = np.zeros([len(final_N),12])
for exp_data_idx,exp_data_item in enumerate(exp_data_set):
	for residue_idx,residue_item in enumerate(exp_data_item):
		experimental_chemshift_set[residue_idx][2*exp_data_idx] = residue_item[0]
		experimental_chemshift_set[residue_idx][2*exp_data_idx+1] = residue_item[1]
		
# Compute RMSD Values
difference_data_list = computed_chemshift_set - experimental_chemshift_set
rmsd_list = ['C', 'CB', 'CA', 'N', 'H', 'HA'] # final column in HA but no data is present 
full_rmsd_list = []
for rmsd_idx,rmsd_item in enumerate(rmsd_list):
	rmsd_data_holder = []
	for diff_res_idx,diff_res_item in enumerate(difference_data_list):
		if computed_chemshift_set[diff_res_idx][2*rmsd_idx] != 0.0 and experimental_chemshift_set[diff_res_idx][2*rmsd_idx] != 0.0 and randcoil_chemshift_set[diff_res_idx][2*rmsd_idx] != 0.0:
			rmsd_data_holder.append(diff_res_item[2*rmsd_idx]**2)
			full_rmsd_list.append(diff_res_item[2*rmsd_idx]**2)
	rmsd_list[rmsd_idx] = np.sqrt(np.average(rmsd_data_holder))
full_rmsd = np.sqrt(np.average(full_rmsd_list))
rmsd_list.append(full_rmsd)

# Computing and Saving Random Coil Subtracted Experiments and Computed Values
rand_sub_compute = computed_chemshift_set - randcoil_chemshift_set
rand_sub_experimental = experimental_chemshift_set - randcoil_chemshift_set
for randcoilres_idx, randcoilres_item in enumerate(experimental_chemshift_set):
	for randcoilatom_idx, randcoilatom_item in enumerate(randcoilres_item):
		if randcoilatom_idx %2 == 0:
			if experimental_chemshift_set[randcoilres_idx][randcoilatom_idx] == 0.0 or computed_chemshift_set[randcoilres_idx][randcoilatom_idx] == 0.0:
				rand_sub_compute[randcoilres_idx][randcoilatom_idx] = 0.0
				rand_sub_experimental[randcoilres_idx][randcoilatom_idx] = 0.0
				rand_sub_compute[randcoilres_idx][randcoilatom_idx+1] = 0.0
				rand_sub_experimental[randcoilres_idx][randcoilatom_idx+1] = 0.0
				
np.savetxt('Experiment_RandomCoilSub_ChemShifts.txt', rand_sub_experimental, fmt='%s', delimiter=' ', newline='\n')
np.savetxt('Computed_RandomCoilSub_ChemShifts.txt', rand_sub_compute, fmt='%s', delimiter=' ', newline='\n')
np.savetxt('Chemical_Shift_RMSDs.txt', rmsd_list, fmt='%s', delimiter=' ', newline='\n')

## Saving Final Arrays to Text
np.savetxt('Chemical_Shifts_N.txt', final_N, fmt='%s', delimiter=' ', newline='\n')
np.savetxt('Chemical_Shifts_HA.txt', final_HA, fmt='%s', delimiter=' ', newline='\n')
np.savetxt('Chemical_Shifts_C.txt', final_C, fmt='%s', delimiter=' ', newline='\n')
np.savetxt('Chemical_Shifts_CA.txt', final_CA, fmt='%s', delimiter=' ', newline='\n')
np.savetxt('Chemical_Shifts_CB.txt', final_CB, fmt='%s', delimiter=' ', newline='\n')
np.savetxt('Chemical_Shifts_HN.txt', final_HN, fmt='%s', delimiter=' ', newline='\n')