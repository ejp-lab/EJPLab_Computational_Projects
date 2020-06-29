## To Run: run PALES_Analysis RDC_data pred_files_*
import numpy as np
from numpy import genfromtxt
import sys
import os
import glob

list_of_tbl_files = glob.glob(str(sys.argv[1]) + '*.tbl')
list_of_out_files = glob.glob(str(sys.argv[1]) + 'PALES_*/base_*.out')
opened_file = open(str(list_of_out_files[0]))
skip_header_line_val = 0
for txt_line_num,txt_line in enumerate(opened_file):
	if 'FORMAT' in txt_line:
		skip_header_line_val = txt_line_num + 1
opened_file.close()
single_data_set = np.genfromtxt(str(list_of_out_files[0]), dtype=None, encoding=None, skip_header=skip_header_line_val)
residue_list = single_data_set['f0']
exp_data_list = single_data_set['f7']
number_of_structures = len(list_of_tbl_files)/len(residue_list)
full_rdc_set = np.zeros([int(len(residue_list)),int(number_of_structures)])
for file_name in list_of_tbl_files:
	residue_number = int(file_name.split('/')[len(file_name.split('/'))-1].split('_')[2])
	file_number_for_res = int(file_name.split('/')[len(file_name.split('/'))-1].split('_')[5].split('.')[0])
	opened_file = open(file_name)
	skip_header_line_val = 0
	for txt_line_num,txt_line in enumerate(opened_file):
		if 'FORMAT' in txt_line:
			skip_header_line_val = txt_line_num + 1
	opened_file.close()		
	current_data = np.genfromtxt(file_name, dtype=None, encoding=None, skip_header=skip_header_line_val)
	for data_line_idx, data_line_item in enumerate(current_data):
		if int(data_line_item[0]) == residue_number:
			full_rdc_set[np.where(residue_list == residue_number)[0][0]][file_number_for_res] = data_line_item['f8']
			break
			
average_rdc = np.average(full_rdc_set, axis=1)
stddev_rdc = np.std(full_rdc_set, axis=1)
compiled_rdc = np.zeros([int(len(average_rdc)),2])
for row_idx,row_item in enumerate(compiled_rdc):
	compiled_rdc[row_idx][0] = average_rdc[row_idx]
	compiled_rdc[row_idx][1] = stddev_rdc[row_idx]

## Saving Final Arrays to Text
np.savetxt('Average_RDCs.txt', compiled_rdc, fmt='%s', delimiter=' ', newline='\n')

RMSD = np.sqrt(np.average((average_rdc-exp_data_list)**2))
Q_VAL = RMSD/(np.sqrt(np.average(exp_data_list**2)))
RMSD_val = [RMSD, Q_VAL]
np.savetxt('RDC_RMSD.txt', RMSD_val, fmt='%s')