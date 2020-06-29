## To Run: run Process_PALES_2.py seq_len structures_.pdb

import sys
from numpy import genfromtxt
import numpy as np
import os
from shutil import copy

foldernumber = FOLDERNUMBER - 2
baseinput = '/home/gianna1/FloppyTail/Analysis_Packages/pales-linux -bestFit -inD ' + str(sys.argv[1]) + ' -pdb ' + str(sys.argv[2]) + ' -outD base_' + str(foldernumber) + '.out > /dev/null'
os.system(baseinput)
opened_file = open('base_' + str(foldernumber) + '.out')
skip_header_line_val = 0
for txt_line_num,txt_line in enumerate(opened_file):
	if 'FORMAT' in txt_line:
		skip_header_line_val = txt_line_num + 1
opened_file.close()
base_data = np.genfromtxt('base_' + str(foldernumber) + '.out', dtype=None, encoding=None, skip_header=skip_header_line_val)
residue_list = base_data['f0']
seq_len = residue_list[len(residue_list)-1]
for res_num_idx,res_num_item in enumerate(residue_list):
	res_num = int(res_num_item)
	if res_num < 8:
		law_start = 1
		law_end = 15
	elif res_num > seq_len - 7:
		law_start = seq_len - 15
		law_end = seq_len
	else:
		law_start = res_num - 7
		law_end = res_num + 7
	for idx,item in enumerate(sys.argv[2:]):
		if (idx - foldernumber) % 24 ==0:
			outfile = '../res_num_' + str(res_num) + '_pales_output_' + str(idx) + '.tbl'
			infile = item
			print(res_num)
			print(law_start)
			print(law_end)
			runinput = '/home/gianna1/FloppyTail/Analysis_Packages/pales-linux -bestFit -inD ' + str(sys.argv[1]) + ' -pdb ' + str(infile) + ' -outD ' + str(outfile) + ' -r1 ' + str(law_start) + ' -rN ' + str(law_end) + ' -s1 ' + str(law_start) + ' -sN ' + str(law_end) + ' > /dev/null'
			print(runinput)
			os.system(runinput)
