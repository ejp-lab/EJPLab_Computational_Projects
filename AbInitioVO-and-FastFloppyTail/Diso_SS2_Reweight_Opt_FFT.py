## Syntax: run Disorder_Computed_Fragments_Input.py psipred_format_file_as_template.txt output_file_name.txt
import numpy as np
from numpy import genfromtxt
import sys
import os
import argparse
from itertools import groupby

## ARGUEMENT PARSING ##
parser = argparse.ArgumentParser(description='Program')
parser.add_argument('-ssin', '--Input_SecStruc_File', action='store', type=str, required=True,
	help='name of file containing secondary structure prediction from server of interest')
parser.add_argument('-disoin', '--Input_DisoPred_File', action='store', type=str, required=False,
	help='name of file containing RaptorX prediction of residue disorder probability')
parser.add_argument('-sstype', '--SecStruc_File_Type', action='store', type=str, required=True,
	help='name of server uses to produce the Input_SecStruc_File, currently supports RaptorX = rapx, PsiPred = ppred, Jufo = jufo')
parser.add_argument('-out', '--Output_File_Name', action='store', type=str, required=True,
	help='name of the output file')
parser.add_argument('-disoout', '--Output_Disorder_File_Name', action='store', type=str, required=False,
	help='name of the output file, ONLY specify if you want the disordered file re-written based on the input SS2 data')
args = parser.parse_args()

# Import the RaptorX Disorder File
diso_dat = 1
if args.Input_DisoPred_File:
	disorder_dat = np.genfromtxt(args.Input_DisoPred_File, dtype=[('dat1', 'f8'), ('dat2', '|S2'), ('dat3', '|S1'), ('dat4', 'f8')], delimiter=' ', skip_header=3) # 3 for raptorx
	diso_dat = np.empty((len(disorder_dat),1))
	for i in range(len(disorder_dat)):
		diso_dat[i] = disorder_dat[i][3]	
else:
	if 'ppred' in args.SecStruc_File_Type:
		disorder_dat = genfromtxt(args.Input_SecStruc_File, skip_header = 2, usecols=[0,1,2,3,4,5], dtype=None, encoding=None)
	if 'rapx' in args.SecStruc_File_Type:
		disorder_dat	= genfromtxt(args.Input_SecStruc_File, skip_header = 2, usecols=[0,1,2,3,4,5], dtype=None, encoding=None)
	if 'jufo' in args.SecStruc_File_Type:
		disorder_dat = genfromtxt(args.Input_SecStruc_File, usecols=[0,1,2,3,4,5], dtype=None, encoding=None)		
	diso_dat = np.empty((len(disorder_dat),1))
	for i in range(len(disorder_dat)):
		diso_dat[i] = 0.0

# Import the Secondary Structure Predition File
if 'ppred' in args.SecStruc_File_Type:
	sspred_file = genfromtxt(args.Input_SecStruc_File, skip_header = 2, usecols=[0,1,2,3,4,5], dtype=None, encoding=None)
if 'rapx' in args.SecStruc_File_Type:
	import_file	= genfromtxt(args.Input_SecStruc_File, skip_header = 2, usecols=[0,1,2,3,4,5], dtype=None, encoding=None)
	sspred_holder = genfromtxt(args.Input_SecStruc_File, skip_header = 2, usecols=[0,1,2,3,4,5], dtype=None, encoding=None)
	sspred_file = import_file
	for line_num,line_val in enumerate(sspred_file):
		sspred_file[line_num][3] = sspred_holder[line_num][5]
		sspred_file[line_num][4] = sspred_holder[line_num][3]
		sspred_file[line_num][5] = sspred_holder[line_num][4]
if 'jufo' in args.SecStruc_File_Type:
	sspred_file = genfromtxt(args.Input_SecStruc_File, usecols=[0,1,2,3,4,5], dtype=None, encoding=None)		
		
# Compute New Fragment Probabilities
new_frag_probs = sspred_file
for nfp_line_num, nfp_line_val in enumerate(new_frag_probs):
	res_dis_prob = float(diso_dat[nfp_line_num])
	res_ord_prob = 1.000 - res_dis_prob
	if new_frag_probs[nfp_line_num][4] == 0:
		if new_frag_probs[nfp_line_num][3] <= res_dis_prob:
			new_frag_probs[nfp_line_num][3] = res_dis_prob
			new_frag_probs[nfp_line_num][5] = res_ord_prob
		else:
			new_frag_probs[nfp_line_num][3] = sspred_file[nfp_line_num][3]
			new_frag_probs[nfp_line_num][5] = sspred_file[nfp_line_num][5]
	elif new_frag_probs[nfp_line_num][5] == 0:
		if new_frag_probs[nfp_line_num][3] <= res_dis_prob:
			new_frag_probs[nfp_line_num][3] = res_dis_prob
			new_frag_probs[nfp_line_num][4] = res_ord_prob
		else:
			new_frag_probs[nfp_line_num][3] = sspred_file[nfp_line_num][3]
			new_frag_probs[nfp_line_num][4] = sspred_file[nfp_line_num][4]
	elif new_frag_probs[nfp_line_num][3] <= res_dis_prob:
		new_frag_probs[nfp_line_num][3] = res_dis_prob
		new_frag_probs[nfp_line_num][4] = res_ord_prob*(sspred_file[nfp_line_num][4]/(sspred_file[nfp_line_num][4] + sspred_file[nfp_line_num][5]))
		new_frag_probs[nfp_line_num][5] = res_ord_prob*(sspred_file[nfp_line_num][5]/(sspred_file[nfp_line_num][4] + sspred_file[nfp_line_num][5]))
	elif new_frag_probs[nfp_line_num][3] > res_dis_prob:
		new_frag_probs[nfp_line_num][3] = sspred_file[nfp_line_num][3]
		new_frag_probs[nfp_line_num][4] = sspred_file[nfp_line_num][4]
		new_frag_probs[nfp_line_num][5] = sspred_file[nfp_line_num][5]
	if new_frag_probs[nfp_line_num][3] > new_frag_probs[nfp_line_num][4] and new_frag_probs[nfp_line_num][3] > new_frag_probs[nfp_line_num][5]:
		new_frag_probs[nfp_line_num][2] = 'C'
	elif new_frag_probs[nfp_line_num][4] > new_frag_probs[nfp_line_num][3] and new_frag_probs[nfp_line_num][4] > new_frag_probs[nfp_line_num][5]:
		new_frag_probs[nfp_line_num][2] = 'H'
	elif new_frag_probs[nfp_line_num][5] > new_frag_probs[nfp_line_num][4] and new_frag_probs[nfp_line_num][5] > new_frag_probs[nfp_line_num][3]:
		new_frag_probs[nfp_line_num][2] = 'E'	

# Compute New Fragment Probabilities
new_frag_probs = sspred_file
current_ss2 = []
output_diso_list = []
for nfp_line_num, nfp_line_val in enumerate(new_frag_probs):
	if new_frag_probs[nfp_line_num][3] >= 0.75:
		current_ss2.append('C')
	elif new_frag_probs[nfp_line_num][4] >= 0.75:
		current_ss2.append('H')
	elif new_frag_probs[nfp_line_num][5] >= 0.75:
		current_ss2.append('E')
	else:
		current_ss2.append('X')

for key, value in groupby(current_ss2):
	num_con = len(list(value))
	for i in range(num_con):
		if num_con >= 5 and key == 'H':
			output_diso_list.append('H')
		elif num_con >= 3 and key == 'E':
			output_diso_list.append('E')
		else:
			output_diso_list.append('C')
			
for nfp_line_num, nfp_line_val in enumerate(new_frag_probs):
	if output_diso_list[nfp_line_num] == 'C':
		new_frag_probs[nfp_line_num][2] = 'C'
		new_frag_probs[nfp_line_num][3] = 1.000
		new_frag_probs[nfp_line_num][4] = 0.000
		new_frag_probs[nfp_line_num][5] = 0.000
	else:
		continue

# Maybe if something passes this criteria then you can say if
# there is a contiguous stretch before the correction then you are good there too.
# but maybe only for sheets??

# Write the Output File
outf = open(args.Output_File_Name, 'w')
outf.write("""# PSIPRED VFORMAT (PSIPRED V3.3)""")
outf.write("\n")
outf.write("\n")
for line in new_frag_probs:
#	outf.write("\t%i\n" % (line[0]))
	if line[0] < 10:
		outf.write("   %i %s %s   %.3f  %.3f  %.3f\n" % (line[0],line[1],line[2],line[3],line[4],line[5]))
	elif line[0] < 100:
		outf.write("  %i %s %s   %.3f  %.3f  %.3f\n" % (line[0],line[1],line[2],line[3],line[4],line[5]))
	elif line[0] < 1000:
		outf.write(" %i %s %s   %.3f  %.3f  %.3f\n" % (line[0],line[1],line[2],line[3],line[4],line[5]))
outf.close()	

if args.Output_Disorder_File_Name:
	for res_idx, res_item in enumerate(diso_dat):
		if res_item < new_frag_probs[res_idx][3]:
			diso_dat[res_idx] = new_frag_probs[res_idx][3]
	outfd = open(args.Output_Disorder_File_Name, 'w')
	file_tracker = []
	with open(args.Input_DisoPred_File, 'r') as indisofile:
		indisodata = indisofile.readlines() 	 
		for line_idx, line_item in enumerate(indisodata):
			if line_idx > 2:
				file_tracker.append(line_item[:9])
				file_tracker.append("%.3f" % round(float(diso_dat[line_idx-3]),3) + ' \n')
			else:
				file_tracker.append(line_item)
	np.savetxt(args.Output_Disorder_File_Name, file_tracker, fmt='%s', newline='')			