##After output add:
#DATA SEQUENCE MDVFKKGFSIAKEGVVGAVEKTKQGVTEAAEKTKEGVMYVGAKTKENVVQSVTSVAEKTK
#DATA SEQUENCE EQANAVSEAVVSSVNTVATKTVEEAENIAVTSGVVRKEDLRPSAPQQEGEASKEKEEVAE
#DATA SEQUENCE EAQSGGD
#
#VARS   RESID_I RESNAME_I ATOMNAME_I RESID_J RESNAME_J ATOMNAME_J D DD W
#FORMAT %8d     %8s       %8s        %8d     %8s    %8s %8.1f %8.1f %8.1f

import numpy as np
import sys

fasta_file= sys.argv[1]
fasta_file = open(args.Input_FASTA_File, 'r')
fasta_lines = fasta_file.readlines()
fasta_counter = 0
fasta_sequence = ' '
for fasta_line in fasta_lines:
	if '>' not in fasta_line:
		if fasta_counter == 0:
			if '\n' in fasta_line:
				fasta_sequence = fasta_line.split('\n')[0]
			else:
				fasta_sequence = fasta_line
			fasta_counter = 1	
		else:
			if '\n' in fasta_line:
				fasta_sequence = fasta_sequence + fasta_line.split('\n')[0]
			else:
				fasta_sequence = fasta_sequence + fasta_line
#fasta = "MDVFKKGFSIAKEGVVGAVEKTKQGVTEAAEKTKEGVMYVGAKTKENVVQSVTSVAEKTKEQANAVSEAVVSSVNTVATKTVEEAENIAVTSGVVRKEDLRPSAPQQEGEASKEKEEVAEEAQSGGD"
fasta = str(fasta_sequence)
input_template = np.genfromtxt("SOMEOTHERDIRECTORY/dc_1IGD.tab", skip_header=6, dtype=None)
new_input_array = np.empty((len(fasta)), dtype=[('f0', 'i4'), ('f1', 'S3'), ('f2', 'S2'), ('f3', 'i4'), ('f4', 'S3'), ('f5', 'S1'), ('f6', 'f8'), ('f7', 'f8'), ('f8', 'f8')])

aa_3_mat   = { \
        	   'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP',\
        	   'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY',\
        	   'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',\
        	   'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER',\
 	           'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
 	           
for idx,item in enumerate(fasta):
	print idx
	new_input_array[idx][0] = idx + 1
	new_input_array[idx][1] = aa_3_mat[item]
	new_input_array[idx][2] = 'HN'
	new_input_array[idx][3] = idx + 1
	new_input_array[idx][4] = aa_3_mat[item]
	new_input_array[idx][5] = 'N'
	new_input_array[idx][6] = 10.000
	new_input_array[idx][7] = 1.0
	new_input_array[idx][8] = 1.0
	

np.savetxt('PALES_IN.txt', new_input_array, fmt='%8s')