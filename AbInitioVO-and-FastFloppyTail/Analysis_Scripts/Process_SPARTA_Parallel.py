# To run Sparta+ first enter the command "./sparta+Init.com" in the /cygdrive/c/cygwin/Sparta/SPARTA+ folder
# Sparta+ is run via the command "./sparta+"
# BEFORE RUNNING mkdir Sparta AND CHANGE DIRECTORY IN in_folder VARIABLE

import sys
from numpy import genfromtxt
import numpy as np
import os
from shutil import copy

foldernumber = FOLDERNUMBER - 2

#in_folder_set = sys.argv[1].split('/')
#in_folder = ''
#for folder_idx,folder_item in enumerate(in_folder_set):
#	if folder_idx < len(in_folder_set) - 1:
#		in_folder += folder_item
#		if folder_idx < len(in_folder_set) -2:
#			in_folder += '/'

for idx,item in enumerate(sys.argv[1:]):
	if (idx - foldernumber) % 24 ==0:
		outfile_1 = 'SPARTA/Pred_output_' + '_' + str(idx) + '.txt'
		outfile_2 = 'SPARTA/Struct_output_' + '_' + str(idx) + '.txt'
		infile = item
		print(outfile_1)
		print(infile)
		runinput = '/home/gianna1/FloppyTail/Analysis_Packages/SPARTA+/sparta+ -in ' + str(infile) + ' -out ' + str(outfile_1) + ' -outS ' + str(outfile_2) + ' -spartaDir /home/gianna1/FloppyTail/Analysis_Packages/SPARTA+' 
		print(runinput)
		os.system(runinput)
