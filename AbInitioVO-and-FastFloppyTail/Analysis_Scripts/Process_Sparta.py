# To run Sparta+ first enter the command "./sparta+Init.com" in the /cygdrive/c/cygwin/Sparta/SPARTA+ folder
# Sparta+ is run via the command "./sparta+"
# BEFORE RUNNING mkdir Sparta AND CHANGE DIRECTORY IN in_folder VARIABLE

import sys
from numpy import genfromtxt
import numpy as np
import os
from shutil import copy

in_folder_set = sys.argv[1].split('/')
in_folder = ''
for folder_idx,folder_item in enumerate(in_folder_set):
	if folder_idx < len(in_folder_set) - 1:
		in_folder += folder_item
		if folder_idx < len(in_folder_set) -2:
			in_folder += '/'

os.mkdir(in_folder + 'Sparta')

for idx,item in enumerate(sys.argv[1:]):
	outfile_1 = 'Sparta/Pred_output_' + '_' + str(idx) + '.txt'
	outfile_2 = 'Sparta/Struct_output_' + '_' + str(idx) + '.txt'
	infile = item
	print outfile_1
	print infile
	runinput = '../Analysis_Packages/SPARTA+/sparta+ -in ' + str(in_folder) + str(infile) + ' -out ' + str(in_folder) + str(outfile_1) + ' -outS ' + str(in_folder) + str(outfile_2) + ' -spartaDir C:/cygwin/Sparta/SPARTA+' 
	print runinput
	os.system(runinput)
