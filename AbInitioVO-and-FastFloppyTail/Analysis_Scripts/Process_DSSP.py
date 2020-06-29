import sys
from numpy import genfromtxt
import numpy as np
import os
from shutil import copy

for idx,item in enumerate(sys.argv[1:]):
	outfile = 'dssp_output_' + str(idx) + '.txt'
	infile = item
	print(outfile)
	print(infile)
	runinput = 'mkdssp -i ' + str(infile) + ' -o ' + str(outfile)
	print(runinput)
	os.system(runinput)