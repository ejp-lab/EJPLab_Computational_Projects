## To Run: run Process_PALES_bestFit.py structures_.pdb

import sys
from numpy import genfromtxt
import numpy as np
import os
from shutil import copy


for idx,item in enumerate(sys.argv[1:]):
	outfile = 'pales_output_' + str(idx) + '.tbl'
	infile = item
	runinput = '../../Analysis_Packages/pales-linux -bestFit -inD ../../PRE_Data_Compiled/asyn_PALES.in -pdb ' + str(infile) + ' -outD ' + str(outfile)
	print(runinput)
	os.system(runinput)
