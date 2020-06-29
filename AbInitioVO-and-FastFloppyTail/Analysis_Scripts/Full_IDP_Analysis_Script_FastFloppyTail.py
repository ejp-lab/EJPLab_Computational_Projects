import sys
from numpy import genfromtxt
import numpy as np
import os
from shutil import copy
import time
import subprocess
import glob

foldernumber = FOLDERNUMBER

# Items for Series run on First Thread
if foldernumber == 1:
	# Create folder for Paralleled Analyses
	os.mkdir('PALES')
	os.mkdir('SPARTA')
	# Run PDB Compiler
	os.system('python Analysis_Scripts/pdb_assembler.py asyn_compiled.pdb FloppyTail_Relaxed_*.pdb')
	# Run Polymer Analysis
	os.system('python Analysis_Scripts/polymer_analysis.py asyn_compiled.pdb')
	# Run Distance Extraction
	os.system('python Analysis_Scripts/Distance_Extraction.py Data_Compiled/asyn_distances.cst asyn_distances.out asyn_compiled.dm asyn_compiled.std')
	# Run FRET Computes
	os.system('python Analysis_Scripts/EFRETs_from_Ensembles.py Data_Compiled/asyn_efret.txt FloppyTail_Relaxed_*.pdb')
	# Run PRE Computes
	os.system('python Analysis_Scripts/PREs_from_Ensembles.py Data_Compiled/asyn.pre FloppyTail_Relaxed_*.pdb')
	os.system('python Analysis_Scripts/PRE_Data_Comparison.py Data_Compiled/asyn.pre')
	# Run DSSP Analysis
	os.mkdir('DSSP')
	os.chdir('DSSP')
	os.system('python Analysis_Scripts/Process_DSSP.py ../FloppyTail_Relaxed_*.pdb')
	os.system('python Analysis_Scripts/dssp_output_analysis.py ../asyn_compiled.nu dssp_output_*.txt')
	dssp_output_list = ['output_tco.txt', 'output_racc.txt', 'output_hbond.txt', 'output_hbondtotal.txt', 'output_acc_total.txt', 'output_phi_psi_2his.txt', 'output_phi_psi_no_GLY_2his.txt', 'Percent_HEL.txt']
	for dssp_output in dssp_output_list:
		copy(dssp_output, '../asyn_' + dssp_output.split('.')[0] + '.txt')
	# Run J-Coupling Computes
	os.system('python Analysis_Scripts/J_Couplings.py dssp_output_*.txt')
	j_couplings_output_list = ['J-Coupling_RMSD.txt']
	for j_coupling_output in j_couplings_output_list:
		copy(j_coupling_output, '../asyn_' + j_coupling_output.split('.')[0] + '.txt')
	# Wait for PALES to finish and analyze
	os.chdir('../')
	analyze_PALES = False
	number_of_pdb = len(glob.glob('FloppyTail_Relaxed_*.pdb'))
	number_of_res = len(np.genfromtxt('Data_Compiled/asyn_PALES.in', dtype=None, encoding=None, skip_header=7))
	while analyze_PALES == False:
		list_of_PALES_outputs = glob.glob('PALES/PALES_*/*.out')
		list_of_PALES_tables = glob.glob('PALES/*.tbl')
		residue_list = []
		if len(list_of_PALES_outputs) > 0:
			opened_file = open(list_of_PALES_outputs[0])
			skip_header_line_val = 0
			for txt_line_num,txt_line in enumerate(opened_file):
				if 'FORMAT' in txt_line:
					skip_header_line_val = txt_line_num + 1
			opened_file.close()
			base_data = np.genfromtxt(list_of_PALES_outputs[0], dtype=None, encoding=None, skip_header=skip_header_line_val)
			residue_list = base_data['f0']			
		if len(list_of_PALES_tables) == len(residue_list)*number_of_pdb and len(residue_list) > 0:
			analyze_PALES = True
		else:
			time.sleep(300)
	os.system('python Analysis_Scripts/PALES_Analysis_Parallel_bestFit.py PALES/')
	# Wait for Sparta+ to finish and analyze
	analyze_Sparta = False
	while analyze_Sparta == False:
		list_of_Sparta_outputs = glob.glob('SPARTA/Pred_output_*.txt')
		if len(list_of_Sparta_outputs) == number_of_pdb:
			analyze_Sparta = True
		else:
			time.sleep(300)
	os.system('python Analysis_Scripts/Sparta_Analysis_Post_Process_2.py Data_Compiled/asyn_chemical_shifts.nmr SPARTA/Pred_output_*.txt')
	
	
# Items for Parallel run on Remaining Threads
if foldernumber > 1:
	# Wait for PALES
	while not os.path.exists('PALES'):
		time.sleep(1)
	# Run PALES
	os.chdir('PALES')
	os.mkdir('PALES_' + str(foldernumber))
	os.chdir('PALES_' + str(foldernumber))
	os.system('python ../../Process_PALES_Parallel_' + str(foldernumber) + '.py Data_Compiled/asyn_PALES.in ../../FloppyTail_Relaxed_*.pdb &> /dev/null')
	os.chdir('../../')
	# Wait from SPARTA+
	while not os.path.exists('SPARTA'):
		time.sleep(1)
	# Run SPARTA+
	os.system('python Process_SPARTA_Parallel_' + str(foldernumber) + '.py FloppyTail_Relaxed_*.pdb &> /dev/null')	

