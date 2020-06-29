# Syntax run ReWrite.py OneLetterOrderCode StartResidue EndResidue Native.pdb sub_directory_*/
## OneLetterOrderCodes: O = ordered (compares RMSD to native pdb), P = partially ordered (compares RMSD to segment of native PDB), D = disordered (compares RMSD to lowest energy)
import sys
from numpy import genfromtxt
import numpy as np
import os
from shutil import copy
from Bio.PDB import *
import glob
import time
from pyrosetta import *  ## for newer versions make all start-up commands with pyrosetta
init()
from pyrosetta.rosetta.core.scoring import *
from pyrosetta.rosetta.core.scoring.methods import *
from pyrosetta.rosetta.core.scoring.methods import EnergyMethodOptions
from pyrosetta.rosetta.protocols.grafting import *
from pyrosetta.rosetta.protocols.simple_moves import *
from pyrosetta.rosetta.protocols.moves import *
from pyrosetta.rosetta.core.fragment import *
from pyrosetta.rosetta.protocols.minimization_packing import *

sge_task_id = FOLDERNUMBER
parallel_num = sge_task_id - 1
working_dir = 'FloppyTail_Relaxed_'

# Creates a single array/file that contains all scores from all output structures
abin_out_file = "FloppyTail_Compiled.sc"
rewrite_abin_out_file = "FloppyTail_Relaxed_Scores_"
perresval_abin_out_file = "FloppyTail_PerResEnergy_"
new_directory = 'FloppyTail_Relaxed_' + str(sge_task_id)
new_parent_directory = 'FloppyTail_Relaxed_Compiled'
new_abin_structure_names = 'FloppyTail_Relaxed_'
old_abin_structure_names = 'FloppyTail_'
color_pdb_abin = 'AbInitio_Color_by_Energy.pdb'
analysis_folder_name = 'Post_FloppyTail_Selection/'
#os.makedirs(new_directory)
#os.chmod(new_directory,0o777)
process_list = [abin_out_file]
rewrite_list = [rewrite_abin_out_file]
perresval_list = [perresval_abin_out_file]
struct_name_list = [new_abin_structure_names]
old_struct_name_list = [old_abin_structure_names]
color_pdb_name_list = [color_pdb_abin]
parser = PDBParser()
io = PDBIO()
sup = Superimposer()
os.chdir('..')

# Relaxing the lowest X output stuctures in parallel
sf2015 = create_score_function('ref2015')
sfrg = ScoreFunction()
sfrg.set_weight(rg, 1.0)

relax = rosetta.protocols.relax.FastRelax()
relax.min_type('lbfgs_armijo_nonmonotone')
relax.dualspace(False)
relax.set_scorefxn(sf2015)
relax.max_iter(200)

for input_file_idx,input_file in enumerate(process_list):
	new_structure_names = struct_name_list[input_file_idx]
	old_structure_names = old_struct_name_list[input_file_idx]
	pdb_file_list = np.genfromtxt(str(analysis_folder_name) + "FloppyTail_Compiled.sc", dtype=str)
	residue_ref2015_array = []
	for pdb_file_idx, pdb_file_item in enumerate(pdb_file_list):
		if (pdb_file_idx - parallel_num) % 25 == 0:
			p = pose_from_pdb(str(analysis_folder_name) + str(pdb_file_item[0]))
			residue_ref2015_set = []
			relax.apply(p)
			sf2015(p)
			sf2015.show(p)
			for res_num in range(p.total_residue()):
				residue_ref2015_set.append(p.energies().residue_total_energy(res_num+1))
			residue_ref2015_array.append(residue_ref2015_set)
			pdb_out_name = str(new_structure_names) + str(pdb_file_idx) + '.pdb'
			os.chdir(new_directory)
			outf = open(str(rewrite_list[input_file_idx]) + str(parallel_num) + '.sc', 'a')
			outf.write("%s\t%s\t%.4f\t%.4f\n" % (pdb_out_name, str(pdb_file_item[0]), sf2015(p), sfrg(p)))
			outf.close()
			p.dump_pdb(pdb_out_name)
			os.chdir('..')
		else:
			continue
	residue_ref2015_array = np.average(residue_ref2015_array, axis=0)
	os.chdir(new_directory)
	final_ref2015_array_file = str(perresval_list[input_file_idx]) + str(parallel_num) + '.txt'
	np.savetxt(final_ref2015_array_file, residue_ref2015_array, fmt='%s', delimiter=' ')

# Combining all of the per residue energy files into a singular output
if parallel_num == 0:
	os.chdir('..')
	os.makedirs(new_parent_directory)
	os.chmod(new_parent_directory,0o777)
	wait_for_others = True
	files_for_averaging = []
	while wait_for_others == True:
		for in_fil_idx, in_fil_item in enumerate(perresval_list):
			check_for_file_set = str(working_dir) + '*/' + str(perresval_list[in_fil_idx]) + '*.txt'
			files_for_final_averaging = glob.glob(check_for_file_set)
		if len(files_for_final_averaging) == 25:
			wait_for_others = False
		else:
			time.sleep(300)
		print('Not ready just yet =(')
	print('Ready to get the average')	
	for in_fil_idx, in_fil in enumerate(rewrite_list):
		check_for_file_set = str(working_dir) + '*/' + str(perresval_list[in_fil_idx]) + '*.txt'
		check_for_sc_file_set = str(working_dir) + '*/' + str(rewrite_list[in_fil_idx]) + '*.sc'
		files_for_final_averaging = glob.glob(check_for_file_set)
		files_for_final_compilation = glob.glob(check_for_sc_file_set)
		final_averaging_ref2015_array = []
		for sub_file in files_for_final_averaging:
			sub_file_array = np.genfromtxt(sub_file)
			final_averaging_ref2015_array.append(sub_file_array)
		for sub_sc_file in files_for_final_compilation:
			sub_file_array = np.genfromtxt(sub_sc_file, dtype=None, encoding=None)
			for sc_line in sub_file_array:
				pdb_file_for_copy = sub_sc_file.split('/')[0] + '/' + sc_line[0]
				copy(pdb_file_for_copy, new_parent_directory)
		final_averaging_ref2015_array = np.average(final_averaging_ref2015_array, axis=0)
		os.chdir(new_parent_directory)
		np.savetxt(str(perresval_list[in_fil_idx]) + '_Compiled.txt', final_averaging_ref2015_array, fmt='%s', delimiter=' ')
		np.savetxt('What_was_analyzed.txt', files_for_final_averaging, fmt='%s', delimiter=' ')		
		rmsd_color_struct = parser.get_structure('native', str(new_structure_names) + '0.pdb')
		for rmsd_color_res_idx, rmsd_color_res in enumerate(rmsd_color_struct[0]['A']):
			for rmsd_color_atom in rmsd_color_res:
				rmsd_color_atom.bfactor = final_averaging_ref2015_array[rmsd_color_res_idx]
		io.set_structure(rmsd_color_struct)
		io.save(color_pdb_name_list[in_fil_idx])		