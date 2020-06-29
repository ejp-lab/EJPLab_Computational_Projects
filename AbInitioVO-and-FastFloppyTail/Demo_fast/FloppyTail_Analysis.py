# Syntax run ReWrite.py OneLetterOrderCode StartResidue EndResidue Native.pdb sub_directory_*/
## OneLetterOrderCodes: O = ordered (compares RMSD to native pdb), P = partially ordered (compares RMSD to segment of native PDB), D = disordered (compares RMSD to lowest energy)
import sys
from numpy import genfromtxt
import numpy as np
import os
from shutil import copy
from Bio.PDB import *
import glob

# Creates a single array/file that contains all scores from all output structures
abin_out_file = "FloppyTail.sc"
rewrite_abin_out_file = "FloppyTail_Compiled.sc"
new_directory = 'Post_FloppyTail_Selection'
new_abin_structure_names = 'FloppyTail_'
color_pdb_abin = 'FloppyTail_Color_by_RMSD.pdb'
os.makedirs(new_directory)
os.chmod(new_directory,0o777)
process_list = [abin_out_file]
rewrite_list = [rewrite_abin_out_file]
struct_name_list = [new_abin_structure_names]
color_pdb_name_list = [color_pdb_abin]
parser = PDBParser()
io = PDBIO()
sup = Superimposer()

for input_file_idx,input_file in enumerate(process_list):
	new_structure_names = struct_name_list[input_file_idx]
	full_file_name_array = np.empty((0,), dtype=[('fa_name', 'S130'), ('fa_score', '<f8'), ('rg', '<f8')]) # Match value to number of columns of data in output file
	for i in sys.argv[5:]:
		directory_list_inter = str(i)
		scoring_file = directory_list_inter + input_file
		lines_names = np.genfromtxt(scoring_file, dtype=full_file_name_array.dtype)
		part_file_name_array = np.empty((0,0), dtype='|S100')
		for j in range(len(lines_names)):
			full_file_name = directory_list_inter + str(lines_names[j][0].decode('UTF-8'))
			part_file_name_array = np.append(part_file_name_array, full_file_name)
		lines_names['fa_name'] = part_file_name_array
		full_file_name_array = np.concatenate((full_file_name_array, lines_names))
	new_list = full_file_name_array[full_file_name_array['fa_score'].argsort(kind='mergesort')] # Match column to the column of interest for sorting the data
	## Figure out which way to run the RMSD comparison
	order_code = str(sys.argv[1])
	start_res = int(sys.argv[2])
	end_res = int(sys.argv[3])
	native_structure_pdb = str(sys.argv[4])
	rmsd_array = np.empty([1000,], dtype=[('new_name', 'U130'), ('fa_name', 'U130'), ('fa_score', '<f8'), ('rg', '<f8')])
	native_struct = parser.get_structure('native', str(new_list[0]['fa_name'].decode('UTF-8')))
	
## Place Top 1000 in Single Directory
	for idx,item in enumerate(new_list[0:1000]):
		path_item = item['fa_name']
		path_item = path_item.decode('UTF-8')
		print(path_item)
		copy(path_item, new_directory)
		os.chdir(new_directory)
		i_piece=path_item.split('/')[1]
		new_i_name = new_structure_names + str(idx) + '.pdb'
		os.rename(i_piece, new_i_name)
		os.chdir('..')
		rmsd_array[idx]['fa_name'] = new_list[idx]['fa_name']
		rmsd_array[idx]['fa_score'] = new_list[idx]['fa_score']
		rmsd_array[idx]['rg'] = new_list[idx]['rg']
		rmsd_array[idx]['new_name'] = new_i_name
	os.chdir(new_directory)
	rewrite_file = rewrite_list[input_file_idx]
	np.savetxt(rewrite_file, rmsd_array, fmt='%s', delimiter=' ', newline='\n')
	os.chdir('..')