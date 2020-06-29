# Syntax run ReWrite.py OneLetterOrderCode StartResidue EndResidue Native.pdb sub_directory_*/
## OneLetterOrderCodes: O = ordered (compares RMSD to native pdb), P = partially ordered (compares RMSD to segment of native PDB), D = disordered (compares RMSD to lowest energy)
#Identical to AbInitioAnalysis_062419
import sys
from numpy import genfromtxt
import numpy as np
import os
from shutil import copy
from Bio.PDB import *

# Creates a single array/file that contains all scores from all output structures
abin_out_file = "AbInitio_FullAtom.sc"
rewrite_abin_out_file = "Compiled_AbInitio_FullAtom.sc"
abin_vo_out_file = "AbInitioVO_FullAtom.sc"
rewrite_abin_vo_out_file = "Compiled_AbInitioVO_FullAtom.sc"
new_directory = 'Partially_Ordered_ReAnalysis'
new_abin_structure_names = 'AbInitio_'
new_abin_vo_structure_name = 'AbInitioVO_'
os.makedirs(new_directory)
os.chmod(new_directory,0o777)
process_list = [abin_vo_out_file]
rewrite_list = [rewrite_abin_vo_out_file]
struct_name_list = [new_abin_vo_structure_name]

#input_file ='Relax_Pap_CL_Pep.sc'
#rewrite_file = 'minimum_score.fasc'
#new_directory = 'Minimum_Full_Score_Relax_FH_ALL'
#new_structure_names = 'Minimum_Full_Score_'

for input_file_idx,input_file in enumerate(process_list):
	new_structure_names = struct_name_list[input_file_idx]
	full_file_name_array = np.empty((0,), dtype=[('fa_name', 'S130'), ('cen_name', 'S130'), ('fa_score', '<f8'), ('rg', '<f8')]) # Match value to number of columns of data in output file
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
	rmsd_array = np.empty([len(new_list),], dtype=[('fa_name', 'S130'), ('cen_name', 'S130'), ('fa_score', '<f8'), ('rg', '<f8'), ('rmsd', '<f8')])
	parser = PDBParser()
	native_atom_list = []
	full_native_atom_list = []
	native_resi_list = []
	# Setting up the RMSD Computation
	if order_code == 'O':
		native_struct = parser.get_structure('native', native_structure_pdb)
		native_atoms_gen = native_struct.get_residues()
		native_atom_list = []
		for atom in native_atoms_gen:
			native_atom_list.append(atom['CA'])
	elif order_code == 'P':
		native_struct = parser.get_structure('native', native_structure_pdb)
		native_res_gen = native_struct.get_residues()
		full_nat_res_list = []
		for res in native_res_gen:
			full_nat_res_list.append(res)
			full_native_atom_list.append(res['CA'])
		start_res_nat = int(full_nat_res_list[0].id[1])
		dssp_start_res = 1
		native_model = native_struct[0]
		dssp = DSSP(native_model, native_structure_pdb)
		for resi_dssp_idx,resi_dssp in enumerate(dssp):
			if resi_dssp_idx == 0:
				dssp_start_res = int(resi_dssp[0])
			if resi_dssp[2] == 'G' or resi_dssp[2] == 'H' or resi_dssp[2] == 'I' or resi_dssp[2] == 'T' or resi_dssp[2] == 'E':
				current_res = resi_dssp[0]-dssp_start_res+start_res_nat
				native_atom_list.append(native_model['A'][current_res]['CA'])
				native_resi_list.append(current_res)		
	else:
		native_struct = parser.get_structure('native', str(new_list[0]['fa_name'].decode('UTF-8')))
		native_atoms_gen = native_struct.get_residues()
		native_atom_list = []
		for atom in native_atoms_gen:
			native_atom_list.append(atom['CA'])
	# Performing Superimpose and RMSD Compute
	sup = Superimposer()
	for idx,item in enumerate(new_list[0:]):
		path_item = item['fa_name']
		path_item = path_item.decode('UTF-8')
		print(path_item)
		test_struct = parser.get_structure('test', str(path_item))
		test_residue_gen = test_struct.get_residues()
		test_atom_list = []
		test_full_atom_list = []
		if order_code == 'P':
			for residue in test_residue_gen:
				res_other_1, res_num, res_other_2 = residue.get_id()
				if res_num >= start_res and res_num <= end_res:
					test_full_atom_list.append(residue['CA'])
					if res_num in native_resi_list:
						test_atom_list.append(residue['CA'])
		else:
			for residue in test_residue_gen:
				test_atom_list.append(residue['CA'])
		sup.set_atoms(native_atom_list, test_atom_list)
		print(sup.rms)	
		for sub_item_idx, sub_item_item in enumerate(item):
			rmsd_array[idx][sub_item_idx] = sub_item_item	
		rmsd_array[idx]['rmsd'] = sup.rms	
	rmsd_array = rmsd_array[rmsd_array['rmsd'].argsort(kind='mergesort')] # Match column to the column of interest for sorting the data
	for idx,item in enumerate(rmsd_array[0:10]):
		path_item = item['fa_name']
		path_item = path_item.decode('UTF-8')
		print(path_item)
		copy(path_item, new_directory)
		os.chdir(new_directory)
		i_piece=path_item.split('/')[1]
		new_i_name = new_structure_names + str(idx) + '.pdb'
		os.rename(i_piece, new_i_name)
		os.chdir('..')
	os.chdir(new_directory)
	rewrite_file = rewrite_list[input_file_idx]
	np.savetxt(rewrite_file, rmsd_array, fmt='%s', delimiter=' ', newline='\n')
	os.chdir('..')
