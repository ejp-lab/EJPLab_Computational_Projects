import os 
import sys 
import glob 
import shutil 
import numpy as np 
import scipy as sp
from scipy import *
import scipy.spatial
import numpy.linalg as LA 
from numpy import cross, eye, dot
from scipy.linalg import expm, norm
import pandas as pd 
import itertools
import re
import time
import argparse
from Bio.PDB import *
pdb_parser = PDBParser()

parser = argparse.ArgumentParser(description='Program')
parser.add_argument('-a', '--Analysis', action='store', type=str, required=False,
	default='PolymerDssp', help='Name of the analysis you want to perform on the data, can specify multiple options in camelcase: Polymer / Dssp')
parser.add_argument('-i', '--InputStructures', action='store', type=str, required=False,
	help='Structure name or search name for import of structures')
parser.add_argument('-i2', '--InputStructures2', action='store', type=str, required=False,
	help='Structure name or search name for comparison structures, such as wild-type')
parser.add_argument('-o', '--OutputName', action='store', type=str, required=False,
	help='Prefix of outputs')
parser.add_argument('-contact', '--ContactDistance', action='store', type=float, required=False,
	default=10.0, help='Distance cutoff for assigning and interresidue contact')
parser.add_argument('-residue', '--FocalResidue', action='store', type=int, required=False,
	help='Specific residue of interest for performing additional analyses')
parser.add_argument('-domain', '--FocalDomain', action='store', type=str, required=False,
	help='Specific domain of interest for performing additional analyses identified as start_residue,end_residue')
parser.add_argument('-fold', '--FoldRef', action='store', type=int, required=False,
	help='Single reference structure for counting number of similar contacts')

args = parser.parse_args()

aa_acc_max   = { \
        	   'A': 129.0, 'R': 274.0, 'N': 195.0, 'D': 193.0,\
        	   'C': 167.0, 'Q': 225.0, 'E': 223.0, 'G': 104.0,\
        	   'H': 224.0, 'I': 197.0, 'L': 201.0, 'K': 236.0,\
        	   'M': 224.0, 'F': 240.0, 'P': 159.0, 'S': 155.0,\
 	           'T': 172.0, 'W': 285.0, 'Y': 263.0, 'V': 174.0}

def collectPDBCoords(pdb_file):
	pdb_coords = []
	pdb_lines = filter(lambda s: ' CA ' in s, open(pdb_file).readlines())
	for line in pdb_lines:
		if 'HETATOM' in line:
			line_info = line.split()
			pdb_coords.append([float(line_info[6]), float(line_info[7]), float(line_info[8])])
		elif 'ATOM' in line:
			pdb_coords.append([float(line[30:38]), float(line[39:46]), float(line[47:55])])
	return pdb_coords
	
'''	open_files = open(pdb_file)
	pdb_lines = open_file.readlines()
	pdb_coords = []
	for line in pdb_lines:
		if 'ATOM' in line or 'HET' in line:
			line_info = line.split()
			if line_info[2] == 'CA':
				pdb_coords.append([float(line_info[6]), float(line_info[7]), float(line_info[8])])'''
				
	

def computeDistanceContactInfo(pdb_coords):
	n_residues = len(pdb_coords)
	distance_map = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(pdb_coords))
	contact_map = where(distance_map < args.ContactDistance, ones_like(distance_map), zeros_like(distance_map))
	contact_number = np.sum(contact_map)
	rad_gyr = np.sqrt(np.sum(distance_map**2)/(2*n_residues**2))
	return distance_map, contact_map, contact_number, rad_gyr, n_residues

'''	distance_map = np.zeros([len(pdb_coords),len(pdb_coords)])
	contact_map = np.zeros([len(pdb_coords),len(pdb_coords)])
	contact_number = 0
	for pdb_coord_idx_1, pdb_coord_item_1 in enumerate(pdb_coords):
		for pdb_coord_idx_2, pdb_coord_item_2 in enumerate(pdb_coords):
			interresidue_distance = np.linalg.norm(np.array(pdb_coord_item_1) - np.array(pdb_coord_item_2))
			distance_map[pdb_coord_idx_1][pdb_coord_idx_2] = interresidue_distance
			if interresidue_distance < contact_distance:
				contact_map[pdb_coord_idx_1][pdb_coord_idx_2] = 1.0
				contact_number += 1'''
	
def computePolymerInfo(input_structure_list):
	distance_map_set = []
	contact_map_set = []
	contact_number_set = []
	radii_gyr = []
	end_to_end_dist = []
	for structure in input_structure_list:
		structure_coords = collectPDBCoords(structure)
		distance_map,contact_map,contact_number,rad_gyr,n_residues = computeDistanceContactInfo(structure_coords)
		distance_map_set.append(distance_map); contact_map_set.append(contact_map); contact_number_set.append(contact_number); radii_gyr.append(rad_gyr); end_to_end_dist.append(distance_map[0][0])
	avg_distance_map = np.average(distance_map_set, axis=0)
	std_distance_map = np.std(distance_map_set, axis=0)
	avg_contact_map = np.average(contact_map_set, axis=0)
	avg_distance_scaling = []
	std_distance_scaling = []
	avg_contact_scaling = []
	for residue_offset in range(n_residues):
		avg_distance_scaling.append([residue_offset,np.average(diagonal(avg_distance_map, residue_offset))])
		std_distance_scaling.append([residue_offset,np.sqrt(np.average(diagonal(np.square(std_distance_map), residue_offset)))])
		avg_contact_scaling.append([residue_offset,np.average(diagonal(avg_contact_map, residue_offset))])
	return avg_distance_map, std_distance_map, avg_distance_scaling, std_distance_scaling, avg_contact_scaling, avg_contact_map, contact_number_set, radii_gyr, end_to_end_dist

def residuePolymerInfo(avg_distance_map, std_distance_map, avg_contact_map):
	up_avg_distance_scaling = []; down_avg_distance_scaling = []; up_std_distance_scaling = []; down_std_distance_scaling = []
	up_contact_scaling = []; down_contact_scaling = []
	for residue_offset in range(args.FocalResidue):
		up_avg_distance_scaling.append([residue_offset,np.average(diagonal(avg_distance_map[:args.FocalResidue,:args.FocalResidue], residue_offset))])
		up_std_distance_scaling.append([residue_offset,np.sqrt(np.average(diagonal(np.square(std_distance_map[:args.FocalResidue,:args.FocalResidue]), residue_offset)))])
		up_contact_scaling.append([residue_offset,np.average(diagonal(avg_contact_map[:args.FocalResidue,:args.FocalResidue], residue_offset))])
	for residue_offset in range(args.FocalResidue,len(avg_distance_map),1):	
		down_avg_distance_scaling.append([residue_offset-args.FocalResidue,np.average(diagonal(avg_distance_map[args.FocalResidue:,args.FocalResidue:], residue_offset-args.FocalResidue))])
		down_std_distance_scaling.append([residue_offset-args.FocalResidue,np.sqrt(np.sum(diagonal(np.square(std_distance_map[args.FocalResidue:,args.FocalResidue:]), residue_offset-args.FocalResidue)))])
		down_contact_scaling.append([residue_offset-args.FocalResidue,np.average(diagonal(avg_contact_map[args.FocalResidue:,args.FocalResidue:], residue_offset-args.FocalResidue))])
	return up_avg_distance_scaling, down_avg_distance_scaling, up_std_distance_scaling, down_std_distance_scaling, up_contact_scaling, down_contact_scaling 
	
def domainPolymerInfo(avg_distance_map, std_distance_map, avg_contact_map):
	domain_start, domain_end = args.FocalDomain.split(',')
	domain_start = int(domain_start); domain_end = int(domain_end)
	up_avg_distance_scaling = []; down_avg_distance_scaling = []; up_std_distance_scaling = []; down_std_distance_scaling = []
	up_contact_scaling = []; down_contact_scaling = []
	for residue_offset in range(domain_start):
		up_avg_distance_scaling.append([residue_offset,np.average(diagonal(avg_distance_map[:domain_start,:domain_start], residue_offset))])
		up_std_distance_scaling.append([residue_offset,np.sqrt(np.average(diagonal(np.square(std_distance_map[:domain_start,:domain_start]), residue_offset)))])
		up_contact_scaling.append([residue_offset,np.average(diagonal(avg_contact_map[:domain_start,:domain_start], residue_offset))])
	for residue_offset in range(domain_end,len(avg_distance_map),1):	
		down_avg_distance_scaling.append([residue_offset-domain_end,np.average(diagonal(avg_distance_map[domain_end:,domain_end:], residue_offset-domain_end))])
		down_std_distance_scaling.append([residue_offset-domain_end,np.sqrt(np.average(diagonal(np.square(std_distance_map[domain_end:,domain_end:]), residue_offset-domain_end)))])
		down_contact_scaling.append([residue_offset-domain_end,np.average(diagonal(avg_contact_map[domain_end:,domain_end:], residue_offset-domain_end))])
	return up_avg_distance_scaling, down_avg_distance_scaling, up_std_distance_scaling, down_std_distance_scaling, up_contact_scaling, down_contact_scaling 

def contactFoldMatching(input_structure_list, avg_contact_map):
	ref_struct_coods = collectPDBCoords(args.FoldRef)
	ref_distance_map,ref_contact_map,ref_contact_number,ref_rad_gyr,ref_n_residues = conputeDistanceContactInfo(ref_structure_coords)
	matching_contact_prob = ref_contact_map * avg_contact_map
	matching_contact_number = (np.sum(matching_contact_prob)/2)*len(input_structure_list)
	return matching_contact_prob, matching_contact_number

def computeDSSPInfo(input_structure_list):
	struct = pdb_parser.get_structure('test', input_structure_list[0])
	n_residues = 0
	for residue in struct.get_residues():
		n_residues += 1
	percent_ss2 = np.zeros([n_residues, 3])
	phi_list = []
	psi_list = []
	TASA_list = []
	for structure in input_structure_list:
		struct = pdb_parser.get_structure('test', structure)
		model = struct[0]
		struct_TASA_list = []
		dssp = DSSP(model, structure)
		for resi_dssp_idx, resi_dssp in enumerate(dssp):
			dssp_idx = resi_dssp[0]
			AA = resi_dssp[1]
			SS = resi_dssp[2]
			if SS == 'E' or SS == 'B':
				percent_ss2[resi_dssp_idx][1] += 1
			elif SS == 'H' or SS == 'G' or SS == 'I':
				percent_ss2[resi_dssp_idx][0] += 1
			else:
				percent_ss2[resi_dssp_idx][2] += 1
			RASA = resi_dssp[3]
			TASA = RASA*aa_acc_max[AA]
			struct_TASA_list.append(TASA)
			Phi = resi_dssp[4]
			phi_list.append(Phi)
			Psi = resi_dssp[5]
			psi_list.append(Psi)
		TASA_list.append(sum(struct_TASA_list))
	percent_ss2 /= len(input_structure_list)
	phi_psi_hist, phi_psi_hist_x, phi_psi_hist_y = np.histogram2d(phi_list, psi_list, bins=121, range=[[-180,180], [-180,180]])
	phi_psi_hist /= (n_residues*len(input_structure_list))
	return percent_ss2, phi_psi_hist, TASA_list, n_residues

def computeSizeLandscape(input_structure_list, radii_gyr, TASA_list):
	size_landscape, size_landscape_x, size_landscape_y = np.histogram2d(radii_gyr, TASA_list, bins=101, range=[[min(radii_gyr)-5,max(radii_gyr)+5], [min(TASA_list)-100,max(TASA_list)+100]])
	size_landscape /= float(len(input_structure_list))
	return size_landscape, size_landscape_x, size_landscape_y
	
##### Running
## Importing and Analyzing a Single Ensemble
ins = str(args.InputStructures) + '*.pdb'
input_structure_list = glob.glob(ins)
if 'Polymer' in args.Analysis:
	avg_distance_map, std_distance_map, average_distance_scaling, std_distance_scaling, avg_contact_scaling, avg_contact_map, contact_number_set, radii_gyr, end_to_end_dist = computePolymerInfo(input_structure_list)
if 'Dssp' in args.Analysis:	
	percent_ss2, phi_psi_hist, TASA_list, n_residues = computeDSSPInfo(input_structure_list)
if 'Dssp' in args.Analysis and 'Polymer' in args.Analysis:	
	size_landscape, size_landscape_x, size_landscape_y = computeSizeLandscape(input_structure_list, radii_gyr, TASA_list)

## Performing Analysis against a Comparative Ensemble
if args.InputStructures2:
	ins_2 = str(args.InputStructures2) + '*.pdb'
	input_structure_list_2 = glob.glob(ins_2)
	if 'Polymer' in args.Analysis:	
		avg_distance_map_2, std_distance_map_2, average_distance_scaling_2, std_distance_scaling_2, avg_contact_scaling_2, avg_contact_map_2, contact_number_set_2, radii_gyr_2, end_to_end_dist_2 = computePolymerInfo(input_structure_list_2)
		diff_distance_map = avg_distance_map - avg_distance_map_2
		diff_contact_map = avg_contact_map - avg_contact_map_2
	if 'Dssp' in args.Analysis:	
		percent_ss2_2, phi_psi_hist_2, TASA_list_2, n_residues_2 = computeDSSPInfo(input_structure_list_2)
		diff_phi_psi_hist = phi_psi_hist - phi_psi_hist_2
		diff_ss2 = percent_ss2 - percent_ss2_2
	if 'Dssp' in args.Analysis and 'Polymer' in args.Analysis:	
		size_landscape_2, size_landscape_x_2, size_landscape_y_2 = computeSizeLandscape(input_structure_list_2, radii_gyr_2, TASA_list_2)
		
## Performing Analysis for a specific focal residue
if args.FocalResidue:
	if 'Polymer' in args.Analysis:
		up_avg_distance_scaling, down_avg_distance_scaling, up_std_distance_scaling, down_std_distance_scaling, up_contact_scaling, down_contact_scaling = residuePolymerInfo(avg_distance_map, std_distance_map, avg_contact_map)
		if args.InputStructures2:
			up_avg_distance_scaling_2, down_avg_distance_scaling_2, up_std_distance_scaling_2, down_std_distance_scaling_2, up_contact_scaling_2, down_contact_scaling_2 = residuePolymerInfo(avg_distance_map_2, std_distance_map_2, avg_contact_map_2)
		
## Performing Analysis for a specific focal domain
if args.FocalDomain:
	if 'Polymer' in args.Analysis:
		up_avg_distance_scaling, down_avg_distance_scaling, up_std_distance_scaling, down_std_distance_scaling, up_contact_scaling, down_contact_scaling = domainPolymerInfo(avg_distance_map, std_distance_map, avg_contact_map)
		if args.InputStructures2:
			up_avg_distance_scaling_2, down_avg_distance_scaling_2, up_std_distance_scaling_2, down_std_distance_scaling_2, up_contact_scaling_2, down_contact_scaling_2 = domainPolymerInfo(avg_distance_map_2, std_distance_map_2, avg_contact_map_2)

## Performing Analysis for a specific contact set based on a single structure
if args.FoldRef:
	if 'Polymer' in args.Analysis:
		matching_contact_prob, matching_contact_number = contactFoldMatching(input_structure_list, avg_contact_map)
		if args.InputStructures2:
			matching_contact_prob_2, matching_contact_number_2 = contactFoldMatching(input_structure_list_2, avg_contact_map_2)

## Final analyses
if 'Polymer' in args.Analysis:
	rg_hist, rg_hist_axis = np.histogram(radii_gyr, bins=50, range=[0,100])
	if args.InputStructures2:
		rg_hist_2, rg_hist_axis_2 = np.histogram(radii_gyr_2, bins=50, range=[0,100])
if 'Dssp' in args.Analysis:
	TASA_hist, TASA_hist_axis = np.histogram(TASA_list, bins=50, range=[min(TASA_list),max(TASA_list)])
	if args.InputStructures2:
		TASA_hist_2, TASA_hist_axis_2 = np.histogram(TASA_list_2, bins=50, range=[min(TASA_list_2),max(TASA_list_2)])
		
## Writing the Outputs
if 'Polymer' in args.Analysis:
	if args.OutputName:
		np.savetxt(args.OutputName + '_Polymer_Avg_Distance_Map.txt', avg_distance_map, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Polymer_Std_Distance_Map.txt', std_distance_map, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Polymer_Avg_Distance_Scaling.txt', average_distance_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Polymer_Std_Distance_Scaling.txt', std_distance_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Polymer_Avg_Contact_Scaling.txt', avg_contact_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Polymer_Avg_Contact_Map.txt', avg_contact_map, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Polymer_Contacts_Per_Structure.txt', contact_number_set, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Polymer_Rg_Histogram.txt', rg_hist, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Polymer_Rg_Histogram_Axis.txt', rg_hist_axis, fmt='%s', delimiter=' ', newline='\n')
		
	else:	
		np.savetxt('Polymer_Avg_Distance_Map.txt', avg_distance_map, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Polymer_Std_Distance_Map.txt', std_distance_map, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Polymer_Avg_Distance_Scaling.txt', average_distance_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Polymer_Std_Distance_Scaling.txt', std_distance_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Polymer_Avg_Contact_Scaling.txt', avg_contact_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Polymer_Avg_Contact_Map.txt', avg_contact_map, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Polymer_Contacts_Per_Structure.txt', contact_number_set, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Polymer_Rg_Histogram.txt', rg_hist, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Polymer_Rg_Histogram_Axis.txt', rg_hist_axis, fmt='%s', delimiter=' ', newline='\n')
		
if 'Dssp' in args.Analysis:
	if args.OutputName:
		np.savetxt(args.OutputName + '_Dssp_Percent_SS2.txt', percent_ss2, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Dssp_Phi_Psi_Histogram.txt', phi_psi_hist, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Dssp_TASA_Histogram.txt', TASA_hist, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Dssp_TASA_Histogram_Axis.txt', TASA_hist_axis, fmt='%s', delimiter=' ', newline='\n')
	else:
		np.savetxt('Dssp_Percent_SS2.txt', percent_ss2, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Dssp_Phi_Psi_Histogram.txt', phi_psi_hist, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Dssp_TASA_Histogram.txt', TASA_hist, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Dssp_TASA_Histogram_Axis.txt', TASA_hist_axis, fmt='%s', delimiter=' ', newline='\n')
		
if 'Dssp' in args.Analysis and 'Dssp' in args.Analysis:
	if args.OutputName:
		np.savetxt(args.OutputName + '_Size_Landscape.txt', size_landscape, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Size_Landscape_X.txt', size_landscape_x, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Size_Landscape_Y.txt', size_landscape_y, fmt='%s', delimiter=' ', newline='\n')
	else:	
		np.savetxt('Size_Landscape.txt', size_landscape, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Size_Landscape_X.txt', size_landscape_x, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Size_Landscape_Y.txt', size_landscape_y, fmt='%s', delimiter=' ', newline='\n')

if args.FocalDomain:
	if args.OutputName:
		np.savetxt(args.OutputName + '_Domain_Up_Distance_Avg_Scaling.txt', up_avg_distance_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Domain_Down_Distance_Avg_Scaling.txt', down_avg_distance_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Domain_Up_Distance_Std_Scaling.txt', up_std_distance_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Domain_Down_Distance_Std_Scaling.txt', down_std_distance_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Domain_Up_Contact_Scaling.txt', up_contact_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Domain_Down_Contact_Scaling.txt', down_contact_scaling, fmt='%s', delimiter=' ', newline='\n')
	else:
		np.savetxt('Domain_Up_Distance_Avg_Scaling.txt', up_avg_distance_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Domain_Down_Distance_Avg_Scaling.txt', down_avg_distance_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Domain_Up_Distance_Std_Scaling.txt', up_std_distance_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Domain_Down_Distance_Std_Scaling.txt', down_std_distance_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Domain_Up_Contact_Scaling.txt', up_contact_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Domain_Down_Contact_Scaling.txt', down_contact_scaling, fmt='%s', delimiter=' ', newline='\n')

if args.FocalResidue:
	if args.OutputName:
		np.savetxt(args.OutputName + '_Residue_Up_Distance_Avg_Scaling.txt', up_avg_distance_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Residue_Down_Distance_Avg_Scaling.txt', down_avg_distance_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Residue_Up_Distance_Std_Scaling.txt', up_std_distance_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Residue_Down_Distance_Std_Scaling.txt', down_std_distance_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Residue_Up_Contact_Scaling.txt', up_contact_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Residue_Down_Contact_Scaling.txt', down_contact_scaling, fmt='%s', delimiter=' ', newline='\n')
	else:
		np.savetxt('Residue_Up_Distance_Avg_Scaling.txt', up_avg_distance_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Residue_Down_Distance_Avg_Scaling.txt', down_avg_distance_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Residue_Up_Distance_Std_Scaling.txt', up_std_distance_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Residue_Down_Distance_Std_Scaling.txt', down_std_distance_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Residue_Up_Contact_Scaling.txt', up_contact_scaling, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Residue_Down_Contact_Scaling.txt', down_contact_scaling, fmt='%s', delimiter=' ', newline='\n')

if args.InputStructures2:
	if args.OutputName:
		np.savetxt(args.OutputName + '_Difference_Distance_Map.txt', diff_distance_map, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Difference_Contact_Map.txt', diff_contact_map, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt(args.OutputName + '_Difference_SS2.txt', diff_ss2, fmt='%s', delimiter=' ', newline='\n')
	else:
		np.savetxt('Difference_Distance_Map.txt', diff_distance_map, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Difference_Contact_Map.txt', diff_contact_map, fmt='%s', delimiter=' ', newline='\n')
		np.savetxt('Difference_SS2.txt', diff_ss2, fmt='%s', delimiter=' ', newline='\n')
	
if args.FoldRef:
	if args.OutputName:
		np.savetxt(args.OutputName + '_Matched_Contact_Probability_Map.txt', matching_contact_prob, fmt='%s', delimiter=' ', newline='\n')
	else:
		np.savetxt('Matched_Contact_Probability_Map.txt', matching_contact_prob, fmt='%s', delimiter=' ', newline='\n')
	
'''	size_landscape_x, size_landscape_y
	
	np.savetxt('Polymer_Landscape.txt', end_to_end_dist, fmt='%s', delimiter=' ', newline='\n')

	radii_gyr
	end_to_end_dist
	np.savetxt(rewrite_file, rmsd_array, fmt='%s', delimiter=' ', newline='\n')
	if args.InputStructures2:
	elif args.FocalResidue:	
	elif args.FocalDomain:
	elif args.FoldRef:
		
if 'Dssp' in args.Analysis:
	if args.InputStructures2:'''