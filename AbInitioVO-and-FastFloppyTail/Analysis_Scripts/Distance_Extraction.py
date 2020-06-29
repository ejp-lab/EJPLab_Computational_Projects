# The Start-Up
from math import exp, log, pi, sqrt
from random import random as rnd
import numpy as np
import sys
from operator import itemgetter
from numpy import genfromtxt
from scipy import *
import scipy.spatial.distance as d

constraints = sys.argv[1]
output_name = sys.argv[2]
avg_map = sys.argv[3]
std_map = sys.argv[4]

cst_file = open(constraints).readlines()
cst_matrix_in = genfromtxt(constraints)
cst_matrix_out = np.append(cst_matrix_in, np.zeros([len(cst_matrix_in),2]),1)
avg_matrix = genfromtxt(avg_map)
std_matrix = genfromtxt(std_map)
average_matrix = []
stand_matrix = []
rmsd_matrix = []
for line in cst_file:
	if len(line) > 0:
		words = line.split()
		residue_a = int(words[0]) - 1
		residue_b = int(words[1]) - 1
		cst_val = float(words[2])
		avg_dist = avg_matrix[residue_a][residue_b]
		average_matrix.append(avg_dist)
		std_dist = std_matrix[residue_a][residue_b]
		stand_matrix.append(std_dist)	
		rmsd_matrix.append((avg_dist - cst_val)**2)
		
for i in range(0,len(cst_matrix_in)):
	cst_matrix_out[i][2] = average_matrix[i]
	cst_matrix_out[i][3] = stand_matrix[i]
np.savetxt(output_name, cst_matrix_out, delimiter=' ', newline='\n')

rmsd_val = [np.sqrt(np.average(rmsd_matrix))]
np.savetxt('Distances_RMSD.txt', rmsd_val, fmt='%s')
