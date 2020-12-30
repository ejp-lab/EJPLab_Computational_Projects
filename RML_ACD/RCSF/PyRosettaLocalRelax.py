import os 
import sys 
import glob
import shutil 
import numpy as np 
from numpy import linalg as LA
import pandas as pd 
import Bio
from Bio.PDB.PDBParser import PDBParser
from pyrosetta import *
init(extra_options='-beta -corrections::beta_nov16 -extra_res_fa acd.params -ex1 -ex2 -ex2aro -score:fa_max_dis 9 -ignore_zero_occupancy False')

def FindNeighborResidues(pdb, resnum):

    parser = PDBParser()

    neighbor_list = []
    acridone_calpha = []

    structure = parser.get_structure('pdbf', pdb)
    model = structure[0]
    for chain in model:
        if chain.full_id[2] == 'A':
            for residue in chain:
                if residue.get_id()[1] == resnum:
                    for atom in residue:                    
                        if str(atom) == '<Atom CA>':             
                            atm1_vect = list(atom.get_vector())
                            atm1_vect = np.array(atm1_vect)  
                            acridone_calpha.append(atm1_vect) 
                        else:
                            pass                              
                else:
                    pass              
        else:
            pass

    for chain in model:
        for residue in chain:
            for atom in residue:                    
                if str(atom) == '<Atom CA>':             
                    atm2_vect = list(atom.get_vector())
                    atm2_vect = np.array(atm2_vect)  
                    distance = LA.norm(atm2_vect - acridone_calpha[0])
                    if distance <= 8.0:
                        neighbor_list.append(residue.get_id()[1])
                    else:
                        pass            
                else:
                    pass            

    neighbor_list.append(resnum)
    neighbor_list = sorted(list(set(neighbor_list)))   

    return neighbor_list

def MutateToAcridone(p, res_num):

    mutator = pyrosetta.rosetta.protocols.simple_moves.MutateResidue(res_num, 'ACD')
    mutator.apply(p)

    return p

def MakeMoveMap(neighbor_list):
    
    mvmap = MoveMap() 
    mvmap.set_bb(False) 
    mvmap.set_chi(False)

    for res in neighbor_list:
        mvmap.set_bb(res, True) 
        mvmap.set_chi(res, True) 

    return mvmap

def FastRelax(p, sf, mvmap):

    fast_relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    fast_relax.set_scorefxn(sf)
    fast_relax.set_movemap(mvmap)
    fast_relax.constrain_relax_to_start_coords(False)
    fast_relax.dualspace(True)
    fast_relax.minimize_bond_angles(True)
    fast_relax.min_type('lbfgs_armijo_nonmonotone')

    fast_relax.apply(p)

    return p

if __name__ == "__main__":

    prot_name = sys.argv[1]
    position = sys.argv[2]

    starting_pose = pose_from_pdb(prot_name)

    for trial in range(1,6):

        wt_control = Pose()
        wt_control.assign(starting_pose)

        mutated_pose = Pose()
        mutated_pose.assign(starting_pose)  

        nb_list = FindNeighborResidues(prot_name, int(position))
        mutated_pose = MutateToAcridone(mutated_pose, int(position))

        poses = [mutated_pose, wt_control]
        outnames = [prot_name.split('.')[0] + '_' + position + '_mutated_' + str(trial) + '.pdb', prot_name.split('.')[0] + '_' + position + '_WT_' + str(trial) + '.pdb']                    

        for idx, pose in enumerate(poses):
            movemap = MakeMoveMap(nb_list)    
            scorefunction = create_score_function('beta_nov16_cart')

            relaxed_pose = FastRelax(pose, scorefunction, movemap)
            relaxed_pose.dump_pdb(outnames[idx])