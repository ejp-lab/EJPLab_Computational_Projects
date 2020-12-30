import os 
import sys 
import glob 
import re 
import yaml
import numpy as np 
import numpy.linalg as LA
import pandas as pd 
import Bio
import Bio.PDB
from Bio.PDB.PDBParser import PDBParser
from Bio import BiopythonWarning
import warnings
import doctest

parser = PDBParser()
doctest.testmod()
warnings.simplefilter('ignore', BiopythonWarning)

cwd = os.getcwd() + '/'
MAX_ASA_TIEN = {'A':129.0, 'R':274.0, 'N':195.0, 'D':193.0, 'C':167.0,
                'E':223.0, 'Q':225.0, 'G':104.0, 'H':224.0, 'I':197.0,
                'L':201.0, 'K':236.0, 'M':224.0, 'F':240.0, 'P':159.0,
                'S':155.0, 'T':172.0, 'W':285.0, 'Y':263.0, 'V':174.0}

def ComputeBioinformaticsMatrices(pdb_name, index_num):

    position = int(pdb_name.split('_')[1])
    position_identity = ''

    structure = parser.get_structure('pdbf', pdb_name)
    model = structure[0]
    for chain in model:
        for residue in chain:
            resnum = residue.get_id()[1]
            if resnum == position:
                position_identity = residue.get_resname()
            else:
                pass    

    matrices = 'BioinformaticsMatrices.txt'
    f = open(matrices, 'r')
    g = f.readlines()
    f.close()

    feature_name_list = ['name']
    feature_list = [(pdb_name.split('_')[0] + '_' + pdb_name.split('_')[1]) + '_' + str(index_num)]
    matrix_list = []

    for line in g:
        split_line = line.split('= ')
        feature_name = split_line[0].strip(' ')
        feature_name_list.append(feature_name)

        dict_string = split_line[1].strip('\n')
        dictt = yaml.load(dict_string)

        matrix_list.append(dictt)

    for matrix in matrix_list:
        feature_list.append(matrix[position_identity])

    dssp_dat = RunDSSP(pdb_name, position)
    dssp_dat = dssp_dat.set_index('site')
    dssp_dat = dssp_dat.loc[[position]]
    vals = dssp_dat.values
    vals = vals[0]

    feature_name_list.append('ASA')
    feature_name_list.append('RSA')

    feature_list.append(vals[0])
    feature_list.append(vals[1])

    d = pd.DataFrame(0, index=np.arange(1), columns=feature_name_list)
    d.loc[0] = feature_list
    d.set_index('name', inplace=True)  

    return d

def RunDSSP(pdb_name, pos):

    chain_list = []
    structure = parser.get_structure('pdbf', pdb_name)
    model = structure[0]
    for chain in model:
        chain_list.append(str(chain).split('id=')[1].split('>')[0])

    name = (pdb_name.split('_')[0] + '_' + pdb_name.split('_')[1])

    dssp_name = name + '_dssp.txt'
    run_input = 'mkdssp -i ' + pdb_name + ' -o ' + dssp_name  

    os.system(run_input) 

    empty = []
    for chain in chain_list:
        dssp_df = ProcessDSSP(dssp_name, chain) 
        dssp_df = dssp_df.replace('-', np.nan) 
        dssp_df = dssp_df.fillna(0)
        empty.append(dssp_df)

    fin_df = pd.concat(empty)

    return fin_df

def ProcessDSSP(dsspfile, chain=None, max_asa=MAX_ASA_TIEN):

    dssp_cys = re.compile('[a-z]')
    d_dssp = Bio.PDB.make_dssp_dict(dsspfile)[0]
    chains = set([chainid for (chainid, r) in d_dssp.keys()])
    if chain is None:
        assert len(chains) == 1, "chain is None, but multiple chains"
        chain = list(chains)[0]
    elif chain not in chains:
        raise ValueError("Invalid chain {0}".format(chain))
    d_df = {'site':[],
            'ASA':[],
            'RSA':[]
            }
    for ((chainid, r), tup) in d_dssp.items():
        if chainid == chain:
            (tmp_aa, ss, acc) = tup[ : 3]
            if dssp_cys.match(tmp_aa):
                aa = 'C'
            else:
                aa = tmp_aa
            if r[2] and not r[2].isspace():
                d_df['site'].append(str(r[1]) + r[2].strip())
            else:
                d_df['site'].append(r[1])
            d_df['ASA'].append(acc)
            d_df['RSA'].append(acc / float(max_asa[aa]))

    return pd.DataFrame(d_df)

def ComputeESFTerms(pdb_name_acd, index_num):

    position = int(pdb_name_acd.split('_')[1])
    position_identity = ''

    acd_postion_polar_bb_vectors = []
    acd_postion_nonpolar_bb_vectors = []
    acd_postion_polar_sc_vectors = []
    acd_postion_nonpolar_sc_vectors = []

    acd_rest_polar_bb_vectors = []
    acd_rest_nonpolar_bb_vectors = []
    acd_rest_polar_sc_vectors = []
    acd_rest_nonpolar_sc_vectors = []

    polar_bb_atoms = ['N', 'O', 'H']
    nonpolar_bb_atoms = ['C', 'CA', 'CB']
    polar_sc_atoms = ['NH1','NH2','H07','H08','H09','H13','H01','H05','H06','OH', 'NE','OD1','OD2','ND2','SG','OE1','OE2','NE2','ND1','NZ','SD','OG','NE1']

    structure = parser.get_structure('pdbf', pdb_name_acd)
    model = structure[0]
    for chain in model:
        for residue in chain:
            resnum = residue.get_id()[1]
            if resnum == position:
                position_identity = residue.get_resname()
                for atom in residue:
                    if str(atom) in  ['<Atom ' + i + '>' for i in polar_bb_atoms]:
                        atm_vect = np.array(list(atom.get_vector()))
                        acd_postion_polar_bb_vectors.append(atm_vect)
                    elif str(atom) in  ['<Atom ' + i + '>' for i in nonpolar_bb_atoms]:
                        atm_vect = np.array(list(atom.get_vector()))
                        acd_postion_nonpolar_bb_vectors.append(atm_vect)
                    elif str(atom) in  ['<Atom ' + i + '>' for i in polar_sc_atoms]:
                        atm_vect = np.array(list(atom.get_vector()))
                        acd_postion_polar_sc_vectors.append(atm_vect)
                    else:
                        if 'H' not in str(atom):
                            atm_vect = np.array(list(atom.get_vector()))
                            acd_postion_nonpolar_sc_vectors.append(atm_vect)
                        else:
                            pass    
            else:
                for atom in residue:
                    if str(atom) in  ['<Atom ' + i + '>' for i in polar_bb_atoms]:
                        atm_vect = np.array(list(atom.get_vector()))
                        acd_rest_polar_bb_vectors.append(atm_vect)
                    elif str(atom) in  ['<Atom ' + i + '>' for i in nonpolar_bb_atoms]:
                        atm_vect = np.array(list(atom.get_vector()))
                        acd_rest_nonpolar_bb_vectors.append(atm_vect)
                    elif str(atom) in  ['<Atom ' + i + '>' for i in polar_sc_atoms]:
                        atm_vect = np.array(list(atom.get_vector()))
                        acd_rest_polar_sc_vectors.append(atm_vect)
                    else:
                        if 'H' not in str(atom):
                            atm_vect = np.array(list(atom.get_vector()))
                            acd_rest_nonpolar_sc_vectors.append(atm_vect)
                        else:
                            pass    

    acd_polar_bb_to_polar_sc_contacts = 0       
    acd_nonpolar_bb_to_nonpolar_sc_contacts = 0 

    acd_polar_bb_to_polar_bb_rest_contacts = 0
    acd_polar_bb_to_polar_sc_rest_contacts = 0
    acd_polar_sc_to_polar_sc_rest_contacts = 0

    acd_nonpolar_bb_to_nonpolar_bb_rest_contacts = 0
    acd_nonpolar_bb_to_nonpolar_sc_rest_contacts = 0
    acd_nonpolar_sc_to_nonpolar_sc_rest_contacts = 0

    for idx, vect1 in enumerate(acd_postion_polar_bb_vectors):
        for jdx, vect2 in enumerate(acd_postion_polar_sc_vectors):
            if idx < jdx:
                dist = LA.norm(vect1 - vect2)
                if dist < 4.0:
                    acd_polar_bb_to_polar_sc_contacts += 1
                else:
                    pass    
            else:
                pass

    for idx, vect1 in enumerate(acd_postion_nonpolar_bb_vectors):
        for jdx, vect2 in enumerate(acd_postion_nonpolar_sc_vectors):
            if idx < jdx:
                dist = LA.norm(vect1 - vect2)
                if dist < 4.0:
                    acd_nonpolar_bb_to_nonpolar_sc_contacts += 1
                else:
                    pass    
            else:
                pass

    for idx, vect1 in enumerate(acd_postion_polar_bb_vectors):
        for jdx, vect2 in enumerate(acd_rest_polar_bb_vectors):
            if idx < jdx:
                dist = LA.norm(vect1 - vect2)
                if dist < 4.0:
                    acd_polar_bb_to_polar_bb_rest_contacts += 1
                else:
                    pass    
            else:
                pass

    for idx, vect1 in enumerate(acd_postion_polar_bb_vectors):
        for jdx, vect2 in enumerate(acd_rest_polar_sc_vectors):
            if idx < jdx:
                dist = LA.norm(vect1 - vect2)
                if dist < 4.0:
                    acd_polar_bb_to_polar_sc_rest_contacts += 1
                else:
                    pass    
            else:
                pass

    for idx, vect1 in enumerate(acd_postion_polar_sc_vectors):
        for jdx, vect2 in enumerate(acd_rest_polar_sc_vectors):
            if idx < jdx:
                dist = LA.norm(vect1 - vect2)
                if dist < 4.0:
                    acd_polar_sc_to_polar_sc_rest_contacts += 1
                else:
                    pass    
            else:
                pass

    for idx, vect1 in enumerate(acd_postion_nonpolar_bb_vectors):
        for jdx, vect2 in enumerate(acd_rest_nonpolar_bb_vectors):
            if idx < jdx:
                dist = LA.norm(vect1 - vect2)
                if dist < 4.0:
                    acd_nonpolar_bb_to_nonpolar_bb_rest_contacts += 1
                else:
                    pass    
            else:
                pass

    for idx, vect1 in enumerate(acd_postion_nonpolar_bb_vectors):
        for jdx, vect2 in enumerate(acd_rest_nonpolar_sc_vectors):
            if idx < jdx:
                dist = LA.norm(vect1 - vect2)
                if dist < 4.0:
                    acd_nonpolar_bb_to_nonpolar_sc_rest_contacts += 1
                else:
                    pass    
            else:
                pass

    for idx, vect1 in enumerate(acd_postion_nonpolar_sc_vectors):
        for jdx, vect2 in enumerate(acd_rest_nonpolar_sc_vectors):
            if idx < jdx:
                dist = LA.norm(vect1 - vect2)
                if dist < 4.0:
                    acd_nonpolar_sc_to_nonpolar_sc_rest_contacts += 1
                else:
                    pass    
            else:
                pass

    acd_final_nonpolar_contacts = acd_nonpolar_bb_to_nonpolar_sc_contacts + acd_nonpolar_bb_to_nonpolar_bb_rest_contacts + acd_nonpolar_bb_to_nonpolar_sc_rest_contacts + acd_nonpolar_sc_to_nonpolar_sc_rest_contacts

    acd_final_polar_contacts = acd_polar_bb_to_polar_sc_contacts + acd_polar_bb_to_polar_bb_rest_contacts + acd_polar_bb_to_polar_sc_rest_contacts + acd_polar_sc_to_polar_sc_rest_contacts

    acd_final_combined_contacts = acd_final_nonpolar_contacts + acd_final_polar_contacts

    feature_names = ['name', 'np_bb_sc_intra', 'p_bb_sc_intra', 'np_bb_bb_inter', 'p_bb_bb_inter', 'np_bb_sc_inter', 'p_bb_sc_inter', 'np_sc_sc_inter', 'p_sc_sc_inter', 'np_total', 'p_total', 'total_contacts']

    features = [(pdb_name_acd.split('_')[0] + '_' + pdb_name_acd.split('_')[1]) + '_' + str(index_num), acd_nonpolar_bb_to_nonpolar_sc_contacts, acd_polar_bb_to_polar_sc_contacts, acd_nonpolar_bb_to_nonpolar_bb_rest_contacts, acd_polar_bb_to_polar_bb_rest_contacts, acd_nonpolar_bb_to_nonpolar_sc_rest_contacts, acd_polar_bb_to_polar_sc_rest_contacts, acd_nonpolar_sc_to_nonpolar_sc_rest_contacts, acd_polar_sc_to_polar_sc_rest_contacts, acd_final_nonpolar_contacts, acd_final_polar_contacts, acd_final_combined_contacts]

    d = pd.DataFrame(0, index=np.arange(1), columns=feature_names)
    d.loc[0] = features
    d.set_index('name', inplace=True)   

    return d

if __name__ == "__main__":

    scores = {}

    acd_pdbs = sorted(glob.glob('*mutated_*.pdb'))
    wt_pdbs = sorted(glob.glob('*WT_*.pdb'))

    dfs = []

    for indexx, acd_pdb in enumerate(acd_pdbs):
        acd_df = ComputeESFTerms(acd_pdb, indexx + 1)
        wt_df = ComputeESFTerms(wt_pdbs[indexx], indexx + 1)
        df_sub = acd_df.sub(wt_df)

        df_mat = ComputeBioinformaticsMatrices(wt_pdbs[indexx], indexx + 1)

        df_final = pd.concat([df_sub, df_mat], axis=1)
        dfs.append(df_final)

    out_df = pd.concat(dfs)

    out_df.to_csv('Acd_ESF_Terms_WithoutConservation.csv')