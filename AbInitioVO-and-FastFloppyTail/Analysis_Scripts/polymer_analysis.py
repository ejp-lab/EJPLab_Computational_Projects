#!/usr/bin/python

## polymer_analysis.py
## Generates distance matrix [.dm], contact matrix (w/ specified cutoff)[.cm], rg values (in Angstroms)[.rg], and polymer scaling (r vs. N)[.nu] 
## Takes one or more PDBs from trjconv or pdb_assembler.py as command-line arguments
## syntax: ./polymer_analysis.py trajectory1.pdb trajectory2.pdb assembled_ensemble3.pdb

from scipy import *
import sys
import scipy.spatial.distance as d

cutoff = float(25) # Cutoff in Angstroms for whether a contact exists

# Returns 2-D array of XYZ coordinates from a PDB line
def xyz(s): return array([float(f) for f in (s[30:38], s[38:46], s[46:54])])

for f in sys.argv[1:]:
    print(f)
    models_pieces = filter(lambda s: ' CA ' in s, open(f).readlines()) # Only reads CA positions
    models = []
    for models_piece in models_pieces:
        models.append(models_piece)
    nres = int(models[-1].split()[5]) # Number of residues in the protein, assumed to be the resn of the last CA in the file
    dms = []
    contacts = []
    f_rg = open(f[:-4] + '.rg', 'w') # File with radius of gyration for each model
    print(len(models))
    for i in range(0, len(models), nres):
        coords = array([xyz(l) for l in models[i:i+nres]]) # Coordinates of CA positions
        dm = d.squareform(d.pdist(coords)) # Distance matrix using scipy.pdist
        contact = where(dm < cutoff, ones_like(dm), zeros_like(dm)) # An array with 1 if distance < cutoff, 0 otherwise
        dms.append(dm) # List of distance matrices from all structures
        contacts.append(contact) # List of contact matrices from all structures   
        rg = sqrt(sum(dm**2)/(2*nres**2)) # Calculates radius of gyration
        f_rg.write("%i\t%.3f\n" % (i/nres + 1, rg))
    f_rg.close()
    ave_dms = average(array(dms), axis=0) # Mean distance matrix from all structures
    std_dms = std(array(dms), axis=0) # Std distance matrix from all structures
    ave_contacts = average(array(contacts), axis=0) # Mean contact matrix from all structures
    if d.is_valid_y(ave_dms): ave_dms = d.squareform(ave_dms)
    if d.is_valid_y(std_dms): std_dms = d.squareform(ave_dms)
    if d.is_valid_y(ave_contacts): ave_contacts = d.squareform(ave_contacts)
    savetxt(f[:-4] + '.dm', ave_dms, fmt="%.3f")
    savetxt(f[:-4] + '.std', std_dms, fmt="%.3f")
    savetxt(f[:-4] + '.cm', ave_contacts, fmt="%.3f")
    m,n = ave_dms.shape
    scalingf = open(f[:-4] + '.nu', 'w') # File with polymer scaling (r vs. N)
    scalingfs = open(f[:-4] + '.nus', 'w') # File with std of polymer scaling (r vs. N)
    for i in range(m):
        dm_diag = diagonal(ave_dms, i) # Diagonals of distance matrix
        scalingf.write("%s\t%.3f\n" % (i, average(dm_diag))) # Averaged to get mean r
        dm_diags = diagonal(std_dms, i) # Diagonals of distance matrix
        scalingfs.write("%s\t%.3f\n" % (i, average(dm_diags))) # Averaged to get mean r        
    scalingf.close()
    scalingfs.close()
    