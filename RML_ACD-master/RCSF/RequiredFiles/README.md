Make sure to have these two files in the folder where you a performing your simulation. The final line of the params file specifices the location to the rotamer library. If you want to store the rotamer library file somewhere elese permanently, update the path at the bottom of the params file to match the location.

Pyrosetta should be initialized in the following way:
from pyrosetta import *
init(extra_options='-extra_res_fa acd.params')
