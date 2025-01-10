#FORMAT OF ORIGINAL FLOPPY TAIL [KLEIGER, G ET. AL. "RAPID E2-E3 ASSEMBLY AND DISASSEMBLY ENABLE PROCESSICE UBIQUITYLATION OF CULLIN-RING UBIQUITIN LIGASE SUBSTRATES" CELL. 139(5): 957-968 2009]
##CENTROID MODE [NOT DISCLOSED POTENTIALLY RAMA ONLY]
###19 CYCLES: RANDOM: SMALL_180(40%), SHEAR_180(40%), FRAGMENT_3MER(20%)
###20TH CYCLE: MINIMIZATION
####5000 TOTAL CYCLES -> RECOVER LOW
##FULL-ATOM MODE [SCORE12]
###START BY REPACKING ALL IN TAIL/VICINITY OF TAIL FOLLOWED BY MINIMIZATION
###14 CYCLES: RANDOM: SMALL_4(50%), SHEAR_4(50%) FOLLOWED BY SINGLE ROTAMER_TRIALS
###15TH CYCLE:MINIMIZATION
###30TH CYCLE:REPACK THEN MINIMIZATION
####3000 CYCLES -> RECOVER LOW

# The Start-Up
from rosetta import *  #Comment
init()
from math import exp, log, pi, sqrt
from random import random as rnd

# The Poses
p=Pose()
make_pose_from_sequence(p, "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA", "centroid")
starting_p = Pose()
starting_p.assign(p)
pvdw = Pose()
pvdwc = Pose()
pcen = Pose()
cenmap = MoveMap()
cenmap.set_bb(True)
fullmap = MoveMap()
fullmap.set_bb(True)
fullmap.set_chi(True)

# The Score Functions
sf_vdw = create_score_function('score0')

sf_cen_std = create_score_function('cen_std')
sf_cen_std.set_weight(rama, 1.0)
sf_cen_std.set_weight(cenpack, 1.0)
sf_cen_std.set_weight(hbond_lr_bb, 1.0)
sf_cen_std.set_weight(hbond_sr_bb, 1.0)

sf_score12 = create_score_function('beta')

sfrg = ScoreFunction()
sfrg.set_weight(rg, 1.0)

# The Movers

## Fragment Movers

# Importing the Fragment Files
fragset3 = ConstantLengthFragSet(3)
fragset3.read_fragment_file("as2_frag3.200_v1_3")

# Constructing the Fragment Mover
fragmover3 = ClassicFragmentMover(fragset3, cenmap)

## Minimization Movers
vdwmin = MinMover()
vdwmin.movemap(cenmap)
vdwmin.score_function(sf_vdw)
vdwmin.min_type('linmin')

cenmin = MinMover()
cenmin.movemap(cenmap)
cenmin.score_function(sf_cen_std)
cenmin.min_type('linmin')

fullmin = MinMover()
fullmin.movemap(fullmap)
fullmin.score_function(sf_score12)
fullmin.min_type('linmin')

## Phi-Psi Movers
vdw_small_mover = SmallMover(cenmap, 1.0, 1)
vdw_shear_mover = ShearMover(cenmap, 1.0, 1)
vdw_small_mover.angle_max(180)
vdw_small_mover.angle_max("H", 180)
vdw_small_mover.angle_max("E", 180)
vdw_small_mover.angle_max("L", 180)
vdw_shear_mover.angle_max(180)
vdw_shear_mover.angle_max("H", 180)
vdw_shear_mover.angle_max("E", 180)
vdw_shear_mover.angle_max("L", 180)
vdwrandom = RandomMover()
vdwrandom.add_mover(vdw_small_mover)
vdwrandom.add_mover(vdw_shear_mover)
vdwrepeat = RepeatMover(vdwrandom, 7)

cen_small_mover = SmallMover(cenmap, 0.8, 1)
cen_shear_mover = ShearMover(cenmap, 0.8, 1)
cen_small_mover.angle_max(180)
cen_small_mover.angle_max("H", 180)
cen_small_mover.angle_max("E", 180)
cen_small_mover.angle_max("L", 180)
cen_shear_mover.angle_max(180)
cen_shear_mover.angle_max("H", 180)
cen_shear_mover.angle_max("E", 180)
cen_shear_mover.angle_max("L", 180)

cenrandom = RandomMover()
cenrandom.add_mover(cen_small_mover)
cenrandom.add_mover(cen_small_mover)
cenrandom.add_mover(cen_small_mover)
cenrandom.add_mover(cen_small_mover)
cenrandom.add_mover(cen_shear_mover)
cenrandom.add_mover(cen_shear_mover)
cenrandom.add_mover(cen_shear_mover)
cenrandom.add_mover(cen_shear_mover)
cenrandom.add_mover(fragmover3)
cenrandom.add_mover(fragmover3)

full_small_mover = SmallMover(fullmap, 0.8, 1)
full_shear_mover = ShearMover(fullmap, 0.8, 1)
full_small_mover.angle_max(4)
full_small_mover.angle_max("H", 4)
full_small_mover.angle_max("E", 4)
full_small_mover.angle_max("L", 4)
full_shear_mover.angle_max(4)
full_shear_mover.angle_max("H", 4)
full_shear_mover.angle_max("E", 4)
full_shear_mover.angle_max("L", 4)
full_random = RandomMover()
full_random.add_mover(full_small_mover)
full_random.add_mover(full_shear_mover) 

## Packing/Rotamer Movers

### The Task Operations
fulltask = standard_packer_task(p)
fulltask.restrict_to_repacking()

### The Rotamer Movers
fullpack = PackRotamersMover(sf_score12, fulltask)

fullrottrial = RotamerTrialsMover(sf_score12, fulltask)

## Sequence Movers
vdwmover = SequenceMover()
vdwmover.add_mover(vdwrepeat)
vdwmover.add_mover(vdwmin)

fullmover = SequenceMover()
fullmover.add_mover(full_random)
fullmover.add_mover(fullrottrial)

## Converting the Pose
switch = SwitchResidueTypeSetMover('fa_standard')

## Running the Monte-Carlo
def mc(p, fn, scorefxn, kT, trials):
	old = Pose()
	old.assign(p)
	old_score = scorefxn(old)
	p_low = Pose()
	p_low.assign(p)
	p_low_score = scorefxn(p_low)
	for i in range(trials):
		fn.apply(p)
		new_score = scorefxn(p)
		Dscore = new_score - old_score
		if Dscore < 0:
			old.assign(p)
			old_score = new_score
			Dscore_low = new_score - p_low_score
			if Dscore_low < 0:
				p_low.assign(p)
				p_low_score = new_score
			else:
				print 'Almost'
		elif exp(-Dscore/kT) > rnd():
			old.assign(p)
			old_score = new_score
		else:
			p.assign(old)
			new_score = old_score
	p.assign(p_low)
	return p

## Running the Monte-Carlo
def mcb(p, fn, scorefxn, kT, trials):
	switch.apply(p)
	fulltask = standard_packer_task(p)
	fulltask.restrict_to_repacking()
	fullpack = PackRotamersMover(sf_score12, fulltask)
	fullrottrial = RotamerTrialsMover(sf_score12, fulltask)
	old = Pose()
	old.assign(p)
	old_score = scorefxn(old)
	p_low = Pose()
	p_low.assign(p)
	p_low_score = scorefxn(p_low)
	for i in range(trials):
		fn.apply(p)
		new_score = scorefxn(p)
		Dscore = new_score - old_score
		if Dscore < 0:
			old.assign(p)
			old_score = new_score
			Dscore_low = new_score - p_low_score
			if Dscore_low < 0:
				p_low.assign(p)
				p_low_score = new_score
			else:
				print 'Almost'
		elif exp(-Dscore/kT) > rnd():
			old.assign(p)
			old_score = new_score
		else:
			p.assign(old)
			new_score = old_score
	p.assign(p_low)
	return p
	
# The Simulation and Output
for i in range(340):
	p.assign(starting_p)
	p = mc(p, vdwmover, sf_vdw, 10.0, 1000)
	pvdw.assign(p)
	for j in range(250):
		p = mc(p, cenrandom, sf_cen_std, 1.0, 19)
		p = mc(p, cenmin, sf_cen_std, 1.0, 1)
	pcen.assign(p)
	switch.apply(p)
	for j in range(100):
		p = mc(p, fullmover, sf_score12, 1.0, 14)
		p = mc(p, fullmin, sf_score12, 1.0, 1)
		p = mc(p, fullmover, sf_score12, 1.0, 14)
		task = standard_packer_task(p)
		task.restrict_to_repacking()
		pack_mover = PackRotamersMover(sf_score12, task)
		pack_mover.apply(p)
		p = mc(p, fullmin, sf_score12, 1.0, 1)
		print '=)'
	outf = file("mcrelax_Method3_PhiPsi.sc", 'a')
	pdb_out = "relax_Method3_PhiPsi_out_%i.pdb" %i
	pdb_out_vdw = "relax_Method3_PhiPsi_VDW_%i.pdb" %i
	pdb_out_cen = "relax_Method3_PhiPsi_Cen_%i.pdb" %i
	outf.write("%s\t%.3f\t%.4f\t%.4f\t%.3f\n" % (pdb_out, sf_score12(p), CA_rmsd(pvdw,p), CA_rmsd(pcen,p), sfrg(p)))
	p.dump_pdb(pdb_out)
	pvdw.dump_pdb(pdb_out_vdw)
	pcen.dump_pdb(pdb_out_cen)
	outf.close()