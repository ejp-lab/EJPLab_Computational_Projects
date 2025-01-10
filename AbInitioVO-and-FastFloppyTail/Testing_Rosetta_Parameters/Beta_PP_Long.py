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
maplow = MoveMap()
maplow.set_bb(True)
maplow.set_chi(True)

# The Score Functions

sf0 = create_score_function('score0')

sf1 = create_score_function('score1')

sf2 = create_score_function('score2')

sf3 = create_score_function('score3')

sf3hb = create_score_function('score3')
sf3hb.set_weight(hbond_lr_bb, 1.0)
sf3hb.set_weight(hbond_sr_bb, 1.0)

sfbeta = create_score_function('beta')

sfrg = ScoreFunction()
sfrg.set_weight(rg, 1.0)

# The Movers

## Minimization Movers
minvdw = MinMover()
minvdw.movemap(maplow)
minvdw.score_function(sf0) 
minvdw.min_type('linmin')

min1 = MinMover()
min1.movemap(maplow)
min1.score_function(sfbeta) 
min1.min_type('linmin')
min1.tolerance(0.001)

min2 = MinMover()
min2.movemap(maplow)
min2.score_function(sfbeta) 
min2.min_type('linmin')
min2.tolerance(0.001)

min3 = MinMover()
min3.movemap(maplow)
min3.score_function(sfbeta) 
min3.min_type('linmin')
min3.tolerance(0.001)

min3hb = MinMover()
min3hb.movemap(maplow)
min3hb.score_function(sfbeta)
min3hb.min_type('linmin')
min3hb.tolerance(0.001)

minbeta = MinMover()
minbeta.movemap(maplow)
minbeta.score_function(sfbeta)
minbeta.min_type('linmin')
minbeta.tolerance(0.001)

## Phi-Psi Movers
smMover20T = SmallMover(maplow, 2.0, 5)
shMover20T = ShearMover(maplow, 2.0, 5)
smMover20T.angle_max(20)
smMover20T.angle_max("H", 20)
smMover20T.angle_max("E", 20)
smMover20T.angle_max("L", 20)
shMover20T.angle_max(20)
shMover20T.angle_max("H", 20)
shMover20T.angle_max("E", 20)
shMover20T.angle_max("L", 20)
smMover20T.set_preserve_detailed_balance(True)
shMover20T.set_preserve_detailed_balance(True)
randMover20T = RandomMover()
randMover20T.add_mover(smMover20T)
randMover20T.add_mover(shMover20T)
repMover20T = RepeatMover(randMover20T, 7)

smMover10T = SmallMover(maplow, 2.0, 5)
shMover10T = ShearMover(maplow, 2.0, 5)
smMover10T.angle_max(10)
smMover10T.angle_max("H", 10)
smMover10T.angle_max("E", 10)
smMover10T.angle_max("L", 10)
shMover10T.angle_max(10)
shMover10T.angle_max("H", 10)
shMover10T.angle_max("E", 10)
shMover10T.angle_max("L", 10)
smMover10T.set_preserve_detailed_balance(True)
shMover10T.set_preserve_detailed_balance(True)
randMover10T = RandomMover()
randMover10T.add_mover(smMover10T)
randMover10T.add_mover(shMover10T)
repMover10T = RepeatMover(randMover10T, 7)

smMover5T = SmallMover(maplow, 2.0, 5)
shMover5T = ShearMover(maplow, 2.0, 5)
smMover5T.angle_max(5)
smMover5T.angle_max("H", 5)
smMover5T.angle_max("E", 5)
smMover5T.angle_max("L", 5)
shMover5T.angle_max(5)
shMover5T.angle_max("H", 5)
shMover5T.angle_max("E", 5)
shMover5T.angle_max("L", 5)
smMover5T.set_preserve_detailed_balance(True)
shMover5T.set_preserve_detailed_balance(True)
randMover5T = RandomMover()
randMover5T.add_mover(smMover5T)
randMover5T.add_mover(shMover5T)
repMover5T = RepeatMover(randMover5T, 7)

smMover5 = SmallMover(maplow, 2.0, 5)
shMover5 = ShearMover(maplow, 2.0, 5)
smMover5.angle_max(5)
smMover5.angle_max("H", 5)
smMover5.angle_max("E", 5)
smMover5.angle_max("L", 5)
shMover5.angle_max(5)
shMover5.angle_max("H", 5)
shMover5.angle_max("E", 5)
shMover5.angle_max("L", 5)
randMover5 = RandomMover()
randMover5.add_mover(smMover5)
randMover5.add_mover(shMover5)
repMover5 = RepeatMover(randMover5, 7)

smMover3 = SmallMover(maplow, 2.0, 3)
shMover3 = ShearMover(maplow, 5.0, 3)
smMover3.angle_max(5)
smMover3.angle_max("H", 5)
smMover3.angle_max("E", 5)
smMover3.angle_max("L", 5)
shMover3.angle_max(5)
shMover3.angle_max("H", 5)
shMover3.angle_max("E", 5)
shMover3.angle_max("L", 5)
randMover3 = RandomMover()
randMover3.add_mover(smMover3)
randMover3.add_mover(shMover3)
repMover3 = RepeatMover(randMover3, 7)

smMover1 = SmallMover(maplow, 2.0, 1)
shMover1 = ShearMover(maplow, 2.0, 1)
smMover1.angle_max(5)
smMover1.angle_max("H", 5)
smMover1.angle_max("E", 5)
smMover1.angle_max("L", 5)
shMover1.angle_max(5)
shMover1.angle_max("H", 5)
shMover1.angle_max("E", 5)
shMover1.angle_max("L", 5)
randMover1 = RandomMover()
randMover1.add_mover(smMover1)
randMover1.add_mover(shMover1)
repMover1 = RepeatMover(randMover1, 7)

## Sequence Movers
movervdw = SequenceMover()
movervdw.add_mover(repMover20T)
movervdw.add_mover(repMover20T)
movervdw.add_mover(minvdw)

movervdwf = SequenceMover()
movervdwf.add_mover(repMover20T)
movervdwf.add_mover(repMover20T)
movervdwf.add_mover(minvdw)

mover1 = SequenceMover()
mover1.add_mover(repMover10T)
mover1.add_mover(repMover10T)
mover1.add_mover(min1)

mover2 = SequenceMover()
mover2.add_mover(repMover5T)
mover2.add_mover(repMover5T)
mover2.add_mover(min2)

mover3 = SequenceMover()
mover3.add_mover(repMover5)
mover3.add_mover(repMover5)
mover3.add_mover(min3)

mover4 = SequenceMover()
mover4.add_mover(repMover3)
mover4.add_mover(repMover3)
mover4.add_mover(min3hb)

mover5 = SequenceMover()
mover5.add_mover(repMover3)
mover5.add_mover(minbeta)

mover6 = SequenceMover()
mover6.add_mover(repMover1)
mover6.add_mover(minbeta)

# Converting the Pose
switch = SwitchResidueTypeSetMover('fa_standard')

	
# The Monte Carlo
## The Variables
my_kT = 10.0
my_MCtrials = 1000

## Score Evaluation with Hastings-Metropolis
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

## Score Evaluation with Hastings-Metropolis with Constraints and Sidechain Sampling
def mcb(p, fn, scorefxn, kT, trials):
	switch.apply(p)
	task = standard_packer_task(p)
	task.restrict_to_repacking()
	pack_mover = PackRotamersMover(sfbeta, task)
	pack_mover.apply(p)
	old = Pose()
	old.assign(p)
	old_score = scorefxn(old)
	p_low = Pose()
	p_low.assign(p)
	p_low_score = scorefxn(p_low)
	for i in range(trials):
		fn.apply(p)
		task = standard_packer_task(p)
		task.restrict_to_repacking()
		pack_mover = PackRotamersMover(sfbeta, task)
		pack_mover.apply(p)
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
for i in range(34):
	p.assign(starting_p)
	p = mc(p, movervdw, sf0, 10.0, 1000)
	pvdw.assign(p)
	p = mc(p, movervdwf, sf0, 10.0, 1000)
	pvdwc.assign(p)
	pcen.assign(p)
	switch.apply(p)
	p = mc(p, mover1, sfbeta, 10.0, 3000)
	p = mc(p, mover2, sfbeta, 3.0, 5000)
	p = mc(p, mover3, sfbeta, 3.0, 5000)
	p = mc(p, mover4, sfbeta, 1.0, 5000)
	p = mc(p, mover5, sfbeta, 1.0, 5000)
	p = mc(p, mover6, sfbeta, 1.0, 10000)
	outf = file("mcrelax_Method3_PhiPsi.sc", 'a')
	pdb_out = "relax_Method3_PhiPsi_out_%i.pdb" %i
	pdb_out_vdw = "relax_Method3_PhiPsi_VDW_%i.pdb" %i
	pdb_out_vdwc = "relax_Method3_PhiPsi_VDW_C_%i.pdb" %i
	pdb_out_cen = "relax_Method3_PhiPsi_Cen_%i.pdb" %i
	outf.write("%s\t%.3f\t%.4f\t%.4f\t%.4f\t%.3f\n" % (pdb_out,  sfbeta(p), CA_rmsd(pvdw,p), CA_rmsd(pvdwc,p), CA_rmsd(pcen,p), sfrg(p)))
	p.dump_pdb(pdb_out)
	pvdw.dump_pdb(pdb_out_vdw)
	pvdwc.dump_pdb(pdb_out_vdwc)
	pcen.dump_pdb(pdb_out_cen)
	outf.close()