import numpy as np
from numpy import genfromtxt
from numpy import average
import sys

#Exp_PRE_20 = genfromtxt('../../../PRE_Data_Compiled/Exp_asyn_PRE_20.csv', delimiter=',')
#Exp_PRE_24 = genfromtxt('../../../PRE_Data_Compiled/Exp_asyn_PRE_24.csv', delimiter=',')
#Exp_PRE_42 = genfromtxt('../../../PRE_Data_Compiled/Exp_asyn_PRE_42.csv', delimiter=',')
#Exp_PRE_62 = genfromtxt('../../../PRE_Data_Compiled/Exp_asyn_PRE_62.csv', delimiter=',')
#Exp_PRE_85 = genfromtxt('../../../PRE_Data_Compiled/Exp_asyn_PRE_85.csv', delimiter=',')
#Exp_PRE_87 = genfromtxt('../../../PRE_Data_Compiled/Exp_asyn_PRE_87.csv', delimiter=',')
#Exp_PRE_103 = genfromtxt('../../../PRE_Data_Compiled/Exp_asyn_PRE_103.csv', delimiter=',')
#Exp_PRE_120 = genfromtxt('../../../PRE_Data_Compiled/Exp_asyn_PRE_120.csv', delimiter=',')

#Exp_PRE = [Exp_PRE_20, Exp_PRE_24, Exp_PRE_42, Exp_PRE_62, Exp_PRE_85, Exp_PRE_87, Exp_PRE_103, Exp_PRE_120]

Exp_PRE_list = genfromtxt(sys.argv[1], dtype=str)
Exp_PRE = []
for pre_exp in Exp_PRE_list:
	Exp_PRE.append(genfromtxt(pre_exp, delimiter=','))

Calc_PRE_all = genfromtxt('Calculated_PREs.txt', delimiter=' ')
Calc_PRE_20 = []
Calc_PRE_24 = []
Calc_PRE_42 = []
Calc_PRE_62 = []
Calc_PRE_85 = []
Calc_PRE_87 = []
Calc_PRE_103 = []
Calc_PRE_120 = []

Calc_PRE = [Calc_PRE_20, Calc_PRE_24, Calc_PRE_42, Calc_PRE_62, Calc_PRE_85, Calc_PRE_87, Calc_PRE_103, Calc_PRE_120]

for arr_idx,arr_item in enumerate(Calc_PRE):
	col_num = arr_idx*2
	for line in range(0,len(Calc_PRE_all)):
		arr_item.append(Calc_PRE_all[line][col_num])
		
Res_RMSD = []
Frac_Diff = []

for i in range(0, len(Exp_PRE)):
	Spec_Exp_PRE = Exp_PRE[i]
	Spec_Calc_PRE = Calc_PRE[i]
	exp_vals = []
	calc_vals = []
	print(i)
	for exp_val_idx, exp_val_item in enumerate(Spec_Exp_PRE):
		exp_vals.append(exp_val_item[1])
		spec_calc_val = Spec_Calc_PRE[int(exp_val_item[0]) - 1]
		if spec_calc_val > 1.0:
			spec_calc_val = 1.0
		if spec_calc_val < 0.0:
			spec_calc_val = 0.0
		calc_vals.append(spec_calc_val)	
		sd_val = (float(exp_val_item[1]) - float(spec_calc_val))**2
		Res_RMSD.append(sd_val)
		
Avg_Res_MSD = average(Res_RMSD)
Avg_Res_RMSD = np.sqrt(Avg_Res_MSD)
print(Avg_Res_RMSD)
Avg_Res_RMSD_holder = [Avg_Res_RMSD]
np.savetxt('PRE_RMSD_Output.txt', Avg_Res_RMSD_holder, fmt='%s', delimiter=' ', newline='\n')
#print Avg_Res_RMSD
