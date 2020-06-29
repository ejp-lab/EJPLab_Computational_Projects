#!/bin/bash
#$ -q all.q
#$ -cwd
#$ -N FPasyn
#$ -S /bin/bash
#$ -o FPasyn.log
#$ -e FPasyn.error
#$ -l h_rt=900:00:00
#$ -t 1

# Enable Additional Software
. /etc/profile.d/modules.sh
module unload cmsub
module load shared rosetta/3.9
 
# Execute Script
cd FloppyTail
/cm/shared/apps/rosetta/rosetta3.9/main/source/bin/fragment_picker.static.linuxgccrelease @asyn_FT.flags