#!/bin/bash
#$ -q all.q
#$ -cwd
#$ -N FFTasyn
#$ -S /bin/bash
#$ -o FFTasyn.log
#$ -e FFTasyn.error
#$ -l h_rt=900:00:00
#$ -t 1-25:1

# Enable Additional Software
. /etc/profile.d/modules.sh
module unload cmsub
module load shared python/anaconda/2.4.1
source activate lion

# Setup Output dir
cd FloppyTail
targetdir="FFTasyn/${JOB_NAME}_${SGE_TASK_ID}"
mkdir -p $targetdir
cp FastFloppyTail.py $targetdir
chmod +x $targetdir/*
cp asyn.fasta.txt $targetdir
cp asyn_dcor_frags.200.3mers $targetdir
cd $targetdir
    
# Execute Script
python FastFloppyTail.py -in asyn.fasta.txt -ftnstruct 500 -t_frag asyn_dcor_frags.200.3mers > /dev/null