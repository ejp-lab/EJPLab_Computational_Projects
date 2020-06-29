#!/bin/bash
#$ -q all.q
#$ -cwd
#$ -N Ab1bq9
#$ -S /bin/bash
#$ -o Ab1bq9.log
#$ -e Ab1bq9.error
#$ -l h_rt=900:00:00
#$ -t 1-25:1

# Enable Additional Software
. /etc/profile.d/modules.sh
module unload cmsub
module load shared python/anaconda/2.4.1
source activate lion

# Setup Output dir
cd AbInitioVO
targetdir="Ab1bq9/${JOB_NAME}_${SGE_TASK_ID}"
mkdir -p $targetdir
cp AbInitioVO.py $targetdir
chmod +x $targetdir/*
cp 1bq9.fasta.txt $targetdir
cp 1bq9.diso $targetdir
cp 1bq9_dcor_frags.200.9mers $targetdir
cp 1bq9_dcor_frags.200.3mers $targetdir
cp 1bq9_frags.200.9mers $targetdir
cp 1bq9_frags.200.3mers $targetdir
cd $targetdir
    
# Execute Script
python AbInitioVO.py -in 1bq9.fasta.txt -abinitiovo -abnstruct 5 -diso 1bq9.diso -t_frag 1bq9_dcor_frags.200.3mers -n_frag 1bq9_dcor_frags.200.9mers > /dev/null
python AbInitioVO.py -in 1bq9.fasta.txt -abnstruct 5 -diso 1bq9.diso -t_frag 1bq9_frags.200.3mers -n_frag 1bq9_frags.200.9mers > /dev/null
