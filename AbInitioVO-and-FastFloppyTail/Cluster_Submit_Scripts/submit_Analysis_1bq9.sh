#!/bin/bash
#$ -q all.q
#$ -cwd
#$ -N An1bq9
#$ -S /bin/bash
#$ -o An1bq9.log
#$ -e An1bq9.error
#$ -l h_rt=900:00:00

# Enable Additional Software
. /etc/profile.d/modules.sh
module unload cmsub
module load shared python/anaconda/2.4.1
source activate lion
    
# Execute Script
cd AbInitioVO
cd Ab1bq9/
python ../AbInitio_Analysis.py O 1 54 ../1bq9_STD.pdb Ab1bq9_*/
