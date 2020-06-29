#!/bin/bash
#$ -q all.q
#$ -cwd
#$ -N FDasyn
#$ -S /bin/bash
#$ -o FDasyn.log
#$ -e FDasyn.error
#$ -l h_rt=900:00:00
#$ -t 1-25:1

# Enable Additional Software
. /etc/profile.d/modules.sh
module unload cmsub
module load shared python/anaconda/2.4.1
source activate lion
    
# Execute Script
cd FloppyTail/OtherProteinsOldFragments
targetdir="FFTasyn"
targetanalysisscript="Full_IDP_Analysis_Script_${SGE_TASK_ID}.py"
targetpalesscript="Process_PALES_Parallel_${SGE_TASK_ID}.py"
targetspartascript="Process_SPARTA_Parallel_${SGE_TASK_ID}.py"
sed -e 's/FOLDERNUMBER/'"${SGE_TASK_ID}"'/g' < /home/gianna1/FloppyTail/Analysis_Packages/Full_IDP_Analysis_Script_FastFloppyTail.py > $targetdir/$targetanalysisscript
sed -e 's/FOLDERNUMBER/'"${SGE_TASK_ID}"'/g' < /home/gianna1/FloppyTail/Analysis_Packages/Process_PALES_Parallel_bestFit.py > $targetdir/$targetpalesscript
sed -e 's/FOLDERNUMBER/'"${SGE_TASK_ID}"'/g' < /home/gianna1/FloppyTail/Analysis_Packages/Process_SPARTA_Parallel.py > $targetdir/$targetspartascript
chmod +x $targetdir
cd $targetdir
python $targetanalysisscript
