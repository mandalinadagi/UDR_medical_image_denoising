#!/bin/bash
# 
# CompecTA (c) 2017
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#
# TODO:
#   - Set name of the job below changing "Test" value.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter.
#   - Select the partition (queue) you want to run the job in:
#     - short : For jobs that have maximum run time of 120 mins. Has higher priority.
#     - long  : For jobs that have maximum run time of 7 days. Lower priority than short.
#     - longer: For testing purposes, queue has 31 days limit but only 3 nodes.
#   - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input and output file names below.
#   - If you do not want mail please remove the line that has --mail-type
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch examle_submit.sh

# -= Resources =-
#
#SBATCH --job-name=01tv+denoiser
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
##SBATCH --exclude=da[01-04],ag01,sm01,ai07,ai10,ai11,be[01-12],ai05,ai02,ai06,ai04
#SBATCH --constraint=tesla_k80
##SBATCH --nodelist=ai16
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --time=7-0
#SBATCH --output=sonuc-%j.out
#SBATCH --error=sonuc-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ckorkmaz14@ku.edu.tr

################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################

echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

echo "==============================================================================="
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/kuacc/users/ckorkmaz14/miniconda3/lib/
source /kuacc/users/ckorkmaz14/py_venvs/cansu/bin/activate
nvidia-smi
python train.py
python test.py
