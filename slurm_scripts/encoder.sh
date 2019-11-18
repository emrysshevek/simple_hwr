#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -C 'rhel7&pascal'
#SBATCH --mem 10666
#SBATCH --ntasks 6
#SBATCH --time 72:00:00
#SBATCH --mail-user=masonfp@byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#%Module

module purge
module load cuda/10.1
module load cudnn/7.6

export PATH="/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/env/hwr4_env:$PATH"
cd "/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/simple_hwr"
python -u train.py --config encoder.yaml