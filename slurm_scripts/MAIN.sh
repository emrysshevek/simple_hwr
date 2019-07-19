#!/bin/bash
# #SBATCH --export=config=$1  # "warp_online_lr_schedule4"
#SBATCH --gres=gpu:1
#SBATCH -C 'rhel7&pascal'
#SBATCH --mem 16000M
#SBATCH --ntasks 1
#SBATCH --output="./warp_online_lr_schedule4_resume.slurm"
#SBATCH --time 24:00:00
#SBATCH --mail-user=taylor.archibald@byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#%Module

module purge
module load cuda/10.1
module load cudnn/7.6

group_path="/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr"
export PATH="${group_path}/env/hwr4_env/bin:$PATH"
which python

config="warp_online_lr_schedule4_resume"
cd "${group_path}/taylor_simple_hwr"
echo $config
python -u train.py --config $config

