#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -C 'rhel7&pascal'
#SBATCH --mem 10666
#SBATCH --ntasks 6
#SBATCH --output="/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/taylor_simple_hwr/slurm_scripts/scripts/cnn_architecture/log_resnet101.slurm"
#SBATCH --time 36:00:00
#SBATCH --mail-user=taylornarchibald@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#%Module

module purge
module load cuda/10.1
module load cudnn/7.6

export PATH="/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/env/hwr4_env:$PATH"
cd "/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/taylor_simple_hwr"
which python
python -u train.py --config '/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/taylor_simple_hwr/configs/cnn_architecture/resnet101.yaml'
