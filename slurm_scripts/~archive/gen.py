# -*- coding: future_fstrings -*-
import os
import sys
from subprocess import Popen

# rm -r SA*.sh && rm -r DCG*.sh

# GLOBALS
email = "taylornarchibald@gmail.com" # mikebbrodie@gmail.com
net_dir='/fslgroup/fslg_buni/compute/puzzler'
script_dir=os.path.join(net_dir, "scripts")

hw_req = {
    "cifar10":{"threads":7, "time":"36:00:00"},
    "celeba":{"threads":7, "time":"36:00:00"},
    "mnist":{"threads":4, "time":"8:00:00"}
    }


def get_sh(path):
    for ds,s,fs in os.walk(path):
        if ds.lower() == "old":
        	continue
        for f in fs:
            if f[-3:] == ".sh":
                yield os.path.join(ds,f)

def mkdir(path):
    if path is not None and len(path) > 0 and not os.path.exists(path):
        os.makedirs(path)

def gen(email, dataset, arch, puzzler_size):
        combined_name = arch.upper() + (f"+P{puzzler_size}" if puzzler_size else "")
        exp_arg = ""
	script_name = "puzzlerP4" if puzzler_size < 20 else "puzzlerP20"

        if puzzler_size:
           exp_arg = f"--puzzler --npieces {puzzler_size}"

        time = hw_req[dataset]["time"]
        threads = hw_req[dataset]["threads"]
        mem = int( 64000 / threads )
	n = 2
	path = os.path.join(net_dir,f"scripts/{combined_name}")
        
        mkdir(path)
        for i in xrange(n):
                time = hw_req[dataset]["time"]
                threads = hw_req[dataset]["threads"]
                with open(f"{path}/{dataset}_{i}.sh","w") as f:
                        f.write(f"""#!/bin/bash
#SBATCH --time={time}
#SBATCH --ntasks={threads}
#SBATCH --nodes=1
#SBATCH --mem-per-cpu={mem}MB
#SBATCH --gres=gpu:1
#SBATCH -J "{combined_name}_{dataset}_{i}"
#SBATCH --output "{path}/{dataset}_{i}.slurm"
#SBATCH --mail-user={email}
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --requeue
#SBATCH -C 'rhel7&pascal'

module load cuda/9.0
module load cudnn/7.0
module load python/2.7

source /fslhome/mbrodie/rhel/bin/activate

net_dir={net_dir}

/fslhome/mbrodie/rhel/bin/python $net_dir/src/{script_name}.py --dataset {dataset} --expname {combined_name.upper()}_{dataset.upper()}_{i} {exp_arg} --cuda""")

if __name__=="__main__":
	for arch in "dcgan","sagan":
		for puzzler_size in 0,4,16,20:
			for dataset in "mnist", "cifar10", "celeba":
				gen(email, dataset, arch, puzzler_size)

	print(sys.argv)
	if len(sys.argv)>1:
		keyword = sys.argv[1].lower()
		for y in get_sh(script_dir):
			if keyword in y.lower():
				Popen(f'sbatch {y}', shell=True)
                                print(f'Launching {y}')

	Popen('find . -type f -iname "*.sh" -exec chmod +x {}  \;', shell=True)
