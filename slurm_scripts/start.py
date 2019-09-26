import os
import sys
from subprocess import Popen
from pathlib import Path

sh_root = Path("./slurm_scripts")

def get_sh(path, ext=".sh"):
    for ds,s,fs in os.walk(path):
        if ds.lower() in ("old", "~archive"):
            continue
        for f in fs:
            if f[-len(ext):] == ext:
                yield os.path.join(ds,f)

def start_scripts():
    if len(sys.argv)>1:
        keyword = sys.argv[1].lower()
        for y in get_sh(sh_root):
            if keyword in y.lower():
                Popen(f'sbatch {y}', shell=True)
                print(f'Launching {y}')


if __name__=="__main__":
    start_scripts()
