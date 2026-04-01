#!/bin/bash
#SBATCH --job-name=example   # Job name
#SBATCH --mail-type=ALL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yj2278@columbia.edu         # Where to send mail (e.g. uni123@columbia.edu)
#SBATCH --time=0-10:05:00             # Time limit hrs:min:sec
#SBATCH --output=array_%A-%a.log    # Standard output and error log
#SBATCH --array=0       # array range number of channels in neuropixel 
#SBATCH --mem-per-cpu=2gb
#SBATCH -c 1    #number of cpu cores

monkey=
date=
python analyze_bystim.py $1 $monkey $date