#!/bin/bash
#SBATCH -p ceashpc
#SBATCH -c 1
#SBATCH --mem=10G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cwang25@albany.edu
#SBATCH -o logs/sbatch_top_9.out
#SBATCH -e logs/sbatch_top_9.error
#SBATCH --time=11-0:1 # The job should take 0 days, 0 hours, 1 minutes

# Now, run the python script

python -u /network/rit/lab/ceashpc/chunpai/PycharmProjects/OfflineEval/offline_eval/eval.py
