#!/bin/bash
#SBATCH --job-name=first_test
#SBATCH --ntasks=1

OUTFILE="/data1/val2204/data/myjob_${SLURM_JOB_ID}.out"
ERRFILE="/data1/val2204/data/myjob_${SLURM_JOB_ID}.err"
source venv/bin/activate

PROFILES_JSON='[
  [
    {"type": "gauss", "params": {"A": -6}, "label": "Gaussian (A=-6, k=-0.1)", "name": "pr1"},
    {"type": "gauss", "params": {"A": 0}, "label": "Gaussian (A=0, k=-0.1) (linear)", "name": "pr2"},
    {"type": "gauss", "params": {"A": 0, "k": -0.2}, "label": "Gaussian (A=0, k=-0.2) (linear)", "name": "pr3"}
  ],
  [
    {"type": "linear", "params": {"slope": -0.1}, "label": "Linear (slope=-0.1)", "name": "pr4"},
    {"type": "linear", "params": {"slope": -0.07}, "label": "Linear (slope=-0.07)", "name": "pr5"},
    {"type": "gauss", "params": {"A": -3}, "label": "Gaussian (A=-3)", "name": "pr6"}
  ],
  [
    {"type": "linear", "params": {"slope": -0.1}, "label": "Linear (slope=-0.1)", "name": "pr1"},
    {"type": "linear", "params": {"slope": -0.2}, "label": "Linear (slope=-0.2)", "name": "pr4"},
    {"type": "linear", "params": {"slope": -0.15}, "label": "Linear (slope=-0.15)", "name": "pr3"}
  ],

]'

python /home/val2204/translocation-task/new_different_depths.py --profiles_json "$PROFILES_JSON" --N 50 > ${OUTFILE} 2> ${ERRFILE}