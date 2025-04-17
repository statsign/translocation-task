#!/bin/bash
#SBATCH --job-name=first_test
#SBATCH --ntasks=1

OUTFILE="/data1/val2204/myjob_${SLURM_JOB_ID}.out"
ERRFILE="/data1/val2204/myjob_${SLURM_JOB_ID}.err"
source venv/bin/activate

PROFILES_JSON='[
  [
    {"type": "linear", "params": {"slope": -0.1}, "label": "Linear (slope=-0.1)", "name": "pr1"},
    {"type": "linear", "params": {"slope": -0.07}, "label": "Linear (slope=-0.07)", "name": "pr2"},
    {"type": "linear", "params": {"slope": -0.08}, "label": "Linear (slope=-0.08)", "name": "pr3"}
  ],
  [
    {"type": "linear", "params": {"slope": -0.1}, "label": "Linear (slope=-0.1)", "name": "pr1"},
    {"type": "linear", "params": {"slope": -0.2}, "label": "Linear (slope=-0.2)", "name": "pr12"},
    {"type": "linear", "params": {"slope": -0.08}, "label": "Linear (slope=-0.08)", "name": "pr3"}
  ],
  [
    {"type": "gauss", "params": {"A": -1}, "label": "Gaussian (A=-1)", "name": "pr8"},
    {"type": "gauss", "params": {"A": -3}, "label": "Gaussian (A=-3)", "name": "pr9"},
    {"type": "gauss", "params": {"A": 0}, "label": "Gaussian (A=0)", "name": "pr10"}
  ],
  [
    {"type": "gauss", "params": {"A": -1}, "label": "Gaussian (A=-1)", "name": "pr8"},
    {"type": "gauss", "params": {"A": -3}, "label": "Gaussian (A=-3)", "name": "pr9"},
    {"type": "gauss", "params": {"A": -6}, "label": "Gaussain (A=-6)", "name": "pr11"}
  ]
]'

python /home/val2204/translocation-task/different_depths.py --profiles_json "$PROFILES_JSON" --N 50 --log_scale > ${OUTFILE} 2> ${ERRFILE}