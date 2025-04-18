#!/bin/bash
#SBATCH --job-name=first_test
#SBATCH --ntasks=1

OUTFILE="/data1/val2204/data/myjob_${SLURM_JOB_ID}.out"
ERRFILE="/data1/val2204/data/myjob_${SLURM_JOB_ID}.err"
source venv/bin/activate

PROFILES_JSON='[
  [
    {"type": "linear", "params": {"slope": -0.1}, "label": "Linear (slope=-0.1)", "name": "linear0_1"},
    {"type": "linear", "params": {"slope": -0.07}, "label": "Linear (slope=-0.07)", "name": "linear0_07"},
    {"type": "linear", "params": {"slope": -0.08}, "label": "Linear (slope=-0.08)", "name": "linear0_08"}
  ],
   [
    {"type": "linear", "params": {"slope": -0.1}, "label": "Linear (slope=-0.1)", "name": "linear0_1"},
    {"type": "linear", "params": {"slope": -0.2}, "label": "Linear (slope=-0.2)", "name": "linear0_2"},
    {"type": "linear", "params": {"slope": -0.15}, "label": "Linear (slope=-0.15)", "name": "linear0_15"}
  ],
  [
    {"type": "gauss", "params": {"A": -1}, "label": "Gaussian (A=-1)", "name": "gaussA1"},
    {"type": "gauss", "params": {"A": -2}, "label": "Gaussian (A=-2)", "name": "gaussA2"},
    {"type": "gauss", "params": {"A": 0}, "label": "Gaussian (A=0 (linear))", "name": "gaussA0"}
  ],
  [
    {"type": "gauss", "params": {"A": -1}, "label": "Gaussian (A=-1)", "name": "gaussA1"},
    {"type": "gauss", "params": {"A": -3}, "label": "Gaussian (A=-3)", "name": "gaussA3"},
    {"type": "gauss", "params": {"A": -4}, "label": "Gaussain (A=-4)", "name": "gaussA4"}
  ]
]'

python /home/val2204/translocation-task/new_different_depths.py --profiles_json "$PROFILES_JSON" --N 50 --log_scale > ${OUTFILE} 2> ${ERRFILE}