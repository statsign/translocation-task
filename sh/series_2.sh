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
    {"type": "gauss", "params": {"A": -6}, "label": "Gaussian (A=-6, k=-0.1)", "name": "pr4"},
    {"type": "gauss", "params": {"A": 0}, "label": "Gaussian (A=0, k=-0.1) (linear)", "name": "pr5"},
    {"type": "gauss", "params": {"A": 0, "k": -0.2}, "label": "Gaussian (A=0, k=-0.2) (linear)", "name": "pr6"}
  ],
  [
    {"type": "linear", "params": {"slope": -0.1}, "label": "Linear (slope=-0.1)", "name": "pr1"},
    {"type": "linear", "params": {"slope": -0.07}, "label": "Linear (slope=-0.07)", "name": "pr2"},
    {"type": "gauss", "params": {"A": -3}, "label": "Gaussian (A=-3)", "name": "pr7"}
  ],
    [
    {"type": "gauss", "params": {"A": 1}, "label": "Gaussian (A=1)", "name": "pr8"},
    {"type": "gauss", "params": {"A": 3}, "label": "Gaussian (A=3)", "name": "pr9"},
    {"type": "gauss", "params": {"A": 6}, "label": "Gaussain (A=6)", "name": "pr11"}
  ],
    [
    {"type": "linear", "params": {"slope": 0.1}, "label": "Linear (slope=0.1)", "name": "pr12"},
    {"type": "linear", "params": {"slope": 0.07}, "label": "Linear (slope=0.07)", "name": "pr13"},
    {"type": "linear", "params": {"slope": 0.08}, "label": "Linear (slope=0.08)", "name": "pr14"}
  ],
      [
    {"type": "gauss", "params": {"A": 1, "k": 0}, "label": "Gaussian (A=1)", "name": "pr15"},
    {"type": "gauss", "params": {"A": 3, "k": 0}, "label": "Gaussian (A=3)", "name": "pr16"},
    {"type": "gauss", "params": {"A": 6, "k": 0}, "label": "Gaussain (A=6)", "name": "pr17"}
  ]
]'

python /home/val2204/translocation-task/new_different_depths.py --profiles_json "$PROFILES_JSON" --N 50 --log_scale > ${OUTFILE} 2> ${ERRFILE}