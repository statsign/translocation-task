#!/bin/bash
#SBATCH --job-name=first_test
#SBATCH --ntasks=1

OUTFILE="/data1/val2204/data/myjob_${SLURM_JOB_ID}.out"
ERRFILE="/data1/val2204/data/myjob_${SLURM_JOB_ID}.err"
source venv/bin/activate

PROFILES_JSON='[
  [
    {"type": "linear", "params": {"slope": -0.1}, "label": "Linear (slope=-0.1)", "name": "pr1"},
    {"type": "linear", "params": {"slope": -0.07}, "label": "Linear (slope=-0.07)", "name": "pr2"},
    {"type": "linear", "params": {"slope": -0.08}, "label": "Linear (slope=-0.08)", "name": "pr3"}
  ],
   [
    {"type": "linear", "params": {"slope": -0.1}, "label": "Linear (slope=-0.1)", "name": "pr1"},
    {"type": "linear", "params": {"slope": -0.2}, "label": "Linear (slope=-0.2)", "name": "pr4"},
    {"type": "linear", "params": {"slope": -0.08}, "label": "Linear (slope=-0.08)", "name": "pr3"}
  ],
  [
    {"type": "gauss", "params": {"A": -1}, "label": "Gaussian (A=-1)", "name": "pr5"},
    {"type": "gauss", "params": {"A": -2}, "label": "Gaussian (A=-2)", "name": "pr6"},
    {"type": "gauss", "params": {"A": 0}, "label": "Gaussian (A=0)", "name": "pr7"}
  ],
  [
    {"type": "gauss", "params": {"A": -1}, "label": "Gaussian (A=-1)", "name": "pr5"},
    {"type": "gauss", "params": {"A": -3}, "label": "Gaussian (A=-3)", "name": "pr6"},
    {"type": "gauss", "params": {"A": -4}, "label": "Gaussain (A=-4)", "name": "pr8"}
  ],
  [
    {"type": "gauss", "params": {"A": 1}, "label": "Gaussian (A=1)", "name": "pr9"},
    {"type": "gauss", "params": {"A": 3}, "label": "Gaussian (A=3)", "name": "pr10"},
    {"type": "gauss", "params": {"A": 0}, "label": "Gaussian (A=0)", "name": "pr7"}
  ],
  [
    {"type": "gauss", "params": {"A": 1}, "label": "Gaussian (A=1)", "name": "pr9"},
    {"type": "gauss", "params": {"A": 3}, "label": "Gaussian (A=3)", "name": "pr10"},
    {"type": "gauss", "params": {"A": 4}, "label": "Gaussain (A=4)", "name": "pr11"}
  ],
   [
    {"type": "gauss", "params": {"A": -1, "k": 0}, "label": "Gaussian (A=-1)", "name": "pr12"},
    {"type": "gauss", "params": {"A": -3, "k": 0}, "label": "Gaussian (A=-3)", "name": "pr13"},
    {"type": "gauss", "params": {"A": -4, "k": 0}, "label": "Gaussain (A=-4)", "name": "pr14"}
  ],
     [
    {"type": "gauss", "params": {"A": 1, "k": 0}, "label": "Gaussian (A=1)", "name": "pr15"},
    {"type": "gauss", "params": {"A": 3, "k": 0}, "label": "Gaussian (A=3)", "name": "pr16"},
    {"type": "gauss", "params": {"A": 4, "k": 0}, "label": "Gaussain (A=4)", "name": "pr17"}
  ]
]'

python /home/val2204/translocation-task/new_different_depths.py --profiles_json "$PROFILES_JSON" --N 50 > ${OUTFILE} 2> ${ERRFILE}