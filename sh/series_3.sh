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
    {"type": "quadratic", "params": {"a": 0.007, "b": 25, "c": -4}, "label": "Quadratic (a=0.007)", "name": "pr4"},
    {"type": "quadratic", "params": {"a": 0.008, "b": 25, "c": -5}, "label": "Quadratic (a=0.008)", "name": "pr5"},
    {"type": "quadratic", "params": {"a": 0.01, "b": 25, "c": -6}, "label": "Quadratic (a=0.01)", "name": "pr6"}
  ],
  [
    {"type": "quadratic", "params": {"a": 0.012, "b": 25, "c": -7}, "label": "Quadratic (a=0.012)", "name": "pr7"},
    {"type": "quadratic", "params": {"a": 0.008, "b": 25, "c": -5}, "label": "Quadratic (a=0.008)", "name": "pr5"},
    {"type": "quadratic", "params": {"a": 0.01, "b": 25, "c": -6}, "label": "Quadratic (a=0.01)", "name": "pr6"}
  ],
  [
    {"type": "gauss", "params": {"A": -1}, "label": "Gaussian (A=-1)", "name": "pr8"},
    {"type": "gauss", "params": {"A": -3}, "label": "Gaussian (A=-3)", "name": "pr9"},
    {"type": "gauss", "params": {"A": 0}, "label": "Gaussian (A=0)", "name": "pr10"}
  ],
  [
    {"type": "gauss", "params": {"A": -1}, "label": "Gaussian (A=-1)", "name": "pr8"},
    {"type": "gauss", "params": {"A": -3}, "label": "Gaussian (A=-3)", "name": "pr9"},
    {"type": "gauss", "params": {"A": -7}, "label": "Gaussain (A=-7)", "name": "pr11"}
  ]
]'

python /home/val2204/translocation-task/different_depths.py --profiles_json "$PROFILES_JSON" --N 50 --log_scale > ${OUTFILE} 2> ${ERRFILE}