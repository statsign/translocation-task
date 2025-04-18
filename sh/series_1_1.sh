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
    {"type": "gauss", "params": {"A": 0}, "label": "Gaussian (A=0)", "name": "gaussA0"}
  ],
  [
    {"type": "gauss", "params": {"A": -1}, "label": "Gaussian (A=-1)", "name": "gaussA1"},
    {"type": "gauss", "params": {"A": -3}, "label": "Gaussian (A=-3)", "name": "gaussA3"},
    {"type": "gauss", "params": {"A": -4}, "label": "Gaussain (A=-4)", "name": "gaussA4"}
  ],
  [
    {"type": "gauss", "params": {"A": 1}, "label": "Gaussian (A=1)", "name": "gausspA1"},
    {"type": "gauss", "params": {"A": 3}, "label": "Gaussian (A=3)", "name": "gausspA3"},
    {"type": "gauss", "params": {"A": 0}, "label": "Gaussian (A=0)", "name": "gaussA0"}
  ],
  [
    {"type": "gauss", "params": {"A": 1}, "label": "Gaussian (A=1)", "name": "gausspA1"},
    {"type": "gauss", "params": {"A": 3}, "label": "Gaussian (A=3)", "name": "gausspA3"},
    {"type": "gauss", "params": {"A": 4}, "label": "Gaussain (A=4)", "name": "gausspA4"}
  ],
   [
    {"type": "gauss", "params": {"A": -1, "k": 0}, "label": "Gaussian (A=-1)", "name": "gaussA1k0"},
    {"type": "gauss", "params": {"A": -3, "k": 0}, "label": "Gaussian (A=-3)", "name": "gaussA3k0"},
    {"type": "gauss", "params": {"A": -4, "k": 0}, "label": "Gaussain (A=-4)", "name": "gaussA4k0"}
  ],
     [
    {"type": "gauss", "params": {"A": 1, "k": 0}, "label": "Gaussian (A=1)", "name": "gausspA1k0"},
    {"type": "gauss", "params": {"A": 3, "k": 0}, "label": "Gaussian (A=3)", "name": "gausspA3k0"},
    {"type": "gauss", "params": {"A": 4, "k": 0}, "label": "Gaussain (A=4)", "name": "gaussA4k0"}
  ],
    [
    {"type": "gauss", "params": {"A": -4}, "label": "Gaussian (A=-4, k=-0.1)", "name": "gaussA4"},
    {"type": "gauss", "params": {"A": 0}, "label": "Gaussian (A=0, k=-0.1) (linear)", "name": "gaussA4k0"},
    {"type": "gauss", "params": {"A": 0, "k": -0.2}, "label": "Gaussian (A=0, k=-0.2) (linear)", "name": "gaussA0k0_2"}
  ],
  [
    {"type": "linear", "params": {"slope": -0.1}, "label": "Linear (slope=-0.1)", "name": "linear0_1"},
    {"type": "linear", "params": {"slope": -0.07}, "label": "Linear (slope=-0.07)", "name": "linear0_07"},
    {"type": "gauss", "params": {"A": -3}, "label": "Gaussian (A=-3)", "name": "gaussA3"}
  ],
      [
    {"type": "gauss", "params": {"A": -1, "k": -0.15}, "label": "Gaussian (A=-1)", "name": "gaussA1k0_15"},
    {"type": "gauss", "params": {"A": -2, "k": -0.15}, "label": "Gaussian (A=-2)", "name": "gaussA2k0_15"},
    {"type": "gauss", "params": {"A": -3, "k": -0.15}, "label": "Gaussain (A=-3)", "name": "gaussA3k0_15"}
  ],
  [
    {"type": "gauss", "params": {"A": -1, "k": -0.1}, "label": "Gaussian (A=-1, k=-0.1)", "name": "gaussA1k0_1"},
    {"type": "gauss", "params": {"A": -1, "k": -0.15}, "label": "Gaussian (A=-1, k=-0.15)", "name": "gaussA1k0_15"},
    {"type": "gauss", "params": {"A": -1, "k": -0.2}, "label": "Gaussain (A=-1, k=-0.2)", "name": "gaussA1k0_2"}
  ],
  [
    {"type": "gauss", "params": {"A": -1, "sigma": 2}, "label": "Gaussian (A=-1, sigma=2)", "name": "gaussA1sigma2"},
    {"type": "gauss", "params": {"A": -1, "sigma": 3}, "label": "Gaussian (A=-1, sigma=3)", "name": "gaussA1sigma3"},
    {"type": "gauss", "params": {"A": -1, "sigma": 4}, "label": "Gaussain (A=-1, sigma=4)", "name": "gaussA1sigma4"}
  ]
]'

python /home/val2204/translocation-task/new_different_depths.py --profiles_json "$PROFILES_JSON" --N 50 > ${OUTFILE} 2> ${ERRFILE}