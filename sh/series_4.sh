#!/bin/bash
#SBATCH --job-name=first_test
#SBATCH --ntasks=1

#SBATCH --output=/data1/val2204/data/slurm_%x_%j.out
#SBATCH --error=/data1/val2204/data/slurm_%x_%j.err

OUTFILE="/data1/val2204/data/myjob_${SLURM_JOB_ID}.out"
ERRFILE="/data1/val2204/data/myjob_${SLURM_JOB_ID}.err"
source venv/bin/activate

PROFILES_JSON="$(< /home/val2204/translocation-task/sh/profiles.json)"

python /home/val2204/translocation-task/new_loss_function.py \
  --profiles_json "$PROFILES_JSON" \
  --N 50 --log_scale \
   > ${OUTFILE} 2> ${ERRFILE}