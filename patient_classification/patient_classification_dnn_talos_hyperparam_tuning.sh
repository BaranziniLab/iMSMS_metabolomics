#!/bin/env bash

#$ -cwd  
#$ -l scratch=1G
#$ -l mem_free=2G
#$ -l h_rt=04:00:00
#$ -pe smp 80
#$ -j y
#$ -o ../logs/

echo "Activating virtual environment ..."
source ~/imsms_venv/bin/activate

sample="feces"
python_code="patient_classification_dnn_talos_hyperparam_tuning.py"

echo "Running python code ..."
python $python_code $sample $NSLOTS

[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
