#!/bin/env bash

#$ -cwd  
#$ -l scratch=1G
#$ -l mem_free=2G
#$ -l h_rt=00:30:00
#$ -pe smp 30
#$ -j y
#$ -o logs/

echo "Activating virtual environment ..."
source ~/imsms_venv/bin/activate

sample="feces"
data_type="without_outlier"
python_code="GLM_model_global_iMSMS_compound.py"

echo "Running python code ..."
python $python_code $sample $data_type $NSLOTS

[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"
