#!/usr/local_rwth/bin/zsh

### Job name
#SBATCH --job-name=classifier_runs
#SBATCH --account rwth0934

#SBATCH --output=../output/%A_%a.txt

### Compute time --> calculated generously
#SBATCH --time=0-5:00:00
#SBATCH --mem=20G
#SBATCH --ntasks=1

#SBATCH --gres=gpu:1
#SBATCH --array=0,50,100,150,200,250,300,400,500,600,700,800,900,1000

### Activate environment
source /home/zu992399/.bashrc
conda activate pytorch-rocky9

## Go into run directory
cd ..

signal_number=${SLURM_ARRAY_TASK_ID}
mode=${1}
input_set=${2}
domain_adaptation=${3}
samples_direc=""

gen_direc="/hpcwork/p0025242/weak_supervision_domain_adaptation/domain_weight_None_ensemble5_start50/"
#gen_direc="/hpcwork/p0025242/weak_supervision_domain_adaptation/domain_weight_${4}_regression/"
DE_direc="/hpcwork/p0025242/weak_supervision_domain_adaptation/"

if [[ "${domain_adaptation}" == "True" ]]; then
    direc="${gen_direc}${mode}/${input_set}_DA/Nsig_${signal_number}/"
    extra_args="-d"
else
    direc="${gen_direc}${mode}/${input_set}/Nsig_${signal_number}/"
    extra_args=""
fi

samples_string=""
if [[ "${input_set}" == "DeltaR" ]]; then
    extra_args="${extra_args} --include_DeltaR"
    samples_string="_DeltaR"
elif [[ "${input_set}" == "shifted" ]]; then
    extra_args="${extra_args} --shift_masses"
    samples_string="_shifted"
fi

if [[ "${mode}" == "cathode" ]]; then
    if 
    extra_args="${extra_args} --samples_file "${DE_direc}DE/baseline${samples_string}/Nsig_${signal_number}/samples_inner.npy""
elif [[ "${mode}" == "lacathode" ]]; then
    extra_args="${extra_args} --DE_direc "${DE_direc}DE/baseline${samples_string}/Nsig_${signal_number}/""
fi

python run_pipeline.py --signal_number ${SLURM_ARRAY_TASK_ID} --directory ${direc} --mode ${mode} ${extra_args} --randomize_signal 0 --domain_weight ${4} --domain_classifier --N_ensemble_networks 5 --start_epoch_selection 50