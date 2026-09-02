#!/usr/local_rwth/bin/zsh

### Job name
#SBATCH --job-name=DE_runs
###SBATCH --account rwth0934

#SBATCH --output=../output/DE/%J.txt

### Compute time --> calculated generously
#SBATCH --time=0-5:00:00
#SBATCH --mem=12G
#SBATCH --ntasks=1

#SBATCH --gres=gpu:1
#SBATCH --array=0,50,100,150,200,250,300,400,500,600,700,800,900,1000

### Activate environment
source /home/zu992399/.bashrc
conda activate pytorch-rocky9

## Go into run directory
cd ..
signal_number=${SLURM_ARRAY_TASK_ID}
echo "Signal number: $signal_number"
shifted=${1}
include_DeltaR=${2}

Dextra_args=""

#Needs to be changed for different cluster
direc="/work/zu992399/weak_supervision_domain_adaptation/DE/baseline"
if [[ "$shifted" == "True" ]]; then
    direc="${direc}_shifted"  
    extra_args="${extra_args} --shift_masses"
fi

if [[ "$include_DeltaR" == "True" ]]; then
    direc="${direc}_DeltaR/Nsig_${signal_number}/"
    extra_args="${extra_args} --include_DeltaR"
else
    direc="${direc}/Nsig_${signal_number}/"
fi

num_blocks=4
frequencies=20
batch_size=4096
epochs=2000
hidden_dim=256

# final command
python run_DE.py \
    --signal_number ${signal_number} \
    --directory ${direc} \
    --start_at_run 6  \
    --N_runs 10 \
    --epochs ${epochs} --batch_size ${batch_size} \
    --frequencies ${frequencies} --hidden_dim ${hidden_dim} \
    --num_blocks ${num_blocks} \
    ${extra_args}