# Domain Adaptation Against Background Sculpting

This repository contains the code for the following paper:

*Domain Adaptation Against Background Sculpting*
By Vincent Benne, Marie Hein, Michael Krämer, Humberto Reyes-González, Philipp Soldin and Christopher Wiebusch.
[arxiv:26XX.XXXXX](https://arxiv.org/abs/26XX.XXXXX)

## Structure of the repository and code

This repository contains the following structure: 
- Code to run anomaly detection classifier: 
    - To run, use ```run_pipeline.py```, which requires passing the ```--mode``` (must be one of "IAD", "cathode", "cwola", "supervised" "lacathode") and the ```--directory```, in which to save run files.
    - ```run_pipeline.py```calls other files as needed.
- Code to run density estimation for CATHODE and LaCATHODE. 
    - To run, use ```run_DE.py```.
    - ```run_pipeline.py```calls other files as needed.
- Run cards used to produce paper results in folder ```run_cards/```

## Reproducing the paper results 

*We only walk through the classifier runs here as the density estimation is not the focus of this work. ```run_cards/DE_runs.slurm``` should contain all information needed to reproduce the density estimation runs.*

The easiest way of obtaining the paper results is to use the run cards found in folder ```run_cards```, which use slurm.

For this, the following things must be adjusted: 
- Calling the python installation.
- Directories of DE samples: ```DE_direc```.
- Directories to save results: ```gen_direc```.
- Any requirements of your local cluster. 

The run cards ```classifier_runs.slurm``` and ```classifier_runs_regression.slurm``` are designed to require no changes between runs and to produce all paper results based on the passed arguments. They require, in this order:
1. The ```mode```: "IAD", "cathode", "cwola", "lacathode" passed as string, iterated over to produce all paper results ("lacathode" only without domain adaptation). 
2. The input set: "baseline" or "DeltaR" passed as string, iterate over to produce all paper results. Additional option of "shifted" shifts jet mass features by 10\% of the respective dijet mass to introduce correlations, not used in paper
3. Domain adaptation boolean: "True" or "False" passed as string, with "True" meaning that domain adaptation is used (Note: only ```classifier_runs.slurm``` is designed to run without domain adaptation, ```classifier_runs_regression.slurm``` may overwrite other results if used with domain adaptation "False").
4. Domain adaptation loss weight $\lambda$: passed as float, pick accordingly for each method. Domain adaptation "False" ignores this input.
Based on these inputs, directories and arguments for ```run_pipeline.py``` are constructed. Each call iterates over signal numbers as required for full paper results. 

If you plan to run without the given ```run_cards```, the following options must be chosen appropriately: 
- ```--signal_number```: Signal number, scan ```[0,50,100,150,200,250,300,400,500,600,700,800,900,1000]```.
- ```--directory```: Save directory, choose such that runs are not overwritten.
- ```--mode```: method of obtaining background template, iterate over "IAD", "cathode", "cwola", "lacathode"
- ```--domain_weight```: Pass float of domain adaptation loss weight $\lambda$.
- ```--include_DeltaR```: Use if runs for "Baseline+$\Delta R$" feature sets are to be obtained.
- ```-d```: Activates domain adaptation if used. 
- ```--domain classifier```: Activates the use of the classification comain task, regression task is used by default. 

The following arguments are kept constant across all runs:
- ```--randomize_signal 0```: Activates random selection of signal events.
- ```--N_ensemble_networks 5```: Acitvates ensembling of 5 neural networks for full results. 
- ```--start_epoch_selection 50```: Allows selection of best epochs only after epoch 50. 

## Plotting of paper results

We provide our fully self-containing plotting notebook with ```plotting_paper.ipynb```. If the provided run_cards were used and all runs performed, adjusting only the variable ```"directory"``` in the second cell should allow for the production of all plots as shown in the paper. Otherwise, adjusting the lists of directories ```listofdicts``` in cell 5 according to the used folder structure should allow for all required changes. 