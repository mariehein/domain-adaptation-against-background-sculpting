mode=("lacathode")
# "cwola" "cathode")
#("IAD" "cwola")

input_set=("baseline" "DeltaR")
#input_set=("DeltaR")

#domain_adaptation=("True")
alpha=(3)
#alpha=(1.2 2 3 4 5)
domain_adaptation=("False")

for m in ${mode}; do 
    for i in ${input_set}; do 
        for d in ${domain_adaptation}; do 
            for a in ${alpha}; do
                sbatch classifier_runs.slurm ${m} ${i} ${d} ${a}
            done
        done
    done
done
