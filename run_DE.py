import dataprep_utils as dp
import argparse
import os
from sklearn.model_selection import train_test_split
import numpy as np
import DE_CFM_utils as DE

parser = argparse.ArgumentParser(
    description='Run the full CATHODE analysis chain.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# parameters to vary for the used runs
parser.add_argument('--directory', type=str, required=True)
parser.add_argument('--apply_random_rotation', default=False, action="store_true", help="Applies random rotation to input feature space")
parser.add_argument('--signal_number', type=int, default=1000, help="number of signal events")
parser.add_argument('--unhelpful_features', default=False, action="store_true")
parser.add_argument('--include_DeltaR', default=False, action="store_true", help="appending Delta R to input feature set")

# Need to pass file locations
parser.add_argument('--data_file', type=str, default="/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_v2.extratau_2.features.h5")
parser.add_argument('--extrabkg_file', type=str, default="/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_DelphesPythia8_v2_qcd_extra_inneronly_combined_extratau_2_features.h5")
parser.add_argument('--samples_file', type=str, default=None)

# Dataset arguments
parser.add_argument('--input_set', type=str, default="baseline", choices=["baseline", "baseline41" , "extended1", "extended2", "extended3", "extended7"])
parser.add_argument('--inputs', type=int, default=4)

parser.add_argument('--signal_file', type=str, default=None, help="Specify different signal file")
parser.add_argument('--three_pronged', default=False, action="store_true", help="Activate three-pronged signal file")
parser.add_argument('--minmass', type=float, default=3.3, help="SR lower edge in TeV")
parser.add_argument('--maxmass', type=float, default=3.7, help="SR upper edge in TeV")
parser.add_argument('--gaussian_inputs', default=None, type=int, help="Adds N uninformative Gaussian features")
parser.add_argument('--N_normal_inputs', default=4, type=int, help="Needs to be set for Gaussian inputs")
parser.add_argument('--N_samples', type=int, default=1000000)

# Seeds for dataset preparation
parser.add_argument('--set_seed', type=int, default=1, help="Changes seed used for shuffling")
parser.add_argument('--randomize_seed', default=False, action="store_true", help="Randomizes shuffling")
parser.add_argument('--randomize_signal', default=None, help="Set to int if signal randomization wanted")

# HP for CFM handled via arguments
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--frequencies', type=int, default=3)
parser.add_argument('--num_blocks', type=int, default=4)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--non_linear_context', action='store_true', help='if non linear context is used')

#Classifier Arguments
parser.add_argument('--N_runs', type=int, default=1, help="Number of runs wanted for DE sampling")
parser.add_argument('--start_at_run', type=int, default=0, help="Allows restart at higher run numbers")

parser.add_argument('--density_estimation', default=True, action="store_false")

args = parser.parse_args()

if not os.path.exists(args.directory):
	os.makedirs(args.directory)

if args.unhelpful_features:
    args.input_set="extended1"

if args.input_set=="extended1":
    args.inputs=10
elif args.input_set=="extended1_small":
    args.inputs=6
elif args.input_set=="extended2":
    args.inputs=12
elif args.input_set=="extended3":
    args.inputs=56
elif args.input_set=="extended7":
    args.inputs=20

if args.include_DeltaR:
    args.inputs+=1

if args.three_pronged: # change if not run on HPC RWTH
	args.signal_file = "/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_Z_XY_qqq.extratau_2.features.h5"
     
print(args)

innerdata, outerdata = dp.classifier_data_prep(args)
for i in range(args.start_at_run, args.N_runs):
    print()
    print("------------------------------------------------------")
    print()
    print("DE run no. ", i)
    print()
    direc_run = args.directory+"run"+str(i)+"/"

    DE.run_DE(args, innerdata, outerdata, direc_run)

samples_inner = np.zeros((args.N_runs*args.N_samples,args.inputs+1))
samples_outer = np.zeros((args.N_runs*args.N_samples,args.inputs+1))
for i in range(args.N_runs):
    samples_inner[i*args.N_samples:(i+1)*args.N_samples] = np.load(args.directory+"run"+str(i)+"/samples_inner.npy")
    samples_outer[i*args.N_samples:(i+1)*args.N_samples] = np.load(args.directory+"run"+str(i)+"/samples_outer.npy")
inds = np.array(range(args.N_runs*args.N_samples))
np.random.shuffle(inds)
np.save(args.directory+"samples_inner.npy", samples_inner[inds])
np.save(args.directory+"samples_outer.npy", samples_outer[inds])
