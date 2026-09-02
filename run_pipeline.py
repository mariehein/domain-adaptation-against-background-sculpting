import dataprep_utils as dp
import argparse
import os
from icecream import ic
from pathlib import Path

parser = argparse.ArgumentParser(
    description='Run the full CATHODE analysis chain.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# parameters to vary for the used runs
parser.add_argument('--mode', type=str, choices=["cathode", "cwola", "IAD", "supervised", "lacathode"], required=True)

parser.add_argument('-d', '--use_domain_adaptation', action='store_true', help='Use domain adaptation during the minimization process. Be careful, the model changes by doing this!')

parser.add_argument('--cl_filename', type=str, default=None,
                    help="For default HP change nothing")

parser.add_argument('--directory', type=Path, required=True)

parser.add_argument('--signal_number', type=int,
                    default=1000, help="number of signal events")

parser.add_argument('--include_DeltaR', default=False,
                    action="store_true", help="appending Delta R to input feature set")

parser.add_argument('--shift_masses', default=False,
                    action="store_true", help="shift mass features by 0.1 m_JJ")

parser.add_argument('--domain_weight', type=float, default=None, help="weight for domain adaptation loss, None means homoscedastic loss weighting"),


parser.add_argument('--DE_direc', type=str, default=None, help="DE directory for LaCATHODE")

# DE args for LaCATHODE
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--frequencies', type=int, default=20)
parser.add_argument('--num_blocks', type=int, default=4)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--non_linear_context', action='store_true', help='if non linear context is used')


file_path = Path("data/")

def check_path(base_file: Path, file_name: str) -> Path:
    file_path = base_file / file_name
    if not file_path.exists():
        raise ValueError(f"File {file_name} does not exist in {base_file}")
    return str(file_path)


# Need to pass file locations
parser.add_argument('--data_file', type=str, default=check_path(file_path, "events_anomalydetection_v2.features.h5"))

parser.add_argument('--extrabkg_file', type=str, default=check_path(file_path, "events_anomalydetection_qcd_extra_inneronly_features.h5"))

parser.add_argument('--sculpting_file', type=str, default=check_path(file_path, "Pythia_QCD_Dijet_Events.h5"))

parser.add_argument('--samples_file', type=str, default=None)

# Dataset arguments
parser.add_argument('--input_set', type=str, default="baseline", choices=["baseline", "baseline41", "extended1", "extended2", "extended3", "extended7"])

parser.add_argument('--inputs', type=int, default=4)

parser.add_argument('--signal_file', type=str, default=None, help="Specify different signal file")

parser.add_argument('--three_pronged', default=False, action="store_true", help="Activate three-pronged signal file")

parser.add_argument('--minmass', type=float, default=3.3, help="SR lower edge in TeV")

parser.add_argument('--maxmass', type=float, default=3.7, help="SR upper edge in TeV")

parser.add_argument('--cl_norm', default=True, action="store_false", help="Classifier input normalisation (mean=0 and std=1)")

parser.add_argument('--oversampling_factor', type=float, default=4, help="CATHODE oversampling factor")

parser.add_argument('--ssb_width', type=float, default=0.2, help="Short side band width for cwola hunting")

# Seeds for dataset preparation
parser.add_argument('--set_seed', type=int, default=1, help="Changes seed used for shuffling")

parser.add_argument('--randomize_seed', default=False, action="store_true", help="Randomizes shuffling")

parser.add_argument('--randomize_signal', default=None, help="Set to int if signal randomization wanted")

# Classifier Arguments
parser.add_argument('--N_runs', type=int, default=10, help="Number of runs wanted for errors")

parser.add_argument('--start_at_run', type=int, default=0, help="Allows restart at higher run numbers")

parser.add_argument('--N_best_epochs', type=int, default=10, help="NN best epoch averaging")

parser.add_argument('--start_epoch_selection', type=int, default=0)

parser.add_argument('--N_ensemble_networks', type=int, default=1, help="NN network ensembling")

parser.add_argument('--density_estimation', default=False, action="store_true")

parser.add_argument('--domain_classifier', default=False, action="store_true", help="Use domain classifier for domain adaptation")

args = parser.parse_args()

if args.N_ensemble_networks!=1:
    import NN_ensemble_utils as cl
else:
    import NN_utils as cl

if not args.directory.exists():
    os.makedirs(args.directory)

if args.cl_filename == None:
    args.cl_filename = "hp_NN_default.yaml"

if args.include_DeltaR:
    args.inputs += 1

if args.three_pronged:  # change if not run on HPC RWTH
    args.signal_file = "/hpcwork/rwth0934/LHCO_dataset/original/events_anomalydetection_Z_XY_qqq.features.h5"

print(args)

if not args.randomize_seed and args.randomize_signal is None:
    X_train, Y_train, X_domain, Y_domain, X_test, Y_test, sculpting_test_set = dp.classifier_data_prep(
        args)

    #ic(X_train.shape, Y_train.shape, X_domain.shape, Y_domain.shape,
    #   X_test.shape, Y_test.shape, sculpting_test_set.shape)

for i in range(args.start_at_run, args.N_runs):
    print()
    print("------------------------------------------------------")
    print()
    print("Classifier run no. ", i)
    print()
    direc_run = args.directory / f"run{i}/"
    if args.randomize_seed or args.randomize_signal is not None:
        args.set_seed = i
        if args.randomize_signal is not None:
            args.randomize_signal = i
        X_train, Y_train, X_domain, Y_domain, X_test, Y_test, sculpting_test_set = dp.classifier_data_prep(
            args, run = i)
    cl.classifier_training(X_train, Y_train, X_domain, Y_domain, X_test,
                           Y_test, sculpting_test_set, args, i, direc_run=direc_run)
