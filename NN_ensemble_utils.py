import NN_model_utils as NN
import metric_utils as metrics
import yaml
import os
import numpy as np
import shutil
from icecream import ic
from pathlib import Path


def average_preds_metric(preds, metric, N_epochs=10):
    best_epochs = np.argsort(metric)[:N_epochs]
    return np.mean(preds[best_epochs], axis=0)


def classifier_training(X_train, Y_train, X_domain, Y_domain, X_test, Y_test, sculpting_test_set, args, run, direc_run: Path=None):
    if direc_run is None:
        direc_run = Path(args.directory)

    print("Ensembling networks")

    if not direc_run.exists():
        os.makedirs(direc_run)

    with open(args.cl_filename, 'r') as stream:
        params = yaml.safe_load(stream)

    # Print parameters
    ic(params)

    test_preds_val_sic = np.zeros(len(X_test))
    test_preds_val_loss = np.zeros(len(X_test))
    test_preds_last_epoch = np.zeros(len(X_test))
    sculpting_preds_val_sic = np.zeros(len(sculpting_test_set))
    sculpting_preds_val_loss = np.zeros(len(sculpting_test_set))
    sculpting_preds_last_epoch = np.zeros(len(sculpting_test_set))


    for i in range(args.N_ensemble_networks):

        inds = np.arange(len(X_train))
        np.random.shuffle(inds)
        X_train = X_train[inds]
        Y_train = Y_train[inds]

        if args.use_domain_adaptation:
            inds_domain = np.arange(len(X_domain))
            np.random.shuffle(inds_domain)
            X_domain = X_domain[inds_domain]
            Y_domain = Y_domain[inds_domain]


        model = NN.NeuralNetworkClassifier(use_domain_adaptation=args.use_domain_adaptation,
                                        save_path=direc_run / f"model{i}/", 
                                        n_inputs=args.inputs, 
                                        layers=params["layers"], 
                                        early_stopping=False, 
                                        val_split=0.5, lr=params["lr"], 
                                        batch_size=params["batch_size"], 
                                        epochs=params["epochs"], 
                                        dropout=params["dropout"], 
                                        weight_decay=params["weight_decay"], 
                                        momentum=params["momentum"],
                                        domain_weight=args.domain_weight)

        if args.use_domain_adaptation:
            model.fit(X=X_train, 
                    y=Y_train,
                    X_domain=X_domain, 
                    y_domain=Y_domain)
        else:
            model.fit(X=X_train,
                    y=Y_train)
            
        test_preds_temp = model.get_all_predictions(X_test, use_domain_adaptation=args.use_domain_adaptation)[args.start_epoch_selection:]

        test_preds_val_sic += average_preds_metric(test_preds_temp, -np.load(model._val_SIC_path())[args.start_epoch_selection:], N_epochs=args.N_best_epochs)/args.N_ensemble_networks
        test_preds_val_loss += average_preds_metric(test_preds_temp, np.load(model._val_loss_path())[args.start_epoch_selection:], N_epochs=args.N_best_epochs)/args.N_ensemble_networks
        test_preds_last_epoch += test_preds_temp[-1]/args.N_ensemble_networks
        if args.signal_number == 0:
            sculpting_preds_temp = model.get_all_predictions(sculpting_test_set, use_domain_adaptation=args.use_domain_adaptation)
            sculpting_preds_val_sic += average_preds_metric(sculpting_preds_temp, -np.load(model._val_SIC_path())[args.start_epoch_selection:], N_epochs=args.N_best_epochs)/args.N_ensemble_networks
            sculpting_preds_val_loss += average_preds_metric(sculpting_preds_temp, np.load(model._val_loss_path())[args.start_epoch_selection:], N_epochs=args.N_best_epochs)/args.N_ensemble_networks
            sculpting_preds_last_epoch += sculpting_preds_temp[-1]/args.N_ensemble_networks


    auc_val_sic = metrics.plot_roc(test_preds_val_sic, Y_test, title="val_sic", directory=args.directory)
    print(f"AUC val SIC: {auc_val_sic:.3f}")

    auc_val_loss = metrics.plot_roc(test_preds_val_loss, Y_test, title="val_loss", directory=args.directory)
    print(f"AUC val loss: {auc_val_loss:.3f}")


    auc_last_epoch = metrics.plot_roc(test_preds_last_epoch, Y_test,title="last_epoch", directory=args.directory)
    print(f"AUC last epoch: {auc_last_epoch:.3f}")


    if args.signal_number == 0:
        np.save(str(direc_run / "sculpting_val_SIC_preds.npy"), sculpting_preds_val_sic)
        np.save(str(direc_run / "sculpting_val_loss_preds.npy"), sculpting_preds_val_loss)
        np.save(str(direc_run / "sculpting_last_epoch_preds.npy"), sculpting_preds_last_epoch)

    for i in range(args.N_ensemble_networks):
        shutil.rmtree(str(direc_run / f"model{i}" / "CLSF_models"), ignore_errors=False, onerror=None)

