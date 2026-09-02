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

    print("Single network per run")

    if not direc_run.exists():
        os.makedirs(direc_run)

    with open(args.cl_filename, 'r') as stream:
        params = yaml.safe_load(stream)

    # Print parameters
    ic(params)

    model = NN.NeuralNetworkClassifier(use_domain_adaptation=args.use_domain_adaptation,
                                       save_path=direc_run, 
                                       n_inputs=args.inputs, 
                                       layers=params["layers"], 
                                       early_stopping=False, 
                                       val_split=0.5, lr=params["lr"], 
                                       batch_size=params["batch_size"], 
                                       epochs=params["epochs"], 
                                       dropout=params["dropout"], 
                                       weight_decay=params["weight_decay"], 
                                       domain_classifier =args.domain_classifier,
                                       momentum=params["momentum"],
                                       domain_weight=args.domain_weight
                                       )

    if args.use_domain_adaptation:
        model.fit(X=X_train, 
                  y=Y_train,
                  X_domain=X_domain, 
                  y_domain=Y_domain)
    else:
        model.fit(X=X_train,
                  y=Y_train)

    test_preds = model.get_all_predictions(X_test, use_domain_adaptation=args.use_domain_adaptation)


    auc_val_sic = metrics.plot_roc(average_preds_metric(test_preds, - np.load(model._val_SIC_path())), 
                                   Y_test, 
                                   title="val_sic", 
                                   directory=args.directory)

    print(f"AUC val SIC: {auc_val_sic:.3f}")
    
    auc_val_loss = metrics.plot_roc(average_preds_metric(test_preds, np.load( model._val_loss_path())), 
                                    Y_test, 
                                    title="val_loss", 
                                    directory=args.directory)

    print(f"AUC val loss: {auc_val_loss:.3f}")


    auc_last_epoch = metrics.plot_roc(test_preds[-1], 
                                    Y_test, 
                                    title="last_epoch", 
                                    directory=args.directory)

    print(f"AUC last epoch: {auc_last_epoch:.3f}")

    del test_preds

    if args.signal_number == 0:
        sculpting_preds = model.get_all_predictions(sculpting_test_set, use_domain_adaptation=args.use_domain_adaptation)[args.start_epoch_selection:]


        np.save(str(direc_run / "sculpting_val_SIC_preds.npy"), average_preds_metric(sculpting_preds, -np.load(model._val_SIC_path())[args.start_epoch_selection:]))

        np.save(str(direc_run / "sculpting_val_loss_preds.npy"), average_preds_metric(sculpting_preds, -np.load(model._val_loss_path())[args.start_epoch_selection:]))

        np.save(str(direc_run / "sculpting_last_epoch_preds.npy"), sculpting_preds[-1])

    shutil.rmtree(str(direc_run / "CLSF_models"), ignore_errors=False, onerror=None)
