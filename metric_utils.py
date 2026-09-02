import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, log_loss
from pathlib import Path
from sklearn.utils import class_weight
from zipfile import ZipFile
import io
import yaml
from filelock import FileLock

def save_array_to_zip(zip_path, array, name):
    """Save a numpy array into a zip as .npy"""
    buf = io.BytesIO()
    np.save(buf, array)
    buf.seek(0)
    lock = FileLock(zip_path+".lock")
    with lock:
        with ZipFile(zip_path, "a") as z:   # "a" = append
            z.writestr(f"{name}.npy", buf.read())

def save_dict_to_zip(zip_path, dictionary, name):
    """Save a Python dict into a zip as .yaml"""
    yaml_text = yaml.dump(dictionary)
    lock = FileLock(zip_path+".lock")
    with lock:
        with ZipFile(zip_path, "a") as z:
            z.writestr(f"{name}.yaml", yaml_text)

def get_val_metrics(test_results, test_labels):
    '''for code readability calculates val_sic and val_loss in one go'''
    return val_sic(test_results, test_labels), val_loss(test_results, test_labels)

def max_sic(test_results, test_labels):
    '''Calculates Max SIC from predictions and labels'''
    max_err = 0.2
    fpr, tpr, _ = roc_curve(test_labels, test_results)
    inds = np.argwhere(fpr > 1/220000/max_err**2)[:,0]
    max_SIC = np.max(tpr[inds]/np.sqrt(fpr[inds]))
    return max_SIC

def val_sic(test_results, test_labels):
    '''Calculates Val SIC from predictions and labels'''
    fpr, tpr, _ = roc_curve(test_labels, test_results)
    inds = np.nonzero(fpr)
    tpr = tpr[inds]
    fpr = fpr[inds]
    max_SIC_fpr = np.max(tpr/np.sqrt(fpr)-np.sqrt(fpr))
    return max_SIC_fpr

def val_loss(test_results, test_labels):
    '''Calculates Val loss from predictions and labels (not one-hot encoded)'''
    class_weight_dict = class_weight.compute_class_weight('balanced', classes=np.unique(test_labels), y=test_labels)
    class_weights = class_weight_dict[0]*(1-test_labels)+class_weight_dict[1]*test_labels
    return log_loss(test_labels, test_results, sample_weight=class_weights)

def make_one_array(twod_arr,new_arr):
    '''Helper function to save ROC curves'''
    cols = twod_arr.shape[1]
    new_len = len(new_arr)
    if new_len < cols: 
        new_arr = np.pad(new_arr, (0, cols-new_len))
    elif new_len > cols: 
        twod_arr = np.pad(twod_arr, ((0,0), (0,new_len-cols)))
    return np.vstack([twod_arr, new_arr])

def plot_roc(test_results, test_labels, directory, title="roc"):
    '''Helper function saving ROC curves from multiple runs into one 2D array'''
    fpr, tpr, _ = roc_curve(test_labels, test_results)

    fpr_path = directory / ("fpr_"+title+".npy")
    tpr_path = directory / ("tpr_"+title+".npy")

    if Path(tpr_path).is_file():
        np.save(tpr_path, make_one_array(np.load(tpr_path), tpr))
        np.save(fpr_path, make_one_array(np.load(fpr_path), fpr))
    else: 
        np.save(tpr_path,np.array([tpr]))
        np.save(fpr_path,np.array([fpr]))
	
    return roc_auc_score(test_labels, test_results)
