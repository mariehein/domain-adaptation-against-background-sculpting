import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from icecream import ic


def shuffle_XY(X, Y):
    seed_int = np.random.randint(300)
    np.random.seed(seed_int)
    np.random.shuffle(X)
    np.random.seed(seed_int)
    np.random.shuffle(Y)
    return X, Y


class no_logit_norm:
    def __init__(self, array):
        self.mean = np.mean(array, axis=0)
        self.std = np.std(array, axis=0)

    def forward(self, array0):
        return (np.copy(array0)-self.mean)/self.std, np.ones(len(array0), dtype=bool)

    def inverse(self, array0):
        return np.copy(array0)*self.std+self.mean


def make_features_baseline(features, label_arr, m2=False):
    E_part = np.sqrt(features[:, 0]**2+features[:, 1]**2+features[:, 2]**2+features[:, 3]**2) + \
        np.sqrt(features[:, 7]**2+features[:, 8]**2 +
                features[:, 9]**2+features[:, 10]**2)
    p_part2 = (features[:, 0]+features[:, 7])**2+(features[:, 1] +
                                                  features[:, 8])**2+(features[:, 2]+features[:, 9])**2
    m_jj = np.sqrt(E_part**2-p_part2)
    ind = np.array(features[:, 10] > features[:, 3]).astype(int)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
        if m2:
            feat1 = np.array([m_jj*1e-3, features[:, 3]*1e-3, features[:, 10]*1e-3, features[:, 5]/features[:, 4], features[:, 12]/features[:, 11], features[:, 6]/features[:, 5], features[:, 13]/features[:, 12], label_arr])
            feat2 = np.array([m_jj*1e-3, features[:, 10]*1e-3, features[:, 3]*1e-3, features[:, 12]/features[:, 11], features[:, 5]/features[:, 4], features[:, 13]/features[:, 12], features[:, 6]/features[:, 5], label_arr])
        else:
            feat1 = np.array([m_jj*1e-3, features[:, 3]*1e-3, (features[:, 10]-features[:, 3])*1e-3, features[:, 5]/features[:, 4], features[:, 12]/features[:, 11], features[:, 6]/features[:, 5], features[:, 13]/features[:, 12], label_arr])
            feat2 = np.array([m_jj*1e-3, features[:, 10]*1e-3, (features[:, 3]-features[:, 10])*1e-3, features[:, 12]/features[:, 11], features[:, 5]/features[:, 4], features[:, 13]/features[:, 12], features[:, 6]/features[:, 5], label_arr])
    return np.nan_to_num(feat1*ind+feat2*(np.ones(len(ind))-ind)).T


def file_loading(filename, labels=True, signal=0, stop=None, shift_masses=False):
    ic("File Loading", filename)
    if stop is not None:
        pandas_file = pd.read_hdf(filename, stop=stop)
    else:
        pandas_file = pd.read_hdf(filename)
    if labels:
        label_arr = np.array(pandas_file['label'], dtype=float)
    else:
        label_arr = np.ones((len(pandas_file['pxj1'])), dtype=float)*signal
    features = np.array(pandas_file[['pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1', 'tau2j1',
                        'tau3j1', 'pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2', 'tau2j2', 'tau3j2']], dtype=float)
    features = make_features_baseline(features, label_arr)
    if shift_masses:
        features[:, 1:3] += 0.1*features[:, :1]
    del pandas_file
    return features


def DR(filename, labels=True, stop=None):
    """
    Calculate DeltaR if args.include_DeltaR is true
    """
    if labels:
        features = np.array(pd.read_hdf(filename, stop=stop)[['pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1', 'tau2j1', 'tau3j1', 'pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2', 'tau2j2', 'tau3j2']], dtype=float)
    else:
        features = np.array(pd.read_hdf(filename, stop=stop)[['pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1', 'tau2j1', 'tau3j1', 'pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2', 'tau2j2', 'tau3j2']], dtype=float)
        features = np.concatenate((features, np.zeros((len(features), 1))), axis=1)
    Dphi = np.arccos((features[:, 0]*features[:, 7]+features[:, 1]*features[:, 8])/(np.sqrt(features[:, 1]**2+features[:, 0]**2)*np.sqrt(features[:, 7]**2+features[:, 8]**2)))
    eta1 = np.arcsinh(features[:, 2]/np.sqrt(features[:, 1]**2 + features[:, 0]**2))
    eta2 = np.arcsinh(features[:, 9]/np.sqrt(features[:, 7]**2 + features[:, 8]**2))
    DR = np.sqrt((Dphi)**2 + (eta1-eta2)**2)
    return DR


def classifier_data_prep(args, samples=None, run=None):
    data = file_loading(args.data_file, shift_masses=args.shift_masses)
    extra_bkg = file_loading(args.extrabkg_file, labels=False, shift_masses=args.shift_masses)

    if args.signal_file is not None:
        data_signal = file_loading(
            args.signal_file, labels=False, signal=1, shift_masses=args.shift_masses)

    sculpting = file_loading(args.sculpting_file, labels=False, stop=1000000, shift_masses=args.shift_masses)

    if args.signal_file is not None:
        sig = data_signal
    else:
        sig = data[data[:, -1] == 1]

    if args.include_DeltaR:
        data_DR = DR(args.data_file)
        data = np.concatenate((data[:, :args.inputs], np.array([data_DR]).T, data[:, args.inputs:]), axis=1)
        extra_bkg_DR = DR(args.extrabkg_file)
        extra_bkg = np.concatenate((extra_bkg[:, :args.inputs], np.array([extra_bkg_DR]).T, extra_bkg[:, args.inputs:]), axis=1)

        if args.signal_file is not None:
            sig_DR = DR(args.signal_file, labels=False)
            sig = np.concatenate((sig[:, :args.inputs], np.array([sig_DR]).T, sig[:, args.inputs:]), axis=1)
        else:
            sig = data[data[:, -1] == 1] 
        
        sculpting_DR = DR(args.sculpting_file, stop=1000000)
        sculpting = np.concatenate((sculpting[:, :args.inputs], np.array([sculpting_DR]).T, sculpting[:, args.inputs:]), axis=1)

    bkg = data[data[:, -1] == 0]

    print(len(bkg), len(sig))

    n_sig = args.signal_number

    if args.randomize_signal is not None:
        np.random.seed(int(args.signal_number)+int(args.randomize_signal))
        np.random.shuffle(sig)

    data_all = np.concatenate((bkg, sig[:n_sig]), axis=0)
    np.random.seed(int(args.set_seed))
    np.random.shuffle(data_all)
    extra_sig = sig[n_sig:]
    innersig_mask = (extra_sig[:, 0] > args.minmass) & (
        extra_sig[:, 0] < args.maxmass)
    inner_extra_sig = extra_sig[innersig_mask]

    innermask = (data_all[:, 0] > args.minmass) & (
        data_all[:, 0] < args.maxmass)
    innerdata = data_all[innermask]
    outerdata = data_all[~innermask]

    if args.density_estimation:
        np.save(args.directory+"innerdata.npy", innerdata)
        np.save(args.directory+"outerdata.npy", outerdata)
        return innerdata[:, :args.inputs+1], outerdata[:, :args.inputs+1]

    if args.mode == "cwola":
        mask = (outerdata[:, 0] > args.minmass -
                args.ssb_width) & (outerdata[:, 0] < args.maxmass+args.ssb_width)
        samples_train = outerdata[mask]
    elif args.mode == "cathode":
        if args.samples_file is None:
            raise ValueError("Samples file can not be None for cathode")
        samples_train = np.load(args.samples_file)[:int(
            len(innerdata)*args.oversampling_factor)]
        samples_train = np.concatenate(
            (samples_train, np.zeros((len(samples_train), 1))), axis=1)

    extrabkg1 = extra_bkg[:312858]
    extrabkg2 = extra_bkg[312858:]

    if args.mode == "IAD":
        samples_train = extrabkg1[40000:]

    if args.mode == "lacathode":
        import DE_CFM_model_utils as DE
        import torch
        from DE_CFM_utils import LogitScaler, train_test_split, make_pipeline

        train_data, _ = train_test_split(outerdata, test_size=0.5, shuffle=True, random_state=args.set_seed)
        scaler = make_pipeline(LogitScaler(), StandardScaler())
        scaler.fit(train_data[:,:args.inputs+1])
        del train_data

        min_epoch = np.load(args.DE_direc+"run"+str(args.set_seed)+"/val_logprob_epoch.npy")[np.argmin(np.load(args.DE_direc+"run"+str(args.set_seed)+"/val_logprob.npy"))]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        DE_model = DE.Conditional_ResNet_time_embed(frequencies=args.frequencies, 
                                    context_features=1, 
                                    input_dim=args.inputs, device=device,
                                    hidden_dim=args.hidden_dim, num_blocks=args.num_blocks, 
                                    use_batch_norm=True, 
                                    dropout_probability=0.2,
                                    non_linear_context=args.non_linear_context)
        DE_model.load_state_dict(torch.load(args.DE_direc+"run"+str(args.set_seed)+"/model_epoch_{}.pth".format(min_epoch), map_location=device))

        def prep_data(data, scaler, DE_model, ydata=None):
            data = scaler.transform(data)
            inds = np.invert(np.max(np.isnan(data), axis=1))
            data = data[inds]
            if ydata is not None:
                ydata = ydata[inds]
            print(sum(np.isnan(data)))
            data = torch.from_numpy(data.astype('float32')).to(device)
            #data = DE_model.forward(torch.from_numpy(data[:,1:args.inputs+1].astype('float32')).to(device)
            #                        , context=torch.from_numpy(data[:,:1].astype('float32')).to(device)).cpu().detach().numpy()
            data = DE.sample(DE_model, data[:,1:args.inputs+1], data[:,:1], start=1.0, end=0.0).cpu().detach().numpy()
            if ydata is not None:
                return data, ydata
            return data

        innerdata = prep_data(innerdata[:, :args.inputs+1], scaler, DE_model)
        print(innerdata.shape)
        np.save(args.directory / "innerdata_DE.npy", innerdata)
        X_train = np.concatenate((innerdata, np.random.normal(0,1,(len(innerdata)*args.oversampling_factor, args.inputs))), axis=0)
        Y_train = np.append(np.ones(len(innerdata)), np.zeros(len(X_train)-len(innerdata)))

        X_test = np.concatenate((extrabkg2,inner_extra_sig[:20000],extrabkg1[:40000]))
        Y_test = X_test[:,-1]
        X_test, Y_test = prep_data(X_test[:, :args.inputs+1], scaler, DE_model, ydata=Y_test)
        print(X_test.shape)

        sculpting_test_set, sculpting_masses = prep_data(sculpting[:,:args.inputs+1], scaler, DE_model, ydata=sculpting[:,0])
        if args.signal_number ==0:
            if run is not None:
                np.save(args.directory / "sculpting_masses_run{}.npy".format(run), sculpting_masses)
            else:            
                np.save(args.directory / "sculpting_masses.npy", sculpting_masses)
        print(sculpting_test_set.shape)
        
        X_domain = None
        Y_domain = None

        return X_train, Y_train, X_domain, Y_domain, X_test, Y_test, sculpting_test_set


    if args.mode == "supervised":
        sig_train = innerdata[:120000]
        sig_train = sig_train[sig_train[:, -1] == 1]
        X_train = np.concatenate((samples_train, sig_train), axis=0)
        Y_train = X_train[:, -1]
        X_train = X_train[:, 1:args.inputs+1]
    else:
        X_train = np.concatenate((innerdata[:120000, 1:args.inputs+1], samples_train[:, 1:args.inputs+1]), axis=0)
        Y_train = np.concatenate(
            (np.ones(len(X_train)-len(samples_train)), np.zeros(len(samples_train))), axis=0)

    X_train, Y_train = shuffle_XY(X_train, Y_train)

    # domain adaptation data: SB data, predict mJJ
    X_domain = outerdata[:, 1:args.inputs+1]
    if args.domain_classifier:
        Y_domain = np.zeros(len(X_domain))
        Y_domain[outerdata[:, 0] > args.maxmass] = 1
    else:
        Y_domain = outerdata[:, 0]
    

    X_test = np.concatenate((extrabkg2, inner_extra_sig[:20000], extrabkg1[:40000]))
    Y_test = X_test[:, -1]
    X_test = X_test[:, 1:args.inputs+1]

    sculpting_test_set = sculpting[:, 1:args.inputs+1]
    if args.signal_number == 0:
        np.save(str(args.directory / "sculpting_masses.npy"), sculpting[:, 0])

    if args.cl_norm:
        normalisation = no_logit_norm(X_train)
        X_train, _ = normalisation.forward(X_train)
        X_test, _ = normalisation.forward(X_test)
        sculpting_test_set, _ = normalisation.forward(sculpting_test_set)
        X_domain, _ = normalisation.forward(X_domain)
        if not args.domain_classifier:
            scaler = StandardScaler()
            Y_domain = scaler.fit_transform(Y_domain.reshape(-1, 1))[:, 0]

    print("Train set: ", len(X_train), "; Test set: ", len(X_test))

    return X_train, Y_train, X_domain, Y_domain, X_test, Y_test, sculpting_test_set
