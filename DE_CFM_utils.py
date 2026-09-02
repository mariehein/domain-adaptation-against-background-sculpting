import numpy as np
import os
import yaml
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import rv_histogram
from sklearn.preprocessing import MinMaxScaler
from numbers import Real
from scipy.special import logit, expit
import torch
from DE_CFM_model_utils import Conditional_ResNet_time_embed, train_flow, sample

"""
Code based on https://github.com/rd804/cut_and_count_FM
"""

class LogitScaler(MinMaxScaler):
    """Preprocessing scaler that performs a logit transformation on top
    of the sklean MinMaxScaler. It scales to a range [0+epsilon, 1-epsilon]
    before applying the logit. Setting a small finitie epsilon avoids
    features being mapped to exactly 0 and 1 before the logit is applied.
    If the logit does encounter values beyond (0, 1), it outputs nan for
    these values.
    """

    _parameter_constraints: dict = {
        "epsilon": [Real],
        "copy": ["boolean"],
        "clip": ["boolean"],
    }

    def __init__(self, epsilon=0, copy=True, clip=False):
        self.epsilon = epsilon
        self.copy = copy
        self.clip = clip
        super().__init__(feature_range=(0+epsilon, 1-epsilon),
                         copy=copy, clip=clip)

    def fit(self, X, y=None):
        super().fit(X, y)
        return self

    def transform(self, X):
        z = logit(super().transform(X))
        z[np.isinf(z)] = np.nan
        return z

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return super().inverse_transform(expit(X))

    def jacobian_determinant(self, X):
        z = super().transform(X)
        return np.prod(z * (1 - z), axis=1, keepdims=True
                       ) / np.prod(self.scale_)

    def log_jacobian_determinant(self, X):
        z = super().transform(X)
        return np.sum(np.log(z * (1 - z)), axis=1, keepdims=True
                      ) - np.sum(np.log(self.scale_))

    def inverse_jacobian_determinant(self, X):
        z = expit(X)
        return np.prod(z * (1 - z), axis=1, keepdims=True
                       ) * np.prod(self.scale_)

    def inverse_log_jacobian_determinant(self, X):
        z = expit(X)
        return np.sum(np.log(z * (1 - z)), axis=1, keepdims=True
                      ) + np.sum(np.log(self.scale_))

def run_DE(args, innerdata, outerdata, direc_run): 
    if not os.path.exists(direc_run):
        os.makedirs(direc_run)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    n_features = args.inputs
    print('n_features', n_features)

    train_data, val_data = train_test_split(outerdata, test_size=0.5, shuffle=True, random_state=args.set_seed)
    scaler = make_pipeline(LogitScaler(), StandardScaler())
    x_train = scaler.fit_transform(train_data)
    x_train = x_train[np.invert(np.max(np.isnan(x_train), axis=1))]
    x_val = scaler.transform(val_data)
    x_val = x_val[np.invert(np.max(np.isnan(x_val), axis=1))]
    m_inner = scaler.transform(innerdata)[:,0]

    traintensor = torch.from_numpy(x_train.astype('float32')).to(device)
    valtensor = torch.from_numpy(x_val.astype('float32')).to(device)
    print('X_train shape', traintensor.shape)
    print('X_val shape', valtensor.shape)

    for i in range(1):
        print(f'Ensemble {i}')

        model = Conditional_ResNet_time_embed(frequencies=args.frequencies, 
                                    context_features=1, 
                                    input_dim=n_features, device=device,
                                    hidden_dim=args.hidden_dim, num_blocks=args.num_blocks, 
                                    use_batch_norm=True, 
                                    dropout_probability=0.2,
                                    non_linear_context=args.non_linear_context)

                                
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, verbose=True)
        trainloss, logprob_list, logprob_epoch = train_flow(traintensor, 
                model, valdata=valtensor ,optimizer=optimizer,
                num_epochs=args.epochs, batch_size=args.batch_size,
                device=device, sigma_fm=0.001,
                save_model=True, model_path=f'{direc_run}',
                compute_log_likelihood=True,
                likelihood_interval=5, likelihood_start=100,
                early_stop_patience=20,
                scheduler=scheduler)
        
    log_prob_mean = np.array(logprob_list)
    log_prob_mean_sorted = np.argsort(log_prob_mean)
    logprob_epoch = np.array(logprob_epoch)
    lowest_epochs = logprob_epoch[log_prob_mean_sorted[:10].tolist()]

    np.save(f'{direc_run}val_logprob.npy', log_prob_mean)
    np.save(f'{direc_run}val_logprob_epoch.npy', logprob_epoch)

    log_prob_mean_sorted = np.argsort(log_prob_mean)
    logprob_epoch = np.array(logprob_epoch)
    lowest_epochs = logprob_epoch[log_prob_mean_sorted[:10].tolist()] 

    print('lowest epochs', lowest_epochs)

    SR_mass = m_inner
    SR_hist = np.histogram(SR_mass, bins=60, density=True)
    SR_density = rv_histogram(SR_hist)

    CR_mass = np.append(x_train[:,0], x_val[:,0])
    CR_hist = np.histogram(CR_mass, bins=100, density=True)
    CR_density = rv_histogram(CR_hist)

    noise_SR = torch.randn(args.N_samples, n_features).to(device).float()
    mass_samples_SR = SR_density.rvs(size=len(noise_SR))

    noise_CR = torch.randn(args.N_samples, n_features).to(device).float()
    mass_samples_CR = CR_density.rvs(size=len(noise_CR))

    mass_samples_CR = mass_samples_CR.reshape(-1,1)
    mass_samples_SR = mass_samples_SR.reshape(-1,1)
    mass_samples_CR = torch.from_numpy(mass_samples_CR).to(device).float()
    mass_samples_SR = torch.from_numpy(mass_samples_SR).to(device).float()

    mini_batch_length = len(noise_SR)//10

    ensembled_samples_SR = []
    ensembled_mass_SR = []

    ensembled_samples_CR = []
    ensembled_mass_CR = []
    for i,epoch in enumerate(lowest_epochs):
        print("Doing epoch "+str(epoch))
        model.load_state_dict(torch.load(f'{direc_run}model_epoch_{epoch}.pth', map_location=device))
        model.to(device)
        model.eval()
        noise_batch = noise_SR[i*mini_batch_length:(i+1)*mini_batch_length]
        mass_batch = mass_samples_SR[i*mini_batch_length:(i+1)*mini_batch_length] 
        samples = sample(model, noise_batch, mass_batch, start=0.0, end=1.0)
        ensembled_samples_SR.append(samples)
        ensembled_mass_SR.append(mass_batch)

        noise_batch = noise_CR[i*mini_batch_length:(i+1)*mini_batch_length]
        mass_batch = mass_samples_CR[i*mini_batch_length:(i+1)*mini_batch_length]
        samples = sample(model, noise_batch, mass_batch, start=0.0, end=1.0)
        ensembled_samples_CR.append(samples)
        ensembled_mass_CR.append(mass_batch)


    print('deleting models')

    delete_paths = [f'{direc_run}model_epoch_{epoch}.pth' for epoch in logprob_epoch if epoch not in lowest_epochs]

    for path in delete_paths:
        os.remove(path)

    mass_SR = torch.cat(ensembled_mass_SR, dim=0)
    samples_SR = torch.cat(ensembled_samples_SR, dim=0)
    samples_SR = torch.concat([mass_SR, samples_SR], axis=1)
    samples_inverse_SR = scaler.inverse_transform(samples_SR.cpu().detach().numpy())

    mass_CR = torch.cat(ensembled_mass_CR, dim=0)
    samples_CR = torch.cat(ensembled_samples_CR, dim=0)
    samples_CR = torch.concat([mass_CR, samples_CR], axis=1)
    samples_inverse_CR = scaler.inverse_transform(samples_CR.cpu().detach().numpy())

    np.save(f'{direc_run}samples_inner.npy', samples_inverse_SR)
    np.save(f'{direc_run}samples_outer.npy', samples_inverse_CR)