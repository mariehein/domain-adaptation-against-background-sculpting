import torch
from torchdyn.core import NeuralODE
from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *
import time
from torch import Tensor
from zuko.utils import odeint
import torch.nn as nn

from torch.nn import init
from torch.nn import functional as F

from torch.distributions import Normal
import numpy as np
import torch


"""
Code from https://github.com/rd804/cut_and_count_FM
"""

def log_normal(x,mu=0.0,sigma=1.0):
    return Normal(mu, sigma).log_prob(x).sum(-1)

class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model: nn.Module, context: None = None):
        super().__init__()
        self.model = model
        self.context = context

    def forward(self, t, x_, *args, **kwargs):
        x = x_[:, :-1]
        identity = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
        identity = identity.expand(*x.shape, x.shape[-1]).movedim(-1, 0)

      #  print(t.shape)
        with torch.enable_grad():
            x = x.requires_grad_()
            self.model.eval()
            x_input = torch.cat([x, t.repeat(x.shape[0])[:, None]], 1)
            v_input = self.model(x_input, context=self.context)

            jacobian = torch.autograd.grad(v_input, x, identity, 
                        create_graph=True, is_grads_batched=True)[0]
                # calculate the trace of the jacobian
            trace = torch.einsum("i...i", jacobian)

        output = torch.cat([v_input, trace.reshape(-1,1) ], dim=1)
        
        return output

def log_prob_torchdyn(model, x, device,
                      start = 1.0, end=0.0, intervals=2,
                      train=False):
    ode_model = torch_wrapper(model, context=x[:,0].reshape(-1,1))
    ladj = torch.zeros_like(x[..., 0]).reshape(-1,1)
# print(ladj.shape)
    x_input = torch.cat([x[:,1:], ladj], dim=1)
#   print(x_input.shape)
    node = NeuralODE(ode_model, solver="dopri5", 
         sensitivity="autograd", atol=1e-4, rtol=1e-4).to(device)

    t_span = torch.linspace(start,end, intervals)
    t_eval, trajectory = node(x_input, t_span)
    if not train:
        trajectory = trajectory.detach().cpu()

    latent_space = trajectory[-1,:,:-1]
    log_probability = log_normal(latent_space) + trajectory[-1,:,-1]

    return trajectory, log_probability


def sample_conditional_pt(x0, x1, t, sigma):
    """
    Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    xt : Tensor, shape (bs, *dim)

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    #t = t.reshape(-1, *([1] * (x0.dim() - 1)))
    #xt = t * x1 + (1 + (sigma-1) * t) * x0
    mu_t = t * x1
   # epsilon = torch.randn_like(x0)
    sigma_t = 1-(1-sigma)*t

    return mu_t + sigma_t * x0

def compute_conditional_vector_field(x0, x1, sigma):
    """
    Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch

    Returns
    -------
    ut : conditional vector field ut(x1|x0) = x1 - x0

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    #return x1 - x0
    return x1 - (1-sigma) * x0


def create_loader(data, shuffle=True):
    if shuffle:
        shuffled_indices = torch.randperm(data.shape[0])
        data_shuffled = data[shuffled_indices]
        
        return data_shuffled


class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(
        self,
        features,
        context_features,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
        non_linear_context=False
    ):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)]
            )
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
            # self.context_layer = nn.Sequential(nn.Linear(context_features, features),
            #                                     nn.BatchNorm1d(features, eps=1e-3),
            #                                     nn.ReLU(),
            #                                     nn.Dropout(p=dropout_probability),
            #                                     nn.Linear(features,features))
                                               
        self.linear_layers = nn.ModuleList(
            [nn.Linear(features, features) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)
        self.non_linear_context = non_linear_context

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            context = self.context_layer(context)
            if self.non_linear_context:
                context = self.activation(context)
        #    context = self.activation(context)
            temps = F.glu(torch.cat((temps, context), dim=1), dim=1)

        return inputs + temps

class ResidualNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        non_linear_context=False
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        self.non_linear_context = non_linear_context

        if context_features is not None:
            # self.initial_layer = nn.Sequential(nn.Linear(in_features + context_features, hidden_features),
            #                                     nn.BatchNorm1d(hidden_features, eps=1e-3),
            #                                     nn.ReLU(),
            #                                     nn.Dropout(p=dropout_probability),
            #                                     nn.Linear(hidden_features,hidden_features))
           self.initial_layer = nn.Linear(
               in_features + context_features, hidden_features
           )
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=hidden_features,
                    context_features=context_features,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                    non_linear_context=non_linear_context
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_features, out_features)
        self.activation = activation

    def forward(self, inputs, context=None):
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
            if self.non_linear_context:
                temps = self.activation(temps)
        #    temps = self.activation(temps)
        for block in self.blocks:
            temps = block(temps, context=context)
        outputs = self.final_layer(temps)
        return outputs

class Conditional_ResNet_time_embed(nn.Module):
    def __init__(self, frequencies: int=3, context_features: int=1,
                 input_dim: int=2, device: torch.device=torch.device('cpu'),
                 hidden_dim: int=256, num_blocks: int=2,
                 use_batch_norm: bool=True, dropout_probability: float=0.2, time_embed=None,
                 non_linear_context=False):
        super().__init__()

        if time_embed is None:
            freq_dim = frequencies
            self.frequencies = torch.arange(1,frequencies+1,1).float()*torch.pi
            self.frequencies = self.frequencies.to(device)
            self.context_dim = 2*freq_dim + context_features
        else:
            self.context_dim = time_embed.dim + context_features
        
        self.time_embed = time_embed
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.model = ResidualNet(in_features=input_dim, out_features=input_dim, 
                    context_features=self.context_dim, hidden_features=hidden_dim, 
                    num_blocks=num_blocks,dropout_probability=dropout_probability,
                     use_batch_norm=use_batch_norm,
                     non_linear_context=non_linear_context).to(device)


    def forward(self, x, context=None):
        _t = x[:,-1]
        _x = x[:,:-1]

        #context = context.flatten()

        if self.time_embed is not None:
            t = self.time_embed(_t)
        else:
            t = self.frequencies * _t[...,None]
            t = torch.cat((t.cos(), t.sin()), dim=-1).to(x.device)
           # context = self.frequencies * context[...,None]
           # context = torch.cat((context.cos(), context.sin()), dim=-1).to(x.device)

        if context is not None:
            context = torch.cat((t, context), dim=-1)
        else:
            context = t

        return self.model(_x, context=context)


def train_flow(traindata, model, valdata=None, optimizer=None, 
               num_epochs=2, batch_size=5,
               device=torch.device('cuda:0'),sigma_fm=0.001, 
               save_model = False, model_path=None, interval=200,
    ot=False, compute_log_likelihood=False,
    likelihood_interval=20, likelihood_start=300,
    scheduler=False,
    early_stop_patience=50):

    if compute_log_likelihood:
        print(f'valdata is {valdata.shape}')
    if ot:
        ot_sampler = OTPlanSampler(method="exact")
    start = time.time()
    losses = []
    logprob_list = []
    logprob_epoch = []
    early_stop_counter = 0

    best_val_log_prob=10000000
    for epoch in range(num_epochs):
        data_ = create_loader(traindata, shuffle=True)
        running_loss = 0
        if epoch % 1 == 0:
            print('epoch', epoch)

        for i in range(len(data_)//batch_size+1):
            optimizer.zero_grad()
            
            x1 = data_[i*batch_size:(i+1)*batch_size].to(device)

            context = x1[:,0].reshape(-1,1)
            x1 = x1[:,1:]

            x0 = torch.randn_like(x1).to(device)
            t = torch.rand_like(x0[:, 0].reshape(-1,1)).to(device)

            xt = sample_conditional_pt(x0, x1, t, sigma_fm)
            if ot:
                pi = ot_sampler.get_map(x0, x1)
                i, j = ot_sampler.sample_map(pi, x0.shape[0], replace=False)
                xt = x1[j]
                t = t[j]
                x0 = x0[i]
            

            ut = compute_conditional_vector_field(x0, x1, sigma_fm)

            vt = model(torch.cat([xt, t], dim=-1),context=context)
            loss = torch.mean((vt - ut) ** 2)

            running_loss += loss.item()
           
            loss.backward()
            optimizer.step()

        
        total_loss = running_loss/(len(data_)//batch_size+1)
        print(total_loss)
        losses.append(total_loss)
    
        # if total_loss is lower than all losses in best_loss_array, save model
        # if save_model:
        #     if num_epochs-epoch < 11: 
        #         print('saving model at epoch', epoch)
        #         torch.save(model.state_dict(), f'{model_path}/model_epoch_{epoch}.pth')

        if compute_log_likelihood:
            if (likelihood_start < epoch+1) and (epoch % likelihood_interval == 0): 
                log_prob_ = compute_log_prob(model, valdata[:,1:], 
                                    valdata[:,0].reshape(-1,1), 
                                    batch_size=50000,
                                    device=device,
                                    method='torchdyn')
                mean_log_prob = np.mean(-log_prob_)
                #if wandb_log:
                #    wandb.log({'logprob': mean_log_prob, 'lr': optimizer.param_groups[0]['lr']})
                if mean_log_prob < best_val_log_prob:
                    best_val_log_prob = mean_log_prob
                    early_stop_counter = 0
                else:
                    early_stop_counter +=1

                    #torch.save(model.state_dict(), f'{model_path}/model_epoch_{epoch}.pth')
                if early_stop_counter > early_stop_patience:
                    print('Early stopping at epoch', epoch)
                    break


                if scheduler:
                    scheduler.step(mean_log_prob)
                logprob_list.append(mean_log_prob)
                logprob_epoch.append(epoch)
                torch.save(model.state_dict(), f'{model_path}/model_epoch_{epoch}.pth')

                print(f'saving logprob for epoch {epoch}')

    end = time.time()
    print('Time taken: ', end - start)
    if not compute_log_likelihood:
        return losses
    else:
        return losses, logprob_list, logprob_epoch

def sample(model: torch.nn.Module , x: Tensor, context: Tensor, 
            start:float=0.0, end:int=1.0) -> Tensor:

        def augmented(t: Tensor, x: Tensor) -> Tensor:
            model.eval()
            with torch.no_grad():
               # context = data[:,-1].reshape(-1,1)
               # print
                t_array = torch.ones(x.shape[0], 1).to(x.device) * t
                input_to_model = torch.cat([x,t_array], dim=-1)
                vt = model(input_to_model, context=context)

            return vt   

        z = odeint(augmented, x, start, end, phi=model.parameters())

        return z

def compute_log_prob(model, data, context, batch_size=500,
                     device=torch.device('cuda:0'),
                     method='zuko'):
    data = torch.as_tensor(data, device=device)
    context = torch.as_tensor(context, device=device)

    log_likelihood = []
    model.eval()
    start_time = time.time()
    batch_previous = start_time
    with torch.no_grad():
        for i in range(len(data)//batch_size + 1):
            print(f'Computing log likelihood for batch {i}')
            test_input = data[i*batch_size:(i+1)*batch_size].to(device)
            test_context = context[i*batch_size:(i+1)*batch_size].to(device)
            test_data = torch.cat([test_context,test_input], dim=-1)
        # print(test_input.shape)
            if method=='zuko':
                z, log_likelihood_ = log_prob(model, test_input, 
                                            test_context, start=1.0,end=0.0)
            elif method=='torchdyn':
                _, log_likelihood_ = log_prob_torchdyn(model, test_data,
                                                    device=device)
            
            log_likelihood.append(log_likelihood_.detach().cpu().numpy())
            batch_current = time.time()
            print("Time taken:", batch_current-batch_previous)
            batch_previous = batch_current
    print("Time taken whole:", time.time()-start_time)

        # print(test_input.shape)
        #log_likelihood.append(log_prob(model, test_input,start=1,end=0).detach().cpu().numpy())

    log_likelihood = np.concatenate(log_likelihood)

    return log_likelihood
