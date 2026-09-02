import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import metric_utils as metrics

from os import makedirs
from os.path import join
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tqdm import tqdm, trange
from icecream import ic

try:
    from torchsummary import summary
except:
    pass


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)


class NeuralNetwork(nn.Module):
    """A PyTorch module implementing a simple feed-forward neural network.
    """

    def __init__(self, layers=[64, 64, 64], n_inputs=4, dropout=None, gr_alpha=1.0):
        super().__init__()
        self.layers = []
        for nodes in layers:
            self.layers.append(nn.Linear(n_inputs, nodes))
            self.layers.append(nn.ReLU())
            # if dropout is not None:
            #     self.layers.append(nn.Dropout(dropout))
            n_inputs = nodes
        self.layers.append(nn.Linear(n_inputs, 1))
        self.layers.append(nn.Sigmoid())
        self.model_stack = nn.Sequential(*self.layers)

    def forward(self, X):
        return self.model_stack(X)

class FixedWeightLoss(nn.Module):
    def __init__(self, model, lambda_domain=0.01):
        super().__init__()
        # assume model has attributes: log_var_class, log_var_reg
        self.model = model
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.mse = nn.MSELoss()
        self.lambda_domain = lambda_domain

    def forward(self, class_logits, class_targets, reg_out, reg_targets, weight=None):
        class_targets = class_targets.view_as(class_logits)
        loss_per = self.bce(class_logits, class_targets)  # (B,1)

        if weight is not None:
            if weight.ndim == 1:
                weight = weight.view(-1, 1)
            loss_class = (loss_per * weight).mean()
        else:
            loss_class = loss_per.mean()

        loss_reg   = self.mse(reg_out, reg_targets)

        # Kendall–Gal homoscedastic combination
        # L = exp(-s_c)*L_class + s_c + exp(-s_r)*L_reg + s_r
        loss = loss_class + self.lambda_domain * loss_reg

        return loss, {
            "loss_class": loss_class.detach(),
            "loss_reg":   loss_reg.detach(),
            "sigma_class": 0,
            "sigma_reg":   0,
            "loss": loss.detach()
        }


class HomoscedasticMTLLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        # assume model has attributes: log_var_class, log_var_reg
        self.model = model
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.mse = nn.MSELoss()

    def forward(self, class_logits, class_targets, reg_out, reg_targets, weight=None):
        class_targets = class_targets.view_as(class_logits)
        loss_per = self.bce(class_logits, class_targets)  # (B,1)

        if weight is not None:
            if weight.ndim == 1:
                weight = weight.view(-1, 1)
            loss_class = (loss_per * weight).mean()
        else:
            loss_class = loss_per.mean()

        loss_reg   = self.mse(reg_out, reg_targets)

        # Log-variances
        s_c = self.model.log_var_class
        s_r = self.model.log_var_reg

        # Kendall–Gal homoscedastic combination
        # L = exp(-s_c)*L_class + s_c + exp(-s_r)*L_reg + s_r
        loss = torch.exp(-s_c) * loss_class + s_c \
             + 0.5 * torch.exp(-s_r) * loss_reg   + s_r

        return loss, {
            "loss_class": loss_class.detach(),
            "loss_reg":   loss_reg.detach(),
            "sigma_class": torch.exp(0.5 * s_c).detach(),
            "sigma_reg":   torch.exp(0.5 * s_r).detach(),
            "loss": loss.detach()
        }


class FixedWeightLoss_Class(nn.Module):
    def __init__(self, model, lambda_domain=0.01):
        super().__init__()
        # assume model has attributes: log_var_class, log_var_reg
        self.model = model
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.mse = nn.BCEWithLogitsLoss(reduction="none")
        self.lambda_domain = lambda_domain

    def forward(self, class_logits, class_targets, reg_out, reg_targets, weight=None, weight_reg=None):
        class_targets = class_targets.view_as(class_logits)
        loss_per = self.bce(class_logits, class_targets)  # (B,1)
        loss_per_reg = self.mse(reg_out, reg_targets)  # (B,1)

        if weight is not None:
            if weight.ndim == 1:
                weight = weight.view(-1, 1)
            loss_class = (loss_per * weight).mean()
        else:
            loss_class = loss_per.mean()

        if weight_reg is not None:
            if weight.ndim == 1:
                weight_reg = weight_reg.view(-1, 1)
            loss_reg = (loss_per_reg * weight_reg).mean()
        else:
            loss_reg = loss_per_reg.mean()

        # Kendall–Gal homoscedastic combination
        # L = exp(-s_c)*L_class + s_c + exp(-s_r)*L_reg + s_r
        loss = loss_class + self.lambda_domain * loss_reg

        return loss, {
            "loss_class": loss_class.detach(),
            "loss_reg":   loss_reg.detach(),
            "sigma_class": 0,
            "sigma_reg":  0,
            "loss": loss.detach()
        }

class HomoscedasticMTLLoss_Class(nn.Module):
    def __init__(self, model):
        super().__init__()
        # assume model has attributes: log_var_class, log_var_reg
        self.model = model
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.mse = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, class_logits, class_targets, reg_out, reg_targets, weight=None, weight_reg=None):
        class_targets = class_targets.view_as(class_logits)
        loss_per = self.bce(class_logits, class_targets)  # (B,1)
        loss_per_reg = self.mse(reg_out, reg_targets)  # (B,1)

        if weight is not None:
            if weight.ndim == 1:
                weight = weight.view(-1, 1)
            loss_class = (loss_per * weight).mean()
        else:
            loss_class = loss_per.mean()

        if weight_reg is not None:
            if weight.ndim == 1:
                weight_reg = weight_reg.view(-1, 1)
            loss_reg = (loss_per_reg * weight_reg).mean()
        else:
            loss_reg = loss_per_reg.mean()

        # Log-variances
        s_c = self.model.log_var_class
        s_r = self.model.log_var_reg

        # Kendall–Gal homoscedastic combination
        # L = exp(-s_c)*L_class + s_c + exp(-s_r)*L_reg + s_r
        loss = torch.exp(-s_c) * loss_class + s_c \
             + torch.exp(-s_r) * loss_reg   + s_r

        return loss, {
            "loss_class": loss_class.detach(),
            "loss_reg":   loss_reg.detach(),
            "sigma_class": torch.exp(0.5 * s_c).detach(),
            "sigma_reg":   torch.exp(0.5 * s_r).detach(),
            "loss": loss.detach()
        }


class NeuralNetwork_DA(nn.Module):
    def __init__(self, layers=[64, 64], n_inputs=4, dropout=None, gr_alpha=1.0):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(n_inputs, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )

        self.class_head = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
            # nn.Sigmoid()
        )

        self.gr = GradientReversalLayer(alpha=gr_alpha)

        self.log_var_class = nn.Parameter(torch.zeros(1))
        self.log_var_reg   = nn.Parameter(torch.zeros(1))

        self.reg_head = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, X):
        h = self.shared(X)
        class_out = self.class_head(h)
        reg_out = self.reg_head(self.gr(h))
        return class_out, reg_out


def get_model(use_domain_adaptation: bool, **kwargs) -> nn.Module:
    ic(use_domain_adaptation)
    if use_domain_adaptation:
        model = NeuralNetwork_DA(**kwargs)
    else:
        model = NeuralNetwork(**kwargs)
    return model


class NeuralNetworkClassifier(BaseEstimator):
    """Neural network classifier based on torch but wrapped such that it
    mimicks the scikit-learn API, using numpy arrays as inputs and outputs.

    Parameters
    ----------
    save_path : str, optional
        Path to save the model to. If None, no model is saved.
        If provided, the model will use the best checkpoint after training.
    load : bool, optional
        Whether to load the model from save_path.
    n_inputs : int, default=4
        Number of input features.
    layers : list, default=[64, 64, 64]
        List of integers, specifying the number of nodes in each layer.
    lr : float, default=0.001
        Learning rate during training.
    early_stopping : bool, default=False
        Whether to use early stopping. If set, the provided number of
        epochs will be treated as an upper limit.
    patience : int, default=10
        Number of epochs to wait for improvement before stopping, if early
        stopping is used.
    no_gpu : bool, default=False
        Turns off GPU usages. By default the GPU is used if available.
    val_split : float, default=0.2
        Fraction of the training set to use for validation. Only has an
        effect if no validation set is provided to the fit method.
    batch_size : int, default=128
        Batch size during training.
    epochs : int, default=100
        Number of epochs to train for. In case early stopping is used,
        this is treated as an upper limit. Then also None can be provided,
        in which case the training will continue until early stopping
        is triggered.
    use_class_weights : bool, default=True
        Whether to use class weights during training.
    verbose : bool, default=False
        Whether to print progress during training.
    use_domain_adaptation: bool, default=False
        Use Domain adaptation during the minimization process
    """

    def __init__(self, save_path=None, load=False, n_inputs=4,
                 layers=[64, 64, 64], lr=0.001, early_stopping=False,
                 patience=10, no_gpu=False, val_split=0.2, batch_size=128,
                 epochs=100, use_class_weights=True, dropout=None, weight_decay=0, momentum=0.9, domain_weight=None,
                 verbose=True, save_model=True, metric_tracking=True, use_domain_adaptation=False, domain_classifier=False):

        self.save_path = save_path
        if save_path is not None:
            self.clsf_model_path = join(save_path, "CLSF_models/")
        else:
            self.clsf_model_path = None
        self.load = load
        self.save_model = save_model

        self.n_inputs = n_inputs
        self.layers = layers
        self.lr = float(lr)
        self.no_gpu = no_gpu
        self.model = get_model(use_domain_adaptation=use_domain_adaptation,
                               layers=layers, n_inputs=n_inputs, dropout=dropout)
        # self.model = NeuralNetwork(layers, n_inputs=n_inputs, dropout=dropout)
        self.optimizer = optim.Adam(self.model.parameters(), betas=(
            momentum, 0.999), weight_decay=weight_decay, lr=float(lr))
        self.loss = F.binary_cross_entropy
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and not no_gpu else "cpu")
        ic(self.device)
        self.early_stopping = early_stopping
        self.patience = patience
        self.val_split = val_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_class_weights = use_class_weights
        self.verbose = verbose
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.momentum = momentum

        self.metric_tracking = metric_tracking
        self.use_domain_adaptation = use_domain_adaptation
        self.domain_classifier = domain_classifier
        self.model.to(self.device)

        if domain_weight is None:
            if domain_classifier:
                self.criterion = HomoscedasticMTLLoss_Class(self.model)
            else: 
                self.criterion = HomoscedasticMTLLoss(self.model)
        else:
            if domain_classifier:
                self.criterion = FixedWeightLoss_Class(self.model, lambda_domain=domain_weight)
            else: 
                self.criterion = FixedWeightLoss(self.model, lambda_domain=domain_weight)

        # defaulting to eval mode, switching to train mode in fit()
        self.model.eval()

        if load:
            self.load_best_model()

    def loss_fn(self, domain_outputs, domain_labels, class_outputs, class_labels):
        lambda_domain = 0.135  # This is coming from Vincent's bachelor thesi
        min_batch = min(domain_outputs.shape[0], domain_labels.shape[0],
                        class_outputs.shape[0], class_labels.shape[0])
        if not (domain_outputs.shape[0] == domain_labels.shape[0] ==
                class_outputs.shape[0] == class_labels.shape[0]):
            domain_outputs = domain_outputs[:min_batch]
            domain_labels = domain_labels[:min_batch]
            class_outputs = class_outputs[:min_batch]
            class_labels = class_labels[:min_batch]

        return F.mse_loss(domain_outputs, domain_labels), F.binary_cross_entropy(class_outputs, class_labels)
        # return lambda_domain * F.mse_loss(domain_outputs, domain_labels) + F.binary_cross_entropy(class_outputs, class_labels)

    def predict(self, X):
        """Predicts the class labels for the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.

        Returns
        -------
        prediction : numpy.ndarray
            Predicted class labels.
        """
        with torch.no_grad():
            self.model.eval()
            X = torch.from_numpy(X).type(torch.FloatTensor).to(self.device)
            #if self.use_domain_adaptation:
            #    X = X[:, 1:]
            prediction = self.model.forward(X)
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            prediction = prediction.detach().cpu().numpy()
        return prediction

    def fit(self, X, y, X_domain=None, y_domain=None,
            sample_weight=None, sample_weight_val=None):
        """Fits (trains) the model to the provided data.

        Parameters
        ----------
        X : numpy.ndarray
            Input data.
        y : numpy.ndarray
            Target data.
        X_val : numpy.ndarray, optional
            Validation input data.
        y_val : numpy.ndarray, optional
            Validation target data.
        sample_weight : numpy.ndarray, optional
            Sample weights for the training data.
        sample_weight_val : numpy.ndarray, optional
            Sample weights for the validation data.

        Returns
        -------
        self : object
            An instance of the classifier.
        """
        assert not (self.epochs is None and not self.early_stopping), (
            "A finite number of epochs must be set if early stopping"
            " is not used!")

        # allowing not to provide validation set, just for compatibility with
        # the sklearn API
        X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=self.val_split, shuffle=True)
        print("size ratio:", len(X_train)/len(X_val))

        if self.clsf_model_path is not None:
            makedirs(self.clsf_model_path, exist_ok=True)

        nan_mask = ~np.isnan(X_train).any(axis=1)
        X_train = X_train[nan_mask]
        y_train = y_train[nan_mask]

        nan_mask = ~np.isnan(X_val).any(axis=1)
        X_val = X_val[nan_mask]
        y_val = y_val[nan_mask]

        # deduce class weights for training and validation sets
        # (move outside class as in sklearn?)
        if self.use_class_weights:
            class_weights_train = class_weight.compute_class_weight(
                'balanced', classes=np.unique(y_train), y=y_train)
            class_weights_train = dict(enumerate(class_weights_train))

            class_weights_val = class_weight.compute_class_weight(
                'balanced', classes=np.unique(y_val), y=y_val)
            class_weights_val = dict(enumerate(class_weights_val))
        else:
            class_weights_train = None
            class_weights_val = None

        use_domain_adaptation = (X_domain is not None and y_domain is not None)

        # Handle domain adaptation error inputs
        if X_domain is None and y_domain is not None:
            raise ValueError("X_domain is None but y_domain is not None")
        elif X_domain is not None and y_domain is None:
            raise ValueError("y_domain is None but X_domain is not None")

        # build data loader out of numpy arrays
        train_loader = numpy_to_torch_loader(
            X_train, y_train,
            batch_size=self.batch_size, shuffle=True, device=self.device)
        val_loader = numpy_to_torch_loader(
            X_val, y_val,
            batch_size=self.batch_size, shuffle=True, device=self.device)

        if use_domain_adaptation:

            X_domain_train, X_domain_val, y_domain_train, y_domain_val = train_test_split(
                        X_domain, y_domain, test_size=self.val_split, shuffle=True)

            # deduce class weights for training and validation sets
            # (move outside class as in sklearn?)
            if self.use_class_weights:
                class_weights_domain_train = class_weight.compute_class_weight(
                    'balanced', classes=np.unique(y_domain_train), y=y_domain_train)
                class_weights_domain_train = dict(enumerate(class_weights_domain_train))

                class_weights_domain_val = class_weight.compute_class_weight(
                    'balanced', classes=np.unique(y_domain_val), y=y_domain_val)
                class_weights_domain_val = dict(enumerate(class_weights_domain_val))
            else:
                class_weights_domain_train = None
                class_weights_domain_val = None

        
            domain_train_loader = numpy_to_torch_loader(
                X_domain_train, y_domain_train, batch_size=self.batch_size, shuffle=True, device=self.device)
            domain_val_loader = numpy_to_torch_loader(
                X_domain_val, y_domain_val, batch_size=self.batch_size, shuffle=True, device=self.device)
    
        # training loop
        self.model.train()

        for epoch in (pbar0 := trange(self.epochs if self.epochs is not None else 10000, position=0)):
            # print('\nEpoch: {}'.format(epoch))
            # pbar = tqdm(total=len(train_loader.dataset))
            epoch_train_loss = 0.
            epoch_val_loss = 0.

            train_iter = iter(train_loader)

            if use_domain_adaptation:
                domain_iter = iter(domain_train_loader)

            # for i, batch in (pbar := tqdm(enumerate(train_loader), total=len(train_loader.dataset), position=1, leave=False)):
            for i in range(len(train_loader)):
            #for i in (pbar := trange(len(train_loader), position=1, leave=False)):
                batch = next(train_iter)

                if use_domain_adaptation:
                    domain_batch = next(domain_iter)

                batch_inputs, batch_labels = batch
                batch_inputs, batch_labels = (batch_inputs.to(self.device),
                                              batch_labels.to(self.device))

                # translating class weights to sample weights
                if class_weights_train is not None:
                    batch_weights = class_weight_to_sample_weight(batch_labels, class_weights_train)
                    batch_weights = batch_weights.type(torch.FloatTensor).to(self.device)

                    if use_domain_adaptation and self.domain_classifier:
                        batch_weights_domain = class_weight_to_sample_weight(domain_batch[1], class_weights_domain_train)
                else:
                    batch_weights = None
                    if use_domain_adaptation and self.domain_classifier:
                        batch_weights_domain = None

                self.optimizer.zero_grad()

                if use_domain_adaptation:
                    mjj_truth = domain_batch[1]
                    # mjj_truth = domain_batch[:, 0].unsqueeze(1)
                    batch_info = batch_inputs
                    class_pred, _ = self.model(batch_info)
                    _, mjj_pred = self.model(domain_batch[0])

                    # batch_loss_classification = self.loss(class_pred, batch_labels, weight=batch_weights)
                    # batch_loss_domain = F.mse_loss(mjj_pred, mjj_truth)
                    if self.domain_classifier:
                        batch_loss, all_losses = self.criterion(class_pred, batch_labels, mjj_pred, mjj_truth, 
                            weight=batch_weights, weight_reg=batch_weights_domain)
                    else:
                        batch_loss, all_losses = self.criterion(class_pred, batch_labels, mjj_pred, mjj_truth, weight=batch_weights)
                    # return loss, {
                    #     "loss_class": loss_class.detach(),
                    #     "loss_reg":   loss_reg.detach(),
                    #     "sigma_class": torch.exp(0.5 * s_c).detach(),
                    #     "sigma_reg":   torch.exp(0.5 * s_r).detach(),
                    #     "loss": loss.detach()
                    # }

                    # batch_loss = 0.135 * batch_loss_domain + batch_loss_classification
                    # def forward(self, class_logits, class_targets, reg_out, reg_targets):

                    # domain_inputs, domain_labels = domain_batch
                    # domain_inputs = domain_inputs.to(self.device)
                    # domain_labels = domain_labels.to(self.device)
                    # # Forward class and domain batches separately so each head
                    # # sees tensors with matching batch sizes.
                    # class_outputs, _ = self.model(batch_inputs)
                    # _, domain_outputs = self.model(domain_inputs)
                    # batch_loss = self.criterion(class_outputs, batch_labels, domain_outputs, domain_labels)
                    # batch_mse, batch_ce = self.loss_fn(domain_outputs,
                    #                                    domain_labels,
                    #                                    class_outputs, 
                    #                                    batch_labels)
                    
                    # batch_loss = 0.135 * batch_mse + batch_ce
                else:
                    batch_outputs = self.model(batch_inputs)
                    batch_loss = self.loss(batch_outputs, batch_labels, weight=batch_weights)

                # batch_outputs = self.model(batch_inputs)
                # batch_loss = self.loss(batch_outputs, batch_labels,
                #                        weight=batch_weights)
                batch_loss.backward()
                self.optimizer.step()
                epoch_train_loss += batch_loss.item()
                if self.verbose and i % 100 == 0:
                    # pbar.update(batch_inputs.size(0))
                    d = {"Train loss": f"{epoch_train_loss / (i+1):.6f}"}
                    
                    if use_domain_adaptation:
                        d["Class loss"] = f"{float(all_losses['loss_class']):.6f}"
                        d["Domain loss"] = f"{float(all_losses['loss_reg']):.6f}"
                        d['sigma class'] = f"{float(all_losses['sigma_class']):.6f}"
                        d['sigma reg'] = f"{float(all_losses['sigma_reg']):.6f}"

                    #pbar.set_postfix(d)

            epoch_train_loss /= (i+1)
            #if self.verbose:
            #    pbar.close()


            with torch.no_grad():
                self.model.eval()
                preds = np.zeros(len(X_val))
                val_labels = np.zeros((len(X_val)))
                num_elements = len(val_loader.dataset)
                num_batches = len(val_loader)

                if use_domain_adaptation:
                    domain_iter = iter(domain_val_loader)

                for i, batch in enumerate(val_loader):
                    batch_inputs, batch_labels = batch

                    batch_inputs, batch_labels = (batch_inputs.to(self.device),
                                                  batch_labels.to(self.device))

                    if class_weights_val is not None:
                        batch_weights = class_weight_to_sample_weight(
                            batch_labels, class_weights_val)
                        batch_weights = batch_weights.type(
                            torch.FloatTensor).to(self.device)
                    else:
                        batch_weights = None

                    if use_domain_adaptation:
                        domain_batch = next(domain_iter)

                        if self.domain_classifier:
                            batch_weights_domain = class_weight_to_sample_weight(
                                domain_batch[1], class_weights_domain_val)
                            batch_weights_domain = batch_weights_domain.type(
                                torch.FloatTensor).to(self.device)
                        else:
                            batch_weights_domain = None

                        mjj_truth = domain_batch[1]
                        batch_info = domain_batch[0]
                        class_pred = self.model(batch_inputs)[0]
                        _, mjj_pred = self.model(batch_info)
                        #print(i, num_batches, mjj_truth.shape, mjj_pred.shape, class_pred.shape, batch_labels.shape, batch_weights.shape)
                        if self.domain_classifier:
                            batch_loss, all_losses = self.criterion(class_pred, batch_labels, mjj_pred, mjj_truth, weight=batch_weights, weight_reg=batch_weights_domain)
                        else:
                            batch_loss, all_losses = self.criterion(class_pred, batch_labels, mjj_pred, mjj_truth, weight=batch_weights)
                    else:
                        batch_outputs = self.model(batch_inputs)
                        batch_loss = self.loss(batch_outputs, batch_labels, weight=batch_weights)

                    start = i*self.batch_size
                    end = start + self.batch_size
                    if i == num_batches-1:
                        end = num_elements

                    if use_domain_adaptation:
                        preds[start:end] = class_pred.cpu().numpy()[:, 0]
                    else:
                        preds[start:end] = batch_outputs.cpu().numpy()[:, 0]
                    epoch_val_loss += batch_loss.item()
                    val_labels[start:end] = batch_labels.cpu().numpy()[:, 0]
                epoch_val_SIC = metrics.val_sic(preds, val_labels)
                epoch_val_loss /= (i+1)

            # print("Validation loss:", epoch_val_loss)
            pbar0.set_description(
                f"Train loss: {epoch_train_loss:.6f} Validation loss: {epoch_val_loss:.6f}")

            if epoch == 0:
                train_losses = np.array([epoch_train_loss])
                val_losses = np.array([epoch_val_loss])
                val_SIC = np.array([epoch_val_SIC])
            else:
                train_losses = np.concatenate(
                    (train_losses, np.array([epoch_train_loss])))
                val_losses = np.concatenate(
                    (val_losses, np.array([epoch_val_loss])))
                val_SIC = np.concatenate(
                    (val_SIC, np.array([epoch_val_SIC])))

            if self.save_path is not None:
                np.save(self._train_loss_path(),
                        train_losses)
                np.save(self._val_loss_path(),
                        val_losses)
                np.save(self._val_SIC_path(),
                        val_SIC)
                if self.save_model:
                    self._save_model(self._model_path(epoch))

            if self.early_stopping:
                if epoch > self.patience:
                    if np.all(val_losses[-self.patience:] >
                              val_losses[-self.patience - 1]):
                        print("Early stopping at epoch", epoch)
                        break

        self.model.eval()
        if self.save_path is not None and self.save_model:
            print("Loading best model state...")
            self.load_best_model()

    def load_best_model(self):
        """Loads the best model state from the provided save_path.
        """
        val_losses = self.load_val_loss()
        best_epoch = np.argmin(val_losses)
        self.load_epoch_model(best_epoch)
        self.model.eval()

    def load_train_loss(self):
        """Loads the training loss from the provided save_path.

        Returns
        -------
        train_loss : numpy.ndarray
            Training loss.
        """
        if self.save_path is None:
            raise ValueError("save_path is None, cannot load train loss")
        return np.load(self._train_loss_path())

    def load_val_loss(self):
        """Loads the validation loss from the provided save_path.

        Returns
        -------
        val_loss : numpy.ndarray
            Validation loss.
        """
        if self.save_path is None:
            raise ValueError("save_path is None, cannot load val loss")
        return np.load(self._val_loss_path())

    def load_epoch_model(self, epoch):
        """Loads the model state from the provided save_path at the
        specified epoch.

        Parameters
        ----------
        epoch : int
            Epoch at which to load the model state.
        """
        self._load_model(self._model_path(epoch))

    def _load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path,
                                              map_location=self.device))

    def _save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def _train_loss_path(self):
        return join(self.save_path, "CLSF_train_loss.npy")

    def _val_loss_path(self):
        return join(self.save_path, "CLSF_val_loss.npy")

    def _val_SIC_path(self):
        return join(self.save_path, "CLSF_val_SIC.npy")

    def _model_path(self, epoch):
        return join(self.clsf_model_path, f"CLSF_epoch_{epoch}.par")

    def get_all_predictions(self, X, use_domain_adaptation=False):
        preds = np.zeros((self.epochs, len(X)))
        for N_epoch in tqdm(range(self.epochs)):
            self.load_epoch_model(N_epoch)
            #print(X.shape)
            #print(self.predict(X))
            preds[N_epoch] = self.predict(X)[:, 0]

        self.load_best_model()
        return preds


def numpy_to_torch_loader(X, y, sample_weights=None,
                          batch_size=128, shuffle=True,
                          device=torch.device("cpu"), drop_last=False):
    """Builds a torch DataLoader from numpy arrays.

    Parameters
    ----------
    X : numpy.ndarray
        Input data.
    y : numpy.ndarray
        Target data.
    sample_weights : numpy.ndarray, optional
        Sample weights.
    batch_size : int, default=128
        Batch size.
    shuffle : bool, default=True
        Whether to shuffle the data.
    device : torch.device, default=torch.device("cpu")
        Device to use.

    Returns
    -------
    dataloader : torch.utils.data.DataLoader
        DataLoader for the provided data.
    """

    X_torch = torch.from_numpy(
        X).type(torch.FloatTensor).to(device)
    y_torch = torch.from_numpy(
        y).type(torch.FloatTensor).to(device).reshape(-1, 1)

    if sample_weights is not None:
        sample_weights_torch = torch.from_numpy(
            sample_weights).type(torch.FloatTensor).to(device)
        dataset = torch.utils.data.TensorDataset(X_torch, y_torch,
                                                 sample_weights_torch)
    else:
        dataset = torch.utils.data.TensorDataset(X_torch, y_torch)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=drop_last)

    return dataloader


def class_weight_to_sample_weight(y, class_weights):
    """Converts class weights to sample weights.

    Parameters
    ----------
    y : torch.Tensor
        Target data.
    class_weights : dict
        Class weights.

    Returns
    -------
    sample_weights : torch.Tensor
        Sample weights.
    """

    y_cpu = y.to(torch.device("cpu"), copy=True)
    return ((torch.ones(y_cpu.shape) - y_cpu)
            * class_weights[0] + y_cpu*class_weights[1])
