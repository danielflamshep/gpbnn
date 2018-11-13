import matplotlib.pyplot as plt
import autograd.scipy.stats.norm as norm
import autograd.scipy.stats.multivariate_normal as mvn
from autograd.numpy.linalg import solve, cholesky, det
from autograd.misc.flatten import flatten_func

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam

import plotting as p
from util import build_toy_dataset, act_dict
rs = npr.RandomState(0)

def full_cov(chol): return np.dot(chol, chol.T)
def full_chol(flat):return np.tril(flat)
def num_sqrt_cov(num_weights):
    n = np.tril(np.ones((num_weights,)))
    return n.sum().astype(int)
def shapes(layer_sizes): return list(zip(layer_sizes[:-1], layer_sizes[1:]))
#def pad(x, n): return np.pad(x, (0, n-len(x)), 'constant')
def pad(x, n): return np.concatenate([x, np.zeros((n-len(x),))])

def reshpe(x, num_weights):

    idx = [sum(range(i+1)) for i in range(num_weights+1)]
    rows = [pad(x[i:j], num_weights) for i, j in shapes(idx)]
    #print(len(rows))
    return np.vstack(rows)

def reshape(a, num_weights):
    mask = np.tri(num_weights, dtype=bool) # or np.arange(n)[:,None] > np.arange(n)
    out = np.zeros((num_weights, num_weights), dtype=int)
    out[mask] = a
    return out

def shapes_and_num(layer_sizes):
    shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    N_weights = sum((m + 1) * n for m, n in shapes)
    return shapes, N_weights

def unpack_layers(weights, layer_sizes):
    """ unpacks weights [ns, nw] into each layers relevant tensor shape"""
    shapes, _ = shapes_and_num(layer_sizes)
    n_samples = len(weights)
    for m, n in shapes:
        yield weights[:, :m * n].reshape((n_samples, m, n)), \
              weights[:, m * n:m * n + n].reshape((n_samples, 1, n))
        weights = weights[:, (m + 1) * n:]

def reshape_weights(weights, layer_sizes):
    return list(unpack_layers(weights, layer_sizes))


def bnn_predict(weights, inputs, layer_sizes, act):
    if len(inputs.shape)<3 : inputs = np.expand_dims(inputs, 0)  # [1,N,D]
    weights = reshape_weights(weights, layer_sizes)
    for W, b in weights:
        #print(W.shape, inputs.shape)
        outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
        inputs = act_dict[act](outputs)
    return outputs


def multivariate_gaussian_entropy(cov_sqrt):
    cov = np.dot(cov_sqrt, cov_sqrt.T)
    return 0.5 * cov.shape[0] * np.log(2*np.pi*np.exp(1)) + 0.5*np.log(det(cov))


def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)  # [ns]


def sample_weights(params, N_samples):
    mean, cov_sqrt_flat = params
    cov_sqrt = reshape(cov_sqrt_flat, mean.shape[0])
    #print(cov_sqrt[-1])
    jitter = 1e-2 * np.eye(mean.shape[0])
    e = rs.randn(N_samples, mean.shape[0])
    return np.dot(e, cov_sqrt.T+jitter) + mean  # [ns, nw]


def sample_bnn(params, x, N_samples, layer_sizes, act, noise=0.0):
    bnn_weights = sample_weights(params, N_samples)
    f_bnn = bnn_predict(bnn_weights, x, layer_sizes, act)[:, :, 0]
    return f_bnn + noise * rs.randn(N_samples, x.shape[0])  # [ns, nd]


def log_pdf_prior(weights, prior_params, sd, type=None):
    if type is None:
        return diag_gaussian_log_density(weights, 0, np.log(sd))
    elif type == "diagonal":
        prior_mean, prior_log_std = prior_params
        return diag_gaussian_log_density(weights, prior_mean, prior_log_std)
    elif type == "full":
        prior_mean, prior_cov = prior_params
        return mvn.logpdf(weights, prior_mean, prior_cov)
    else:
        prior_pis, prior_mus, prior_covs = prior_params
        log_pdf = np.zeros(weights.shape[0])
        for pi, mu, cov in zip(prior_pis, prior_mus, prior_covs):
            log_pdf += pi*mvn.logpdf(weights, mu, cov)
        return log_pdf

def vlb_objective(params, x, y, layer_sizes, n_samples,
                  prior_sd=10, model_sd=11, prior_params=None,
                  prior_type=None, act=np.tanh):
    """ ELBO = -E_q[log q(w))]- E_q(w)[log p(D,w)] """
    mean, sqrt_cov = params
    weights = sample_weights(params, n_samples)
    entropy = multivariate_gaussian_entropy(reshape(sqrt_cov, mean.shape[0]))

    f_bnn = sample_bnn(params, x, n_samples, layer_sizes, act)
    log_likelihood = diag_gaussian_log_density(y.T, f_bnn, np.log(model_sd))
    log_prior = log_pdf_prior(weights, prior_params, prior_sd, prior_type)
    #print(np.mean(log_likelihood))
    return - np.mean(log_likelihood+log_prior)-entropy


def init_var_params(layer_sizes, scale=0.01, scale_mean=.01):
    _, D = shapes_and_num(layer_sizes)
    n = num_sqrt_cov(D)
    mean = rs.randn(D)*scale_mean
    cov_sqrt = np.ones((n, ))*scale
    return mean, cov_sqrt


def train_bnn(data='expx', n_data=50, n_samples=20, arch=[1,20,1],
              prior_params=None, prior_type=None, act='rbf',
              iters=300, lr=0.01, plot=True, save=False):

    if type(data) == str:
        inputs, targets = build_toy_dataset(data=data, n_data=n_data)
    else:
        inputs, targets = data

    if plot: fig, ax = p.setup_plot()

    init_params= init_var_params(arch)

    def loss(params, t):
        return vlb_objective(params, inputs, targets, arch, n_samples, act=act,
                             prior_params=prior_params, prior_type=prior_type)



    def callback(params, t, g):
        plot_inputs = np.linspace(-10, 10, num=500)[:, None]

        f_bnn = sample_bnn(params, plot_inputs, 5, arch, act)
        #print(params[1])
        # Plot data and functions.
        p.plot_iter(ax, inputs, plot_inputs, targets, f_bnn)
        print("ITER {} | LOSS {}".format(t, -loss(params, t)))

    var_params = adam(grad(loss),init_params ,
                      step_size=lr, num_iters=iters, callback=callback)

if __name__ == '__main__':

    train_bnn()


