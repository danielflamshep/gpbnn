import matplotlib.pyplot as plt
import autograd.scipy.stats.norm as norm

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam

import plotting as p
from util import build_toy_dataset, act_dict
rs = npr.RandomState(0)


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
    if len(inputs.shape)<3: inputs = np.expand_dims(inputs, 0)  # [1,N,D]
    weights = reshape_weights(weights, layer_sizes)
    for W, b in weights:
        #print(W.shape, inputs.shape)
        outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
        inputs = act_dict[act](outputs)
    return outputs


def gaussian_entropy(log_std):
    return 0.5 * log_std.shape[0] * (1.0 + np.log(2*np.pi)) + np.sum(log_std)


def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)  # [ns]

def sample_weights(params, N_samples):
    mean, log_std = params
    return rs.randn(N_samples, mean.shape[0]) * np.exp(log_std) + mean  # [ns, nw]

def sample_bnn(params, x, N_samples, layer_sizes, act, noise=0.0):
    bnn_weights = sample_weights(params, N_samples)
    f_bnn = bnn_predict(bnn_weights, x, layer_sizes, act)[:, :, 0]
    return f_bnn + noise * rs.randn(N_samples, x.shape[0])  # [ns, nd]


def log_pdf_prior(weights, prior_params, sd):
    if prior_params is None:
        return diag_gaussian_log_density(weights, 0, np.log(sd))
    else:
        prior_mean, prior_log_std = prior_params
        return diag_gaussian_log_density(weights, prior_mean, prior_log_std)


def log_like(y, f, noise_var=0.01):
    """ computes log P(D|w) where weights: [ns, nw]"""
    # return diag_gaussian_log_density(fbnn, targets, np.log(noise_var))
    return -np.sum((f - y)**2, axis=1) / noise_var  # [ns]


def vlb_objective(params, x, y, layer_sizes, n_samples,
                  prior_sd=10, model_sd=0.1, prior_params=None, act=np.tanh):
    """ estimates ELBO = -H[q(w))]- E_q(w)[log p(D,w)] """
    mean, log_std = params
    weights = sample_weights(params, n_samples)
    entropy = gaussian_entropy(log_std)

    f_bnn = sample_bnn(params, x, n_samples,layer_sizes, act)
    log_likelihood = diag_gaussian_log_density(y.T, f_bnn, np.log(model_sd))
    log_prior = log_pdf_prior(weights, prior_params, prior_sd)

    return -entropy - np.mean(log_likelihood+log_prior)


def init_var_params(layer_sizes, scale=-5, scale_mean=1):
    _, num_weights = shapes_and_num(layer_sizes)
    return rs.randn(num_weights)*scale_mean, np.ones(num_weights)*scale  # mean, log_std

def train_bnn(data='cubic', n_data=5, n_samples=5, arch=[1,20,20,1], act='rbf',
              iters=100, lr=0.1, plot=True, save=False):

    if type(data) == str:
        inputs, targets = build_toy_dataset(data=data, n_data=n_data)
    else:
        inputs, targets = data

    if plot: fig, ax = p.setup_plot()

    def loss(params, t):
        return vlb_objective(params, inputs, targets, arch, n_samples, act=act)

    def callback(params, t, g):
        plot_inputs = np.linspace(-10, 10, num=500)[:, None]
        f_bnn = sample_bnn(params, plot_inputs, 5, arch, act)

        # Plot data and functions.
        p.plot_iter(ax, inputs, plot_inputs, targets, f_bnn)
        print("ITER {} | LOSS {}".format(t, -loss(params, t)))

    var_params = adam(grad(loss), init_var_params(arch),
                      step_size=lr, num_iters=iters, callback=callback)

if __name__ == '__main__':

    train_bnn()


