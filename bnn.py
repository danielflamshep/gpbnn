import matplotlib.pyplot as plt
import autograd.scipy.stats.norm as norm
import autograd.scipy.stats.multivariate_normal as mvn

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
            log_pdf += mvn.logpdf(weights, mu, cov)
        return log_pdf

def log_like(y, f, noise_var=0.01):
    """ computes log P(D|w) where weights: [ns, nw]"""
    # return diag_gaussian_log_density(fbnn, targets, np.log(noise_var))
    return -np.sum((f - y)**2, axis=1) / noise_var  # [ns]


def vlb_objective(params, x, y, layer_sizes, n_samples,
                  prior_sd=10, model_sd=0.1, prior_params=None,
                  prior_type=None, act=np.tanh):
    """ estimates ELBO = -H[q(w))]- E_q(w)[log p(D,w)] """
    mean, log_std = params
    weights = sample_weights(params, n_samples)
    entropy = gaussian_entropy(log_std)

    f_bnn = sample_bnn(params, x, n_samples,layer_sizes, act)
    log_likelihood = diag_gaussian_log_density(y.T, f_bnn, np.log(model_sd))
    log_prior = log_pdf_prior(weights, prior_params, prior_sd, prior_type)

    return -entropy - np.mean(log_likelihood+log_prior)


def init_var_params(layer_sizes, scale=-5, scale_mean=1):
    _, num_weights = shapes_and_num(layer_sizes)
    return rs.randn(num_weights)*scale_mean, np.ones(num_weights)*scale  # mean, log_std

def train_bnn(data='expx', n_data=20, n_samples=5, arch=[1,20,20,1],
              prior_params=None, prior_type=None, act='rbf',
              iters=65, lr=0.07, plot=True, save=False):

    if type(data) == str:
        inputs, targets = build_toy_dataset()
    else:
        inputs, targets = data

    if plot: fig, ax = p.setup_plot()

    def loss(params, t):
        return vlb_objective(params, inputs, targets, arch, n_samples, act=act,
                             prior_params=prior_params, prior_type=prior_type)

    def callback(params, t, g):
        plot_inputs = np.linspace(-8, 8, num=400)[:, None]
        f_bnn = sample_bnn(params, plot_inputs, 5, arch, act)

        # Plot data and functions.
        p.plot_iter(ax, inputs, plot_inputs, targets, f_bnn)
        print("ITER {} | LOSS {}".format(t, -loss(params, t)))
        if t > 50:
            D = inputs, targets
            x_plot = np.reshape(np.linspace(-8, 8, 400), (400, 1))
            pred = sample_bnn(params, x_plot, 5, arch, act)
            p.plot_deciles(x_plot.ravel(), pred.T, D, str(t) + "bnnpostfullprior", plot="gpp")

    var_params = adam(grad(loss), init_var_params(arch),
                      step_size=lr, num_iters=iters, callback=callback)


    D = inputs, targets
    x_plot = np.reshape(np.linspace(-8, 8, 400), (400, 1))
    pred = sample_bnn(var_params, x_plot, 5, arch, act)
    p.plot_deciles(x_plot.ravel(), pred.T, D,"bnnpostfullprior", plot="gpp")



if __name__ == '__main__':

    train_bnn()


