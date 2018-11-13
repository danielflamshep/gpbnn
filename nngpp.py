import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr

from autograd import grad
from autograd.misc import flatten

import autograd.scipy.stats.norm as norm
from autograd.numpy.linalg import cholesky
from autograd.misc.optimizers import adam

from util import act_dict as act
from gp import sample_gpp
from bnn import sample_bnn, bnn_predict, shapes_and_num, reshape_weights
from nn import init_random_params, nn_predict, map_objective, setup_plot
from hypernet_exp import sample_gps
rs = npr.RandomState(0)


def sample_random_functions(x, n_samples=10, arch=[1,1], act='tanh'):
    _, n_weights = shapes_and_num(arch)
    w = rs.randn(n_samples, n_weights)
    return bnn_predict(w, x, arch, act)[:, :, 0]  # [ns, nd]

def fit_nn(x, y, arch):
    def nll(weights, t): return map_objective(weights, x, y)
    return adam(grad(nll), init_random_params(arch), step_size=0.05, num_iters=500)

def plot(x, y, f):
    plt.plot(x.ravel(), y.ravel(), 'b', marker='.')
    plt.plot(x, f, color='r', marker='.')
    plt.draw()
    plt.pause(1.0 / 60.0)
    plt.clf()

def get_weights(x, n_data, layer_sizes, rn_arch =[1,1], n_samples=10, save=False):

    xs, ys = sample_gps(n_samples, n_data, ker = 'rbf')
    weights = []
    for x, f in zip(xs, ys):
        opt_weights = fit_nn(x, f[:, None], layer_sizes)
        plot(x, f, nn_predict(opt_weights, x))
        weight, _ = flatten(opt_weights)
        weights.append(weight[:, None])
        print("fit1")

    weights = np.concatenate(weights, axis=1)

    mu = np.mean(weights, axis=1)
    sig = np.std(weights, axis=1)
    #sig = np.cov(weights)


    return mu, sig

def sample_normal(params, N_samples):
    mean, std = params
    return rs.randn(N_samples, mean.shape[0]) * std + mean  # [ns, nw]

def sample_full_normal(params, N_samples):
    mean, cov = params

def sample_normal(params, N_samples, full_cov=False):
    mean, cov = params
    if full_cov:
        jitter = 1e-7 * np.eye(mean.shape[0])
        L = cholesky(cov + jitter)
        e = rs.randn(N_samples, mean.shape[0])
        return np.dot(e, L) + mean
    else:
        return rs.randn(N_samples, mean.shape[0]) * cov + mean  # [ns, nw]


def plot_samples(params, x, layer_sizes, n_samples=1, act='tanh', save=None):

    #ws = sample_normal(params, n_samples)
    ws = sample_full_normal(params, n_samples)

    fnn = bnn_predict(ws, x, layer_sizes, act)[:, :, 0]  # [ns, nd]
    fgp = sample_gpp(x, n_samples)
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    bx = fig.add_subplot(111, frameon=False)
    bx.plot(x.ravel(), fgp.T, color='green')
    ax.plot(x.ravel(), fnn.T, color='red')
    if save is not None: plt.savefig(save)
    plt.show()


if __name__ == '__main__':

    # Set up
    arch = [1, 10, 1]
    n_data = 100
    x = np.linspace(0, 5, num=n_data).reshape(n_data, 1)

    params = get_weights(x, 100, arch)
    plot_samples(params, x, arch, n_samples=20)
