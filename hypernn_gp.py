import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.numpy.linalg import cholesky

from autograd import grad
from autograd.misc import flatten
import autograd.scipy.stats.norm as norm
from autograd.misc.optimizers import adam

from util import setup_plot, plot_iter, plot_fs
from bnn import shapes_and_num, reshape_weights, bnn_predict, log_pdf_prior, log_like
from nn import nn_predict, map_objective, setup_plot, init_random_params
from nngpp import sample_random_functions, plot_samples
from gp import sample_gpp

rs = npr.RandomState(2)


def log_gaussian(x, mu, std):
    return np.mean(norm.logpdf(x, mu, std))

def hyper_predict(params, x, y, nn_arch, nn_act):  # y shape is [nf, nd]
    weights = nn_predict(params, y, 'relu')  # [nf, nw]
    return bnn_predict(weights, x, nn_arch, nn_act)[:, :, 0]  # [nf, nd]

def hyper_loss(hyper_param, x, y, net_arch, act):
    f = hyper_predict(hyper_param, x, y, net_arch, act)
    return -log_gaussian(f, y, 1)

def get_moments(weights, full_cov=False):  # [ns, nw]
    if full_cov:
        return np.mean(weights, axis=0), np.cov(weights.T)
    else:
        return np.mean(weights, axis=0), np.std(weights, axis=0)

def sample_normal(params, N_samples):
    mean, cov = params
    if len(cov.shape) > 1:
        jitter = 1e-7 * np.eye(mean.shape[0])
        L = cholesky(cov + jitter)
        e = rs.randn(N_samples, mean.shape[0])
        return np.dot(e, L) + mean
    else:
        return rs.randn(N_samples, mean.shape[0]) * cov + mean  # [ns, nw]


def plot_samples(params, x, n_samples, layer_sizes, act, ker, save=None):

    ws = sample_normal(params, n_samples)
    fnn = bnn_predict(ws, x, layer_sizes, act)[:, :, 0]  # [ns, nd]
    fgp = sample_gpp(x, n_samples, ker)
    stdnorm_param = (np.zeros(ws.shape[1]), np.ones(ws.shape[1]))
    f = bnn_predict(sample_normal(stdnorm_param, n_samples), x, layer_sizes, act)[:, :, 0]

    fig, ax = plt.subplots(3, sharex=True, frameon=False)
    plt.ion()
    plt.show(block=False)
    ax[0].plot(x.ravel(), fgp.T, color='green')
    ax[1].plot(x.ravel(), fnn.T, color='red')
    ax[2].plot(x.ravel(), f.T, color='blue')
    if save is not None: plt.savefig(save)
    plt.show()

if __name__ == '__main__':

    n_data = 20
    n_functions = 100
    nn_arch = [1, 30, 30, 1]
    _, num_weights = shapes_and_num(nn_arch)
    hyper_arch = [n_data, 20, 20, num_weights]

    act = 'rbf'; ker = 'rbf'

    save_name = '1nd-'+str(n_data)+'nf-'+str(n_functions)+"-"+act+ker
    x = np.linspace(-10, 10, n_data).reshape(n_data,1)
    ys = sample_gpp(x, n_samples=n_functions, kernel=ker)  # [ns, nd]

    plot = True
    if plot: fig, ax = setup_plot()

    def objective(params, t): return hyper_loss(params, x, ys, nn_arch, act)

    def callback(params, t, g):
        y = sample_gpp(x, 1, kernel=ker)
        xp = np.linspace(-10, 10, 500).reshape(500, 1)
        preds = hyper_predict(params, xp, y, nn_arch, act)
        if plot: plot_iter(ax, x, xp, y, preds)
        print("ITER {} | OBJ {}".format(t, objective(params, t)))

    var_params = adam(grad(objective), init_random_params(hyper_arch),
                      step_size=0.01, num_iters=200, callback=callback)

    n_data_test = 500
    xtest = np.linspace(-10, 10, n_data_test).reshape(n_data_test, 1)

    fgps = sample_gpp(x, n_samples=500, kernel=ker)
    ws = nn_predict(var_params, fgps, "relu")
    fnns = hyper_predict(var_params, xtest, fgps, nn_arch, act)
    plot_fs(xtest, fnns)

    plot_samples(get_moments(ws), xtest, 10, nn_arch, act=act, ker=ker, save = save_name+'.pdf')