import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc import flatten
import autograd.scipy.stats.norm as norm
import numpy
from autograd.misc.optimizers import adam

from util import act_dict as act
from gp import sample_gpp
from bnn import sample_bnn, bnn_predict, shapes_and_num
from nn import init_random_params, nn_predict, map_objective, setup_plot

rs = npr.RandomState(0)


def sample_random_functions(x, n_samples=10, arch=[1,1,1]):
    _, n_weights = shapes_and_num(arch)
    w = rs.randn(n_samples, n_weights)
    return bnn_predict(w, x, arch, np.tanh)[:, :, 0]  # [ns, nd]

def fit_nn(x, y, arch):
    def nll(weights, t): return map_objective(weights, x, y)
    return adam(grad(nll), init_random_params(arch), step_size=0.05, num_iters=150)

def plot(x, y, f):
    plt.plot(x.ravel(), y.ravel(), 'b.')
    plt.plot(x, f, color='r')
    plt.draw()
    plt.pause(1.0 / 60.0)
    plt.clf()

def get_weights(x, n_data, layer_sizes, n_samples=100, save=False):


    fs = sample_random_functions(x, n_samples)
    # fs = sample_gpp(x, n_samples=n_samples)  # [ns, nd]
    print(fs.shape)

    #fig, ax = setup_plot()
    weights = []
    for f in fs:
        opt_weights = fit_nn(x, f[:, None], layer_sizes)
        plot(x, f, nn_predict(opt_weights, x))
        weight, _ = flatten(opt_weights)
        weights.append(weight[:, None])
        print("fit1")

    weights = np.concatenate(weights, axis=1)

    if save:
        numpy.save("wieghts", weights)


    mu = np.mean(weights, axis=1)
    std = np.std(weights, axis=1)

    print(weights.shape, mu.shape, std.shape)
    print(np.mean(mu), np.mean(std))

    return mu, std

def sample_weights(params, N_samples):
    mean, std = params
    return rs.randn(N_samples, mean.shape[0]) * std + mean  # [ns, nw]


def plot_samples(params, x, layer_sizes, n_samples=1):

    ws = sample_weights(params, n_samples)
    fnn = bnn_predict(ws, x, layer_sizes, np.tanh)[:, :, 0]
    fs = sample_random_functions(x, n_samples)

    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)


    print(fs.shape, fnn.shape)
    # get samples to compare

    ax.plot(x.ravel(), fs.T, color='green')
    ax.plot(x.ravel(), fnn.T, color='red')
    plt.show()


if __name__ == '__main__':

    # Set up
    arch = [1,1, 1]
    n_data = 100
    x = np.linspace(0, 10, num=n_data).reshape(n_data, 1)

    params = get_weights(x, 10, arch)
    plot_samples(params, x, arch, n_samples=20)
