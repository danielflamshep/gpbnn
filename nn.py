from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm

from autograd import grad
from autograd.misc import flatten
from autograd.misc.optimizers import adam

from util import act_dict

def init_random_params(layer_sizes, scale =0.1, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples, one for each layer."""
    return [(rs.randn(insize, outsize) * scale,   # weight matrix
             rs.randn(outsize) * scale)           # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]

def nn_predict(params, inputs, act='tanh'):
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = act_dict[act](outputs)
    return outputs

def log_gaussian(params, scale):
    flat_params, _ = flatten(params)
    return np.sum(norm.logpdf(flat_params, 0, scale))

def logprob(weights, inputs, targets, act, noise_scale=0.1):
    predictions = nn_predict(weights, inputs, act)
    return np.sum(norm.logpdf(predictions, targets, noise_scale))

def map_objective(w, x, y, act='rbf', var =10):
    return -logprob(w, x, y, act)-log_gaussian(w, var)

def build_toy_dataset(n_data=8000, noise_std=0.1):
    rs = npr.RandomState(0)
    inputs  = np.concatenate([np.linspace(0, 3, num=n_data/2),
                              np.linspace(6, 8, num=n_data/2)])
    targets = np.cos(inputs) + rs.randn(n_data) * noise_std
    inputs = (inputs - 4.0) / 2.0
    inputs  = inputs[:, np.newaxis]
    targets = targets[:, np.newaxis] / 2.0
    return inputs, targets

def setup_plot():
    fig = plt.figure(figsize=(12,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.show(block=False)
    return fig, ax

if __name__ == '__main__':

    init_scale = 0.1
    weight_prior_variance = 10.0
    init_params = init_random_params(init_scale, layer_sizes=[1, 400, 400, 1])

    inputs, targets = build_toy_dataset()

    def nn_objective(weights, t):
        return -logprob(weights, inputs, targets)\
               -log_gaussian(weights, weight_prior_variance)

    print(grad(nn_objective)(init_params, 0))

    # Set up figure.
    fig, ax = setup_plot()

    def callback(params, t, g):
        print("Iteration {} log likelihood {}".format(t, -nn_objective(params, t)))

        # Plot data and functions.
        plt.cla()
        ax.plot(inputs.ravel(), targets.ravel(), 'bx', ms=12)
        plot_inputs = np.reshape(np.linspace(-7, 7, num=300), (300,1))
        outputs = nn_predict(params, plot_inputs)
        ax.plot(plot_inputs, outputs, 'r', lw=3)
        ax.set_ylim([-1, 1])
        plt.draw()
        plt.pause(1.0/60.0)

    print("Optimizing network parameters...")
    optimized_params = adam(grad(nn_objective), init_params,
                            step_size=0.01, num_iters=1000, callback=callback)