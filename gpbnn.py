import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt

from autograd.misc.optimizers import adam
from autograd import grad

from autograd.scipy.stats import multivariate_normal as mvn
import autograd.scipy.stats.norm as norm

from bnn import shapes_and_num, sample_bnn, sample_weights, init_var_params
from gp import sample_gpp
rs = npr.RandomState(0)

rbf = lambda x: np.exp(-x ** 2)
relu = lambda x: np.maximum(x, 0.)


def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)


def NN_likelihood(x, y, params, arch, act, n_samples=10, noise=1e-1):
    f = sample_bnn(params, x, n_samples, arch, act)
    return diag_gaussian_log_density(f, y, noise)


def outer_objective(params_prior, n_data, n_samples, layer_sizes, act=rbf):

    # x = np.random.uniform(low=0, high=10, size=(n_data, 1))

    x = np.linspace(0, 10, num=n_data).reshape(n_data, 1)
    fgp = sample_gpp(x, n_samples=n_samples, kernel='rbf')


    def objective_inner(params_q, t):
        return -np.mean(NN_likelihood(x, fgp, params_q, layer_sizes, act))

    op_params_q = adam(grad(objective_inner), init_var_params(layer_sizes),
                       step_size=0.1, num_iters=200)

    wq = sample_weights(op_params_q, n_samples)
    log_pnn = NN_likelihood(x, fgp, op_params_q, layer_sizes, act)
    log_p = diag_gaussian_log_density(wq, params_prior[0], params_prior[1])
    log_q = diag_gaussian_log_density(wq, op_params_q[0], op_params_q[1])

    return -np.mean(log_pnn+log_p-log_q)


if __name__ == '__main__':

    n_data, n_samples, arch = 10, 1, [1, 20, 20, 1]

    f, ax = plt.subplots(2, sharex=True)
    plt.ion()
    plt.show(block=False)

    def kl(prior_params,t):
        return outer_objective(prior_params, n_data, n_samples, arch)

    def callback(params, iter, g):

        n_samples = 3
 f_bnn_gpp = sample_bnn(params, plot_inputs[:, None], n_samples, arch, rbf)
        f_gp      = sample_gpp(plot_inputs[:, None], n_samples)
_min
        for axes in ax: axes.cla()
        # ax.plot(x.ravel(), y.ravel(), 'ko')
        ax[0].plot(plot_inputs, f_gp.T, color='green')
        ax[1].plot(plot_inputs, f_bnn_gpp.T, color='red')
        #ax[0].set_ylim([-5, 5])
        #ax[1].set_ylim([-5, 5])

        plt.draw()
        plt.pause(1.0/40.0)

        print("Iteration {} KL {} ".format(iter, kl(params, iter)))

    prior_params = adam(grad(kl), init_var_params(arch),
                        step_size=0.05, num_iters=100, callback=callback)









