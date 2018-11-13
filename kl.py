import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
from autograd.scipy.stats import multivariate_normal as mvn
from autograd import grad
from autograd.misc.optimizers import adam
from autograd.numpy.linalg import solve, cholesky, det

from util import kernel_dict, sample_inputs
from bnn import sample_bnn, init_var_params
from gp import sample_gpp

rs = npr.RandomState(0)


def empirical_cov(x):  # x shape [ns, nd]
    centred = x - np.mean(x, axis=0, keepdims=True)
    return np.dot(centred.T, centred) / x.shape[0]  # [nd, nd]


def entropy_estimate(samples):  # [ns, nd]
    return mvn.entropy(np.zeros(samples.shape[1]), empirical_cov(samples)) # [1]

def log_gp_prior(y_bnn, K):  # [nf, nd] [nd, nd]
    """ computes: log p_gp(f), f ~ p_BNN(f) """
    L = cholesky(K)
    a = solve(L, y_bnn.T)  # a = L^-1 y_bnn  [nf, nd]
    return -0.5*np.mean(a**2, axis=0)  # [nf]


def kl_estimate(params, layer_sizes, n_data, N_samples, act='rbf', kernel='rbf', noise=1e-7):
    x = np.random.uniform(-10, 10, size=(n_data, 1))
    y = sample_bnn(params, x, N_samples, layer_sizes, act)  # [nf, nd]
    covariance = kernel_dict[kernel]
    cov = covariance(x, x) + noise * np.eye(x.shape[0])
    print(cov, y.shape, det(cov))
    log_gp = log_gp_prior(y, cov)
    #log_gp = mvn.logpdf(y, np.zeros(y.shape[1]), cov)
    return -entropy_estimate(y) - np.mean(log_gp)


if __name__ == '__main__':

    n_data, n_samples, arch = 1000, 1, [1, 20, 20, 1]
    act, ker = 'rbf', 'rbf'

    f, ax = plt.subplots(2, sharex=True)
    plt.ion()
    plt.show(block=False)

    def kl(prior_params, t):
        return kl_estimate(prior_params, arch, n_data, n_samples, act, ker)

    def callback(params, iter, g):

        n_samples = 3
        plot_inputs = np.linspace(-10, 10, num=500)

        f_bnn = sample_bnn(params, plot_inputs[:, None], n_samples, arch, act)
        fgp   = sample_gpp(plot_inputs[:, None], n_samples, kernel=ker)

        for axes in ax: axes.cla()
        # ax.plot(x.ravel(), y.ravel(), 'ko')
        ax[0].plot(plot_inputs, fgp.T, color='green')
        ax[1].plot(plot_inputs, f_bnn.T, color='red')
        #ax[0].set_ylim([-5, 5])
        #ax[1].set_ylim([-5, 5])

        plt.draw()
        plt.pause(1.0/40.0)

        print("Iteration {} KL {} ".format(iter, kl(params, iter)))

    prior_params = adam(grad(kl), init_var_params(arch),
                        step_size=0.04, num_iters=100, callback=callback)

