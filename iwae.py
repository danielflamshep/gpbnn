import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt

from autograd.misc.optimizers import adam
from autograd import grad

from autograd.scipy.stats import multivariate_normal as mvn
import autograd.scipy.stats.norm as norm

from bnn import shapes_and_num, sample_bnn, sample_weights, init_var_params
from gp import sample_gpp
from util import kernel_dict, act_dict

rs = npr.RandomState(0)


def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)

def diag_gaussian_density(x, mu, log_std):
    return np.sum(norm.pdf(x, mu, np.exp(log_std)), axis=-1)


def NN_likelihood(x, y, params, arch, act, n_samples, noise=1e-1):
    f = sample_bnn(params, x, n_samples, arch, act)  # [ns, nd]
    return diag_gaussian_density(f, y, noise) # y shape [ns, nd]


def outer_objective(params_prior, n_data, n_samples, layer_sizes, act, ker='rbf'):

    # x = np.random.uniform(low=0, high=10, size=(n_data, 1))

    x = np.linspace(0, 10, num=n_data).reshape(n_data, 1)
    fgp = sample_gpp(x, n_samples=n_samples, kernel=ker)

    def objective_inner(params_q, t):
        return -np.mean(NN_likelihood(x, fgp, params_q, layer_sizes, act, n_samples))

    op_params_q = adam(grad(objective_inner), init_var_params(layer_sizes),
                       step_size=0.1, num_iters=100)

    wq = sample_weights(op_params_q, n_samples) # [ns, nw]
    pnn = NN_likelihood(x, fgp, op_params_q, layer_sizes, act, n_samples)
    p = diag_gaussian_density(wq, params_prior[0], params_prior[1])
    q = diag_gaussian_density(wq, op_params_q[0], op_params_q[1])

    iwae = p*pnn/q

    # print(iwae.shape)

    return np.mean(np.log(iwae))


if __name__ == '__main__':

    n_data, n_samples, arch = 100, 3, [1, 20, 20, 1]
    act = 'tanh'
    ker = 'lin'
    f, ax = plt.subplots(2, sharex=True)
    plt.ion()
    plt.show(block=False)

    def kl(prior_params,t):
        return outer_objective(prior_params, n_data, n_samples, arch, act)

    def callback(params, iter, g):

        n_samples = 3
        plot_inputs = np.linspace(-5, 5, num=100)

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
                        step_size=0.1, num_iters=100, callback=callback)









