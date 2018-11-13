import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
from autograd.scipy.stats import multivariate_normal as mvn
import autograd.scipy.stats.norm as norm
from autograd.numpy.linalg import cholesky

from autograd import grad
from autograd.misc.optimizers import adam

from util import sample_inputs
from bnn import sample_bnn, init_var_params, diag_gaussian_log_density, sample_weights
from gp import sample_gpp, unpack_kernel_params, covariance
rs = npr.RandomState(0)


def sample_gpp(ker_params, x, n_samples): # x shape [nd,1]
    K = covariance(ker_params, x, x) + 1e-7 * np.eye(x.shape[0])
    L = cholesky(K)
    e = rs.randn(n_samples, x.shape[0])
    return np.dot(e, L.T)  # [ns, nd]

def kl_estimate(params, n_samples, arch, act):
    prior_params, noise, kernel_params, x = params
    x = sample_inputs('gridbox', 100, (0,10))
    y = sample_bnn(prior_params, x, n_samples, arch, act, noise) # [nf, nd]
    f = sample_bnn(prior_params, x, n_samples, arch, act)


    w = sample_weights(prior_params, 1)
    mu, log_std = prior_params
    log_prior = diag_gaussian_log_density(w, mu, log_std)
    log_likelihood = diag_gaussian_log_density(y, f, noise)

    jitter = 1e-7 * np.eye(y.shape[0])
    cov = covariance(kernel_params, x, x) + jitter
    log_pgp = mvn.logpdf(y, np.zeros(y.shape[1]), cov)

    print(log_likelihood.shape, log_pgp.shape)

    return np.mean(log_likelihood+log_prior-log_pgp)

def init_params(arch):
    return [init_var_params(arch), rs.randn(2), 1e-5*rs.randn()]


if __name__ == '__main__':

    n_data = 50
    samples = 5
    save_plots = False
    plot_during = True

    bnn_layer_sizes = [1, 20, 20, 1]

    # Training parameters
    param_scale = 0.01
    batch_size = 5
    num_epochs = 200

    step_size_max = 0.01
    step_size_min = 0.01


    def objective(params, t):
        return gan_objective(bnn_param, n_data, samples, bnn_layer_sizes)

    # set up fig
    if plot_during:
        f, ax = plt.subplots(3, sharex=True, frameon=False)
        plt.ion()
        plt.show(block=False)

    def callback(bnn_params, dsc_params, iter, gen_gradient, dsc_gradient):
        # Sample functions from priors f ~ p(f)
        n_samples = 3
        plot_inputs = np.linspace(-8, 8, num=100).reshape(100,1)
        std_norm_param = init_var_params(bnn_layer_sizes, scale_mean=0, scale=1)

        f_bnn_gpp = sample_bnn(bnn_params, plot_inputs, n_samples, bnn_layer_sizes)
        f_gp      = sample_gpp(plot_inputs, n_samples)
        f_bnn     = sample_bnn(std_norm_param, plot_inputs, n_samples, bnn_layer_sizes)

        # Plot samples of functions from the bnn and gp priors.
        if plot_during:
            for axes in ax: axes.cla()

            # ax.plot(x.ravel(), y.ravel(), 'ko')
            ax[0].plot(plot_inputs, f_gp, color='green')
            ax[1].plot(plot_inputs, f_bnn_gpp, color='red')
            ax[2].plot(plot_inputs, f_bnn, color='blue')

            plt.draw()
            plt.pause(1.0/40.0)

        print("Iteration {} ".format(iter))


    # INITIALIZE THE PARAMETERS
    init_gen_params = init_var_params(bnn_layer_sizes, scale=-1.5)

    # OPTIMIZE
    grad_gan = grad(objective)

    optimized_params = adam(grad_gan, init_gen_params, init_dsc_params,
                                    step_size_max=step_size_max, step_size_min=step_size_min,
                                    num_iters=num_epochs, callback=callback)

