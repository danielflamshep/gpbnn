import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.numpy.linalg import solve, cholesky
import autograd.scipy.stats.multivariate_normal as mvn
from autograd import grad
from autograd.misc.optimizers import  sgd, adam
import os
import plotting
from util import kernel_dict
from util import build_toy_dataset

rs = npr.RandomState(0)


def unpack_kernel_params(params, noise=1e-4):
    mean, cov_params, noise_scale = params[0], params[2:], np.exp(params[1]) + noise
    return mean, cov_params, noise_scale  # shape [1] [2] [1]


def covariance(kernel_params, x, xp):
    os = np.exp(kernel_params[0])
    ls = np.exp(kernel_params[1:])
    diffs = x[:, None] -xp[None, :]
    return os * np.exp(-0.5 * np.sum(diffs/ls ** 2, axis=2))


def predict(params, x, y, xstar):
    """ Returns the predictive mean f(xstar) and covariance f(xstar)"""
    mean, cov_params, noise_scale = unpack_kernel_params(params)

    K_ff = covariance(cov_params, xstar, xstar)
    K_yf = covariance(cov_params, x, xstar)
    K_yy = covariance(cov_params, x, x) + noise_scale * np.eye(len(y))

    pred_mean = mean + np.dot(solve(K_yy, K_yf).T, y - mean)
    pred_cov = K_ff - np.dot(solve(K_yy, K_yf).T, K_yf)

    return pred_mean, pred_cov


def log_marginal_likelihood(params, x, y):
    """ computes log p(y|X) = log N(y|mu, K + std*I) """
    mean, cov_params, noise_scale = unpack_kernel_params(params)
    cov_y_y = covariance(cov_params, x, x) + noise_scale * np.eye(len(y))
    prior_mean = mean * np.ones(len(y))
    return mvn.logpdf(y, prior_mean, cov_y_y)


def sample_functions(params, x, y, xs, samples=5):
    pred_mean, pred_cov = predict(params, x, y, xs)
    marg_std = np.sqrt(np.diag(pred_cov))
    sampled_funcs = rs.multivariate_normal(pred_mean, pred_cov, size=samples)
    return sampled_funcs.T


def sample_gpp(x, n_samples, kernel='rbf', noise=1e-6):
    """ samples from the gp prior. x shape [N_data,1]"""
    covariance = kernel_dict[kernel]
    K = covariance(x, x) + noise * np.eye(x.shape[0])
    # print(K[0],K[:,0], K.shape) ; exit()
    L = cholesky(K)
    e = rs.randn(n_samples, x.shape[0])
    return np.dot(e, L.T)  # [ns, nd]


if __name__ == '__main__':

    D = 1
    exp_num = 2
    n_data = 70
    iters = 5
    data = "expx"
    samples = 5
    save_plots = True
    plot_during = False
    rs = npr.RandomState(0)
    mvnorm = rs.multivariate_normal
    save_title = "exp-" + str(exp_num)+data + "-posterior samples {}".format(samples)
    save_dir = os.path.join(os.getcwd(), 'plots', 'gp', save_title)

    num_params = D+3  # mean , 2 kernel params, noise

    X, y = build_toy_dataset(data, n_data)
    y = y.ravel()

    objective = lambda params, t: log_marginal_likelihood(params, X, y)

    if plot_during:
        fig = plt.figure(figsize=(12,8), facecolor='white')
        ax = fig.add_subplot(111, frameon=False)
        plt.show(block=False)

    def callback(params, t, g):
        print("iteration {} Log likelihood {}".format(t, objective(params, t)))

        if plot_during:
            plt.cla()
            x_plot = np.reshape(np.linspace(-8, 8, 400), (400, 1))
            pred_mean, pred_cov = predict(params, X, y, x_plot)  # shapes [N_data], [N_data, N_data]
            std = np.sqrt(np.diag(pred_cov))  # shape [N_data]
            ax.plot(x_plot, pred_mean, 'b')
            ax.fill_between(x_plot.ravel(), pred_mean - 1.96*std, pred_mean + 1.96*std,
                            color=sns.xkcd_rgb["sky blue"])

            # Show sampled functions from posterior.
            sf = mvnorm(pred_mean, pred_cov, size=5)  # [ns, nd]
            ax.plot(x_plot, sf.T)

            ax.plot(X, y, 'k.')
            ax.set_ylim([-2, 3])
            ax.set_xticks([])
            ax.set_yticks([])
            plt.draw()
            plt.pause(1.0/60.0)
            if t == 1:
                D = X, y[:, None]
                p = sample_functions(params, X, y, x_plot, samples)
                plotting.plot_deciles(x_plot.ravel(), p, D, save_dir, plot="gp")


    # Initialize covariance parameters
    rs = npr.RandomState(0)
    init_params = 0.1 * rs.randn(num_params)
    cov_params = adam(grad(objective), init_params,
                      step_size=0.1, num_iters=iters, callback=callback)

    if save_plots:
        D = X, y[:, None]
        x_plot = np.reshape(np.linspace(-8, 8, 400), (400, 1))
        p = sample_functions(cov_params, X, y, x_plot, samples)
        print(p.shape)
        plotting.plot_deciles(x_plot.ravel(), p, D, save_dir, plot="gp")







