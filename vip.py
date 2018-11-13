import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.numpy.linalg import solve, cholesky, inv
from autograd.scipy.stats import multivariate_normal as mvn

from autograd import grad
from autograd.misc import flatten
import autograd.scipy.stats.norm as norm
from autograd.misc.optimizers import adam

from util import act_dict, build_toy_dataset
from bnn import shapes_and_num, sample_bnn, diag_gaussian_log_density
from nngpp import sample_full_normal, sample_normal

def unbiased(s): return 1/(s.shape[0]-1)**.5

def empirical_moments(fs):  # shape [ns, nd]
    mu = np.mean(fs, axis=0, keepdims=True)
    cov= np.dot((fs-mu).T, fs-mu) / fs.shape[0]
    return mu, cov  # ([nd], [nd, nd] )


def prior_mean_cov(prior_params, X, n_samples, arch, act):
    fs = sample_bnn(prior_params, X, n_samples, arch, act)
    return empirical_moments(fs.T)


def qa_posterior_moments(m, K_ff, y, noise):
    B = cholesky(K_ff + 1e-7*np.eye(K_ff.shape[0]))
    Sigma = inv(np.dot(B.T, B)+noise*np.eye(B.shape[0]))/noise

    mu = np.dot(Sigma, np.dot(B.T, (y-m).T))
    print(mu.shape, Sigma.shape, y.shape)
    return mu, Sigma


def predictions(prior_params, X, y, Xstar, noise, f_samples, arch, act):
    fs = sample_bnn(prior_params, X, f_samples, arch, act)

    m, K_ff = prior_mean_cov(prior_params, X, f_samples, arch, act)
    qa_mean, qa_Sigma = qa_posterior_moments(m, K_ff, y, noise)

    fss = sample_bnn(prior_params, Xstar, f_samples, arch, act)
    mstar = np.mean(fs, axis=0, keepdims=True)
    phi = (fss-mstar)/unbiased(fss)

    pred_mean = np.dot(phi.T, qa_mean)
    pred_var = np.dot(phi.T, np.dot(qa_Sigma, phi))

    return pred_mean, pred_var


def predictions_qa(prior_params, X, y, Xstar, noise, f_samples, arch, act):
    fs = sample_bnn(prior_params, Xstar, f_samples, arch, act)
    m, K_ff = prior_mean_cov(prior_params, X, f_samples, arch, act)
    a_samples = sample_full_normal(qa_posterior_moments(m, K_ff, y, noise), 1)
    print(a_samples.shape)
    return np.sum(fs*a_samples, 0)


def elbo(prior_params, qa_params, X, y, f_samples, arch, act, noise):
    fs = sample_bnn(prior_params, X, f_samples, arch, act)
    m = np.mean(fs, axis=0, keepdims=True)
    qa_mean, qa_Sigma = qa_params

    # m, K_ff = prior_mean_cov(prior_params, X, f_samples, arch, act)
    # a_samples = sample_full_normal(qa_posterior_moments(m, K_ff, y, noise),1)
    # qa_mean, qa_Sigma = qa_posterior_moments(m, K_ff, y, noise)
    a_samples = sample_normal(qa_params,1)
    print(a_samples.shape, fs.shape, m.shape)
    mean = a_samples * (fs - m)/ unbiased(fs)

    log_qy = diag_gaussian_log_density(y, mean, noise)
    log_qa = mvn.logpdf(a_samples, qa_mean, qa_Sigma)
    log_pa = diag_gaussian_log_density(a_samples, 0, 1)

    return np.mean(log_qy-log_qa+log_pa)


if __name__ == '__main__':


    arch = [1, 20, 20, 1]
    activation = 'rbf'
    num_fncs = 20
    noise = 0.1
    inputs, targets = build_toy_dataset(data='cubic', n_data=70)

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    plt.ion()
    plt.show(block=False)


    def objective(prior_params, qa_params,t):
        return -elbo(prior_params, qa_params, inputs, targets, num_fncs, arch, activation, noise)

    def callback(params, t, g):

        plot_inputs = np.linspace(-8, 8, num=400).reshape(400,1)
        f_bnn = predictions_qa(params, inputs, targets, noise, num_fncs, arch, activation)

        # Plot data and functions.

        plt.cla()
        ax.plot(inputs.ravel(), targets.ravel(), 'k.')
        ax.plot(plot_inputs, f_bnn.T, color='r')
        ax.set_ylim([-5, 5])
        plt.draw()
        plt.pause(1.0 / 60.0)

        print("ITER {} | OBJ {}".format(t, objective(params, t)))

    _, num_params = shapes_and_num(arch)
    init_var_params = (np.random.randn(num_params), np.random.randn(num_params))

    var_params = adam(grad(objective), init_var_params,
                      step_size=0.1, num_iters=50, callback=callback)


