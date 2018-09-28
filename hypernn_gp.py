import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.numpy.linalg import cholesky

from autograd import grad
from autograd.misc import flatten
import autograd.scipy.stats.norm as norm
from autograd.misc.optimizers import adam

import plotting as p
from bnn import shapes_and_num, reshape_weights, bnn_predict, log_pdf_prior, log_like
from nn import nn_predict, setup_plot, init_random_params
from gp import sample_gpp

rs = npr.RandomState(2)

def log_gaussian(x, mu, std):
    return np.mean(norm.logpdf(x, mu, std))

def hyper_predict(params, x, y, nn_arch, nn_act):  # y shape is [nf, nd]
    weights = nn_predict(params, y, 'relu')  # [nf, nw]
    return bnn_predict(weights, x, nn_arch, nn_act)[:, :, 0]  # [nf, nd]

def hyper_loss(hyper_param, x, y, net_arch, act):
    f = hyper_predict(hyper_param, x,y, net_arch, act)
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
    " plots samples of "
    ws = sample_normal(params, n_samples)
    fnn = bnn_predict(ws, x, layer_sizes, act)[:, :, 0]  # [ns, nd]
    fgp = sample_gpp(x, n_samples, ker)

    # functions from a standard normal bnn prior
    wz = sample_normal((np.zeros(ws.shape[1]), np.ones(ws.shape[1])), n_samples)
    f = bnn_predict(wz, x, layer_sizes, act)[:, :, 0]
    plot_priors(x, (fgp.T, f.T, fnn.T), save)

def train(n_data=100, n_data_test=100, n_functions=100,
          nn_arch=[1,20,1], hyper_arch=[30],
          act='tanh', ker='lin', plot='False', save=False,
          lr=0.01, iters=200):

    _, num_weights = shapes_and_num(nn_arch)
    hyper_arch = [2*n_data]+hyper_arch+[num_weights]
    save_name = 'dan-'+str(n_data)+'nf-'+str(n_functions)+"-"+act+ker



if __name__=='__main__':

    n_data, n_data_test = 100, 200
    n_functions = 200
    nn_arch = [1, 20, 20, 1]
    _, num_weights = shapes_and_num(nn_arch)
    hyper_arch = [n_data, 50, 50, num_weights]

    act = 'tanh'; ker = 'lin'

    save_name = 'exp'+str(n_data)+'nf-'+str(n_functions)+"-"+act+ker

    x = np.linspace(-10, 10, n_data).reshape(n_data,1)
    x = np.random.uniform(-10, 10, n_data)[:, None]
    x=np.sort(x,0)
    xt = np.linspace(-10, 10, n_data_test).reshape(n_data_test, 1)
    ys = sample_gpp(x, n_samples=n_functions, kernel=ker)  # [ns, nd]


    plot = True
    if plot: fig, ax = setup_plot()

    def objective(params, t): return hyper_loss(params, x, ys, nn_arch, act)

    def callback(params, t, g):
        y=sample_gpp(x,1, ker)
        preds = hyper_predict(params, x, y, nn_arch, act)  #[1,nd]
        if plot: p.plot_iter(ax, x, x, y, preds)
        cd = np.cov(y.ravel())-np.cov(preds.ravel())
        print("ITER {} | OBJ {} COV DIFF {}".format(t, objective(params, t), cd))

    var_params = adam(grad(objective), init_random_params(hyper_arch),
                      step_size=0.01, num_iters=400, callback=callback)


    xtest = np.linspace(-10, 10, n_data_test).reshape(n_data_test, 1)


    fgps = sample_gpp(x, n_samples=500, kernel=ker)
    ws = nn_predict(var_params, fgps, "relu")  # [ns, nw]
    fs = bnn_predict(ws, x, nn_arch,act)[:, :, 0]
    p.plot_weights_function_space(ws, fs, save_name)
    moments = get_moments(ws)


    # PLOT HYPERNET
    fgp = sample_gpp(x, n_samples=3, kernel=ker)
    fnns = hyper_predict(var_params, xtest,fgp, nn_arch, act)
    p.plot_fs(xtest, fnns,x, fgp, save_name+".pdf")

    #plot_heatmap(moments,"Cov-heatmap"+save_name+'.pdf')
    plot_samples(moments, xtest, 5, nn_arch, act=act, ker=ker, save = save_name+'.pdf')
    p.plot_weights(moments, num_weights, save_name)

