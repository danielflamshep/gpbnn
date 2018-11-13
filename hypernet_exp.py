import inspect
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.numpy.linalg import cholesky
from autograd import grad
from autograd.misc import flatten
import autograd.scipy.stats.norm as norm
from autograd.misc.optimizers import adam

from util import manage_and_save, get_save_name
import plotting as p
from bnn import shapes_and_num, reshape_weights, bnn_predict, log_pdf_prior, log_like
from nn import nn_predict, setup_plot, init_random_params
from gp import sample_gpp
from hypernn_gp import log_gaussian, get_moments, sample_normal
from prior_reparams import reparameterize
rs = npr.RandomState(2)


def log_gaussian(x, mu, std):
    return np.mean(norm.logpdf(x, mu, std))

def hyper_predict(params, x, xy, nn_arch, nn_act):  # xy shape is [nf, 2*nd]
    weights = nn_predict(params, xy, nn_act)  # [nf, nw]
    #weights = reparameterize(weights, prior='dropout')
    return bnn_predict(weights, x, nn_arch, nn_act)[:, :, 0]  # [nf, nd]

def hyper_loss(hyper_param, x, y, xy, net_arch, act):
    f = hyper_predict(hyper_param, x, xy, net_arch, act)
    return -log_gaussian(f, y, 1)

def sample_inputs(nfuncs, ndata):
    xs = np.random.uniform(-10, 10, size=(nfuncs, ndata))
    xs = np.sort(xs, 1)
    return xs[:,:,None]

def sample_gps(nfuncs, ndata, ker):
    xs = sample_inputs(nfuncs, ndata)
    fgp = [sample_gpp(x, 1, kernel=ker) for x in xs]
    return xs, np.concatenate(fgp, axis=0)

def sample_data(nf, nd, ker):
    xs, ys = sample_gps(nf, nd, ker)
    return xs, ys, np.concatenate((xs[:, :, 0], ys), axis=1)  # [nf,nd,1], [nf, nd], [nf,2*nd]

def plot_samples(params, x, n_samples, layer_sizes, act, ker, save=None):
    "  plots samples of "
    ws = sample_normal(params, n_samples)
    fnn = bnn_predict(ws, x, layer_sizes, act)[:, :, 0]  # [ns, nd]
    fgp = sample_gpp(x, n_samples, ker)

    # functions from a standard normal bnn prior
    wz = sample_normal((np.zeros(ws.shape[1]), np.ones(ws.shape[1])), n_samples)
    f = bnn_predict(wz, x, layer_sizes, act)[:, :, 0]
    p.plot_priors(x, (fgp.T, f.T, fnn.T), save)

def train(n_data=50, n_data_test=100, n_functions=500,
          nn_arch=[1,15,15,1], hyper_arch=[20],
          act='rbf', ker='per',
          lr=0.01, iters=200,
          exp=1, run=1, feed_x=True,
          plot=True, save=False):

    _, num_weights = shapes_and_num(nn_arch)
    hyper_arch = [2*n_data]+hyper_arch+[num_weights]

    xs, ys, xys = sample_data(n_functions, n_data, ker=ker)

    # save_file, args = manage_and_save(inspect.currentframe(),exp,run)
    save_name = 'none-'+str(n_data)+'nf-'+str(n_functions)+"-"+act+ker
    save_name = get_save_name(n_data, n_functions, act, ker, nn_arch, hyper_arch)

    if plot: fig, ax = setup_plot()

    def objective(params, t): return hyper_loss(params, xs, ys, xys, nn_arch, act)

    def callback(params, t, g):
        x, y, xy = sample_data(1, n_data, ker=ker)
        preds = hyper_predict(params, x, xy, nn_arch, act)  # [1, nd]
        if plot: p.plot_iter(ax, x[0], x[0], y, preds)

        # cov_compare = np.cov(y.ravel())-np.cov(preds.ravel())
        print("ITER {} | OBJ {} COV DIFF {}".format(t, objective(params, t), 1))

    var_params = adam(grad(objective), init_random_params(hyper_arch),
                      step_size=lr, num_iters=iters, callback=callback)

    xs, ys, xys = sample_data(10000, n_data, ker=ker)
    ws = nn_predict(var_params, xys, act)  # [ns, nw]
    #ws = reparameterize(wse, prior='dropout')

    fs = bnn_predict(ws, xs, nn_arch, act)[:, :, 0]  # [nf, nd]

    p.plot_weights(ws, save_name)
    #p.plot_weights(ws, 'post'+save_name)
    p.plot_weights_function_space(ws, fs, save_name)
    #p.plot_fs(xs[0], fs[0:3], xs[0], ys[0:3], save_name)

    return ws, var_params


if __name__=='__main__':

    ws, param_ws = train()




