import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.numpy.linalg import cholesky

from autograd import grad
from autograd.misc import flatten
import autograd.scipy.stats.norm as norm
from autograd.misc.optimizers import adam

from plotting import plot_priors, plot_heatmap, setup_plot, plot_iter, plot_fs, plot_weights
from bnn import shapes_and_num, reshape_weights, bnn_predict, log_pdf_prior, log_like
from nn import nn_predict, map_objective, setup_plot, init_random_params
from gp import sample_gpp, sample_gpp_multi
from hypernn_gp import log_gaussian
from util import sample_inputs

def log_gaussian(x, mu, std):
    return np.sum(norm.logpdf(x, mu, std))

def total_loss(weights, x, y, arch, act):
    f = bnn_predict(weights, x, arch, act)[:, :, 0]
    return -log_gaussian(f, y, 1)

if __name__=='__main__':

    rs = npr.RandomState(2)

    n_data, n_data_test = 15, 200
    n_functions = 100
    nn_arch = [1, 20, 20, 1]
    _, num_weights = shapes_and_num(nn_arch)
    hyper_arch = [n_data, 30, 30, num_weights]

    act = 'sin'; ker = 'per'

    save_name = '-'+str(n_data)+'nf-'+str(n_functions)+"-"+act+ker

    xs, ys = sample_gps()  # [nf, nd]
    print(xs.shape, ys.shape)

    plot = True
    if plot: fig, ax = setup_plot()

    def objective(params, t): return total_loss(params, xs, ys, nn_arch, act)

    int=np.random.randint(n_functions)
    y = ys[10]; x = xs[None,0]
    print(y.shape, x.shape)

    def callback(params, t, g):
        preds = bnn_predict(params, x, nn_arch, act)[:,:,0] #[1,nd]
        #print(preds.shape)
        if plot: plot_iter(ax, x.ravel(), x.ravel(), y, preds[0])
        print("ITER {} | OBJ {}".format(t, objective(params, t)))

    opws = adam(grad(objective), rs.randn(n_functions, num_weights),
                     step_size=0.01, num_iters=200, callback=callback)

    plot_weights(opws, save_name)

    # moments = get_moments(opws)
    # plot_samples(moments, xtest, 5, nn_arch, act=act, ker=ker, save = save_name+'.pdf')

