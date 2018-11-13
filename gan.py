import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd.numpy.linalg import norm
from autograd import grad, jacobian, elementwise_grad
from optimizers import adam_minimax
from util import build_toy_dataset

from gp import sample_gpp
from bnn import sample_bnn, init_var_params
from nn import init_random_params, nn_predict
from util import sample_inputs


def gan_objective(prior_params, d_params, n_data, n_samples, bnn_layer_sizes, act, d_act='tanh'):
    '''estimates V(G, D) = E_p_gp[D(f)] - E_pbnn[D(f)]]'''

    x = sample_inputs('uniform', n_data, (-10, 10))
    fbnns = sample_bnn(prior_params, x, n_samples, bnn_layer_sizes, act)  # [nf, nd]
    fgps = sample_gpp(x, n_samples, 'rbf')  # sample f ~ P_gp(f)

    D_fbnns = nn_predict(d_params, fbnns, d_act)
    D_fgps = nn_predict(d_params, fgps, d_act)
    print(D_fbnns.shape)
    eps = np.random.uniform()
    f = eps*fgps + (1-eps)*fbnns

    def D(function): return nn_predict(d_params, function, 'tanh')

    J = jacobian(D)(f)
    print(J.shape)
    g = elementwise_grad(D)(f)
    print(g.shape)
    pen = 10 * (norm(g, ord=2, axis=1)-1)**2

    return np.mean(D_fgps - D_fbnns + pen)


if __name__ == '__main__':

    n_data = 100
    n_samples = 50

    save_plots = False
    plot_during = True

    bnn_arch = [1, 10, 1]
    dsc_arch = [n_data, 50, 1]
    act='rbf'; ker='rbf'

    def objective(bnn_param, d_param, t):
        return gan_objective(bnn_param, d_param, n_data, n_samples, bnn_arch, act)

    if plot_during:
        f, ax = plt.subplots(3, sharex=True, frameon=False)
        plt.ion()
        plt.show(block=False)

    def callback(bnn_params, dsc_params, iter, gen_gradient, dsc_gradient):
        # Sample functions from priors f ~ p(f)
        n_samples, ndata = 3, 500
        plot_inputs = np.linspace(-8, 8, num=ndata).reshape(ndata,1)
        std_norm_param = init_var_params(bnn_arch, scale_mean=0, scale=1)

        f_bnn_gpp = sample_bnn(bnn_params, plot_inputs, n_samples, bnn_arch, act)
        f_gp      = sample_gpp(plot_inputs, n_samples, ker)
        f_bnn     = sample_bnn(std_norm_param, plot_inputs, n_samples, bnn_arch, act)

        if plot_during:
            for axes in ax: axes.cla()

            # ax.plot(x.ravel(), y.ravel(), 'ko')
            ax[0].plot(plot_inputs, f_gp.T, color='green')
            ax[1].plot(plot_inputs, f_bnn_gpp.T, color='red')
            ax[2].plot(plot_inputs, f_bnn.T, color='blue')
            #ax[0].set_ylim([-3,3])
            #ax[1].set_ylim([-3,3])
            #ax[2].set_ylim([-3,3])

            plt.draw()
            plt.pause(1.0/40.0)

        print("Iteration {} ".format(iter))


    init_gen_params = init_var_params(bnn_arch, scale=-1.5)
    init_dsc_params = init_random_params(dsc_arch)

    # OPTIMIZE
    grad_gan = grad(objective, argnum=(0, 1))

    optimized_params = adam_minimax(grad_gan, init_gen_params, init_dsc_params,
                                    step_size_max=0.001, step_size_min=0.001,
                                    num_iters=200, callback=callback)
