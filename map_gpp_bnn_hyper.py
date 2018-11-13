import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam, sgd
from util import build_toy_dataset, make_title

from bnn import sample_bnn
from gp import sample_gpp
import plotting
import os
rs = npr.RandomState(0)

if __name__ == '__main__':

    rs = npr.RandomState(0)

    exp_num = 8
    x_num = 400
    samples = 20
    arch = [1, 20, 20, 1]
    act = "rbf"
    kern = "rbf"

    iters_1 = 40
    scale = -0.5
    step = 0.1

    save_plot = True
    save_during = False
    plot_during = True


    if plot_during:
        f, ax = plt.subplots(3, sharex=True)
        plt.ion()
        plt.show(block=False)


    def callback_kl(prior_params, iter, g):
        n_samples, n_data = 3,500
        plot_inputs = np.linspace(-8, 8, num=n_data).reshape(1,n_data)

        f_bnn_gpp = sample_bnn(plot_inputs, n_samples, arch, act, prior_params)
        f_bnn = sample_bnn(plot_inputs, arch, act, n_samples)
        f_gp = sample_gpp(plot_inputs, n_samples)

        # Plot samples of functions from the bnn and gp priors.
        if plot_during:
            for axes in ax: axes.cla()  # clear plots
            # ax.plot(x.ravel(), y.ravel(), 'ko')
            ax[0].plot(plot_inputs, f_gp, color='green')
            ax[1].plot(plot_inputs, f_bnn_gpp, color='red')
            ax[2].plot(plot_inputs, f_bnn, color='blue')
            #ax[0].set_ylim([-5, 5])
            #ax[1].set_ylim([-5, 5])
            #ax[2].set_ylim([-5, 5])

            plt.draw()
            plt.pause(1.0/40.0)

        fs = (f_gp, f_bnn, f_bnn_gpp)
        kl_val = kl(prior_params, iter)

        if save_during:
            title = " iter {} kl {:5}".format(iter, kl_val)
            plotting.plot_priors(plot_inputs, fs, os.path.join(save_dir, title))

        print("Iteration {} KL {} ".format(iter, kl_val))

    # ----------------------------------------------------------------
    # Initialize the variational prior params (phi) HERE for q(w|phi)

    init_var_params = init_bnn_params(num_weights, scale=scale)

    # ---------------------- MINIMIZE THE KL --------------------------

    prior_params = adam(grad_kl, init_var_params,
                        step_size=step, num_iters=iters_1, callback=callback_kl)


    # --------------------- MINIMIZE THE VLB -----------------------------------

    # Set up
    data = 'xsinx'  # or expx or cosx
    iters_2 = 100
    N_data = 70
    inputs, targets = build_toy_dataset(data, n_data=N_data)

    min_vlb = True







