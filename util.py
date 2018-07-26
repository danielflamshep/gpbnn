import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib.image
import os
rs = npr.RandomState(0)

bell = lambda x: np.exp(-0.5*x**2)
cube =lambda x: 0.1*x*np.sin(x)


def build_toy_dataset(data="xsinx", n_data=70, noise_std=0.1):
    D = 1
    if data == "expx":
        inputs = np.linspace(0, 4, num=n_data)
        targets = bell(inputs) + rs.randn(n_data) * noise_std

    elif data == "cosx":
        inputs  = np.concatenate([np.linspace(0, 2, num=n_data/2),
                                  np.linspace(6, 8, num=n_data/2)])
        targets = np.cos(inputs) + rs.randn(n_data) * noise_std
        inputs = (inputs - 4.0) / 4.0

    elif data == "xsinx":
        inputs = np.linspace(0, 8, num=n_data)
        targets = cube(inputs) + rs.randn(n_data) * noise_std

    elif data == "lin":
        inputs = np.linspace(2, 6, num=n_data)
        targets = inputs+ rs.randn(n_data) * noise_std*14

    inputs = inputs.reshape((len(inputs), D))
    targets = targets.reshape((len(targets), D))
    return inputs, targets


def covariance(x, xp, kernel_params=0.1 * rs.randn(2)):
    output_scale = np.exp(kernel_params[0])
    length_scales = np.exp(kernel_params[1:])
    diffs = (x[:, None] - xp[None, :]) / length_scales
    cov = output_scale * np.exp(-0.5 * np.sum(diffs ** 2, axis=2))
    return cov


def make_title(exp_num, x_num, samples, kern,
               arch, act, iters, scale, step):

    train_info = " exp_num {} x_num {} w samples {} " \
                 " kernel {} arch {} act_fn {} " \
                 "iters {} init_scale {} step_size {}".format(
                 exp_num, x_num, samples, kern, arch, act, iters, scale, step)

    return train_info

#---------LOADING AND Saving----------#


def save():
    # ignore
    save_title = make_title(exp_num, x_num, samples, kern, arch, act, iters_1, scale, step)
    save_dir = os.path.join(os.getcwd(), 'plots', 'exp', save_title)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_plot = False
    plot_during_ = True
    save_dir = os.path.join(os.getcwd(), 'plots', 'bnn', "exp-" + str(exp_num) + data)



rbf = lambda x: np.exp(-x ** 2)
relu = lambda x: np.maximum(x, 0.)
sigmoid = lambda x: 0.5 * (np.tanh(x) ** 2 - 1)
linear = lambda x: x
softp = lambda x: np.log(1 + np.exp(x))

act_dict={
    'rbf': rbf,
    'relu': relu,
    'sigmoid': sigmoid,
    'linear': linear,
    'softp': softp,
    'tanh':np.tanh,
    'sin': np.sin
}


#-------------------------------------------#
#KERNEL STUFF


def d(x, xp): return x[:, None] - xp[None, :]


def L2_norm(x, xp):
    return np.sum(d(x,xp)**2, axis=2)


def L1_norm(x, xp):
    return np.sum((d(x,xp)**2)**0.5, axis=2)


def kernel_rbf(x, xp, s=1, l=1):
    d = L2_norm(x, xp)
    return s*np.exp(-0.5 * d/l**2)


def kernel_per(x, xp, s = 1, p=3., l=1):
    d = L1_norm(x, xp)/p
    return s*np.exp(-2 * (np.sin(np.pi*d)/l)**2)


def kernel_rq(x, xp, alpha=3):
    d = L2_norm(x, xp)
    return 1/(1 + 0.5 * d/alpha)**alpha


def kernel_per_rbf(x, xp):
    return kernel_per(x, xp)*kernel_rbf(x, xp)


def kernel_lin(x, xp, c=0, s=1, h=0):
    x=x.ravel(); xp=xp.ravel()
    return s*(x[:, None]-c)*(xp[None, :]-c)


def kernel_lin_per(x, xp):
    return kernel_lin(x, xp)*kernel_per(x, xp)


kernel_dict = {"rbf": kernel_rbf,
               "per": kernel_per,
               "rq": kernel_rq,
               "lin": kernel_lin,
               "per-rbf": kernel_per_rbf,
               "lin-per": kernel_lin_per,
               }
