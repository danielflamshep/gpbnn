import autograd.numpy as np
from autograd.scipy.special import gamma, yn
from autograd.misc import flatten
import autograd.numpy.random as npr
import inspect
import os

rs = npr.RandomState(0)


def build_toy_dataset(data="expx", n_data=70, noise_std=0.1, D=1):
    bell = lambda x: np.exp(-0.5 * x ** 2)
    cube = lambda x: 0.1 * x * np.sin(x)

    if data == "expx":
        inputs = np.linspace(-2, 2, num=n_data)
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

    elif data == "cubic":
        inputs = np.linspace(-1.2, 1.2, num=n_data)
        targets = inputs**3 + rs.randn(n_data) * noise_std

    elif data == "tb":
        inputs = np.linspace(-3, 3, num=n_data)
        targets = np.tanh(-inputs) + rs.randn(n_data) * noise_std


    inputs = inputs.reshape((len(inputs), D))
    targets = targets.reshape((len(targets), D))
    return inputs, targets

def sample_inputs(type, ndata, range):
    low, high =range
    if type =='gridbox':
        x = np.linspace(low, high, ndata).reshape(ndata,1)
    elif type=="uniform":
        x = np.random.uniform(low, high, size=(ndata,1))
    elif type=='normal':
        x = np.random.randn(ndata, 1)

    return np.sort(x, axis=0)

def covariance(x, xp, kernel_params=0.1 * rs.randn(2)):
    output_scale = np.exp(kernel_params[0])
    length_scales = np.exp(kernel_params[1:])
    diffs = (x[:, None] - xp[None, :]) / length_scales
    cov = output_scale * np.exp(-0.5 * np.sum(diffs ** 2, axis=2))
    return cov


#---------LOADING AND Saving----------#

def get_save_name(n_data, n_functions,act,ker, nn_arch, hyper_arch):
    save_name ="nd"+str(n_data)+'nf-'+str(n_functions)\
               +"-"+act+ker+str(nn_arch)+str(hyper_arch)
    save_dir = os.path.join(os.getcwd(), 'plots', save_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    return save_dir

def manage_and_save(frame, exp, run):

    save_dir = os.path.join(os.getcwd(), 'plots', 'exp{}run{}'.format(exp, run))
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    args, _, _, vals = inspect.getargvalues(frame)
    list = [(i, vals[i]) for i in args]
    file = open("experiment{}run{}".format(exp, run), "w")
    save = ""
    for arg, val in list[:-2]:  # ignore last 2
        str = "{} = {}".format(arg, val)
        file.write(str)
        save.join(str)

    return save, list


rbf = lambda x: np.exp(-x ** 2)
relu = lambda x: np.maximum(x, 0.)
sigmoid = lambda x: 0.5 * (np.tanh(x) ** 2 - 1)
linear = lambda x: x
softplus = lambda x: np.log(1 + np.exp(x))
def logsigmoid(x): return x - np.logaddexp(0, x)


act_dict={
    'rbf': rbf,
    'relu': relu,
    'sigmoid': sigmoid,
    'lin': linear,
    'softplus': softplus,
    'tanh':np.tanh,
    'sin': np.sin,
    'logsigmoid': logsigmoid
}


#KERNEL STUFF


def d(x, xp): return x[:, None] - xp[None, :]
def L2_norm(x, xp): return np.sum(d(x,xp)**2, axis=2)
def L1_norm(x, xp): return np.sum((d(x,xp)**2)**0.5, axis=2)


def kernel_rbf(x, xp, s=1, l=2):
    d = L2_norm(x, xp)
    return s*np.exp(-0.5 * d/l**2)

def kernel_per(x, xp, s = .5, p=25, l=5):
    d = L1_norm(x, xp)/p
    return s*np.exp(-2 * (np.sin(np.pi*d)/l)**2)

def kernel_poly(x, xp, l=2, a=2):
    d = L2_norm(x, xp)
    return (1 + d/a)**l

def kernel_rq(x, xp, alpha=3):
    d = L2_norm(x, xp)
    return 1/(1 + 0.5 * d/alpha)**alpha

def kernel_bm(x, xp, s=1, l=1):
    return s*np.exp(-L1_norm(x,xp)/l)

def kernel_matern(x, xp):
    sd, rho, eta = 1, 1, 1
    d = L1_norm(x, xp)*np.sqrt(2*eta)/rho
    K = sd**2 * (2**(1-eta)/ gamma(eta))
    return K*yn(eta, d)*(d)**eta

def kernel_per_rbf(x, xp):
    return kernel_per(x, xp)*kernel_rbf(x, xp)

def kernel_constant(x,xp, c=1):
    return

def kernel_lin(x, xp, c=0, s=1, h=0):
    x = x.ravel(); xp=xp.ravel()
    return s*(x[:, None]-c)*(xp[None, :]-c)

def kernel_lin_per(x, xp):
    return kernel_lin(x, xp)*kernel_per(x, xp)


kernel_dict = {"rbf": kernel_rbf,
               "per": kernel_per,
               "rq": kernel_rq,
               "lin": kernel_lin,
               "per-rbf": kernel_per_rbf,
               "lin-per": kernel_lin_per,
               "bm": kernel_bm,
               "matern": kernel_matern,
               "poly": kernel_poly
               }
