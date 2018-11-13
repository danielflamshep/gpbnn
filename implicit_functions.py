import numpy as np
import matplotlib.pyplot as plt


def sample_function(x=None, nfuncs=10, ndata=100):
    if x is None: x = np.sort(np.random.uniform(-10, 10, ndata)[:, None], 0)
    xs = np.tile(x, (nfuncs, )).T
    a = np.random.uniform(-10, 10, size=(nfuncs, 1))
    b = np.random.uniform(-2, 2, size=(nfuncs, 1))
    c = np.random.uniform(1, 5, size=(nfuncs, 1))
    return c*np.abs(a+xs)+b



def piecewise_function(inputs):
    pass


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


