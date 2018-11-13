import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm

def reparameterize(x, prior=None):
    if prior == None:
        return x
    elif prior == 'horseshoe':
        eta = np.tan(.5*np.pi * norm.cdf(x))
        return eta**2*x
    elif prior == 'exp':
        return -np.log(x)
    elif prior == 'laplace':
        return -np.sign(x)*np.log(1-2*x)
    elif prior == 'lognorm':
        return np.exp(x)
    elif prior == 'IG':
        return 1/x
    elif prior == 'dropout':
        return npr.binomial(1,.8,size=x.shape)*x
