import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
from autograd.misc.optimizers import adam
from autograd import grad, elementwise_grad
from autograd.scipy.stats import multivariate_normal as mvn
import autograd.scipy.stats.norm as norm
rs = npr.RandomState(0)

egrad=elementwise_grad

def objective(p):
    return norm.cdf(p)

x=np.linspace(0,7,5)
g=grad(objective)(1.0)
eg=egrad(objective)
print(eg(x),norm.pdf(x))

print(g-norm.pdf(1.0))
