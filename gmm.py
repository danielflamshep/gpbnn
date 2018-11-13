from sklearn.mixture import GaussianMixture
import numpy as np

def fit_gmm(weights, num_mixtures=5, num_iters=50):
    covs= ['spherical', 'diag', 'tied', 'full']
    params = {}

    for cov_type in covs:
        gmm = GaussianMixture(n_components=num_mixtures,
                              covariance_type=cov_type,
                              max_iter=num_iters)
        gmm.fit(weights)
        mix_params = (gmm.weights_, gmm.means_, gmm.covariances_)
        params[cov_type] = mix_params

    params['emp_full'] = np.mean(weights, axis=0), np.cov(weights.T)
    params['emp_diag'] = np.mean(weights, axis=0), np.std(weights, axis=0)

    return params

def fit_one_gmm(w):
    gmm = GaussianMixture(covariance_type='full', max_iter=200)
    gmm.fit(w)
    mu, cov =gmm.means_, gmm.covariances_
    print(mu.shape, cov.shape)
    return mu[0],cov[0] # [1, nw]


def fitted_moments(weights, num_mixtures=1, iters=100, type='full'):
    gmm = GaussianMixture(n_components=num_mixtures,
                          covariance_type=type,
                          max_iter=iters)
    gmm.fit(weights)
    if num_mixtures > 1:
        return (gmm.weights_, gmm.means_, gmm.covariances)
    else:
        return (gmm.means_, gmm.covariances)

