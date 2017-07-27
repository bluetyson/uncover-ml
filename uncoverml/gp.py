import logging
import numpy as np
import tensorflow as tf
# from scipy.cluster.vq import kmeans

from sklearn.cluster import MiniBatchKMeans
from GPflow.svgp import SVGP
from GPflow.likelihoods import MultiClass
from GPflow.kernels import RBF, Matern32, Matern52, ArcCosine, Linear

# Allow multiple GPU sessions
import GPflow
GPflow._settings.settings.session.update(gpu_options=dict(allow_growth=True))


log = logging.getLogger(__name__)

kernmap = {
    'RBF': RBF,
    'Matern32': Matern32,
    'Matern52': Matern52,
    'ArcCosine': ArcCosine,
    'Linear': Linear
}


class SparseGPC:

    def __init__(self, kernel='RBF', n_inducing=200, fix_inducing=False,
                 maxiter=1000, minibatch_size=None, random_state=None,
                 **kargs):
        self.kernel = kernel
        self.kargs = kargs
        self.n_inducing = n_inducing
        self.fix_inducing = fix_inducing
        self.maxiter = maxiter
        self.minibatch_size = minibatch_size
        self.random_state = random_state

    def fit(self, X, y):
        log.info("Initialising inducing points.")
        init_size = 3 * max(100, self.n_inducing)
        km = MiniBatchKMeans(n_clusters=self.n_inducing,
                             random_state=self.random_state,
                             init_size=init_size)
        Z = km.fit(X).cluster_centers_

        # Make the likelihood
        K = len(set(y))
        like = MultiClass(K)

        # Make the kernels
        D = X.shape[1]
        kern = kernmap[self.kernel](input_dim=D, **self.kargs)

        # Make the GP
        self.gp = SVGP(X=X, Y=y, kern=kern, likelihood=like, Z=Z,
                       num_latent=K, q_diag=False,
                       minibatch_size=self.minibatch_size)

        # Optimize
        log.info("Optimising hyperparameters.")
        self.gp.Z.fixed = True
        self.optimize()

        if not self.fix_inducing:
            log.info("Optimising inducing points.")
            self.gp.Z.fixed = False
            self.optimize()

        log.info("Kernel: {}".format(self.gp.kern))

        return self

    def predict(self, X):
        p = self.predict_proba(X)
        y = np.argmax(p, axis=1)
        return y

    def predict_proba(self, X):
        p, _ = self.gp.predict_y(X)
        return p

    def optimize(self):
        if self.minibatch_size:
            self.gp.optimize(method=tf.train.AdamOptimizer(),
                             maxiter=self.maxiter)
        else:
            self.gp.optimize(maxiter=self.maxiter)

    def __repr__(self):
        repre = (
            "{}(kernel={}, n_inducing={}, fix_inducing={}, maxiter={},"
            "minibatch_size={}, random_state={}, {})"
            .format(self.__class__.__name__, self.kernel, self.lenscale,
                    self.ARD, self.n_inducing, self.fix_inducing, self.maxiter,
                    self.minibatch_size, self.random_state, **self.kargs)
        )
        return repre
