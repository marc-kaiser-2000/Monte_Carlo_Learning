import tensorflow as tf
from src.model.modelparameter.abstract_hyperparameter import AbstractHyperparameter

class Constants(AbstractHyperparameter):
    """Class implements the "Constants" Hyperparameter.
    This class aggregates all variables required for the 'gen_data' and 'mc_step' functions
    """

    def __init__(self,dim,asset_correlation_conf):

        self.dim = dim

        # Set asset correlation
        asset_correlation_options = {
            0 : 0.25,
            1 : 0.95
        }

        self.asset_cor = asset_correlation_options.get(asset_correlation_conf)

        # Data Type
        self.DTYPE ='float32'

        self.T = tf.constant(1., dtype=self.DTYPE)

        # Risk-free interest rate
        self.r = tf.constant(1./20, dtype=self.DTYPE)

        # Drift
        self.mu = self.r

        # Refinement Cost factor
        self.N = 2

        # Strike price
        self.K = tf.constant(100., dtype=self.DTYPE)

        # Domain-of-interest at t=0
        self.a = 90 * tf.ones((dim), dtype=self.DTYPE)
        self.b = 110 * tf.ones((dim), dtype=self.DTYPE)


        # Diffusion coefficient (volatility)
        #self.sigma = tf.linalg.diag(0.2 + 0.05 * tf.range(1, dim+1, dtype=tf.float32))
        self.sigma = tf.linalg.diag(1./10 + 1./200*tf.range(1, dim+1, dtype=self.DTYPE)) # sigma differs, increasing volatility for bigger dim
        #self.sigma = tf.linalg.diag(1./20 * tf.ones(shape=dim))

        # Asset Correlation
        self.Sig = tf.eye(dim) + self.asset_cor * (tf.ones((dim, dim)) - tf.eye(dim)) # rho differs as well, 0.5 instead of 0.25
        self.C =tf.transpose(tf.linalg.cholesky(self.Sig))



    def get_active_value(self):
        raise NotImplementedError("Class 'Constants' is a container of variables and must be used as a whole!")

    def get_parameter_value(self):
        raise NotImplementedError("Class 'Constants' is a container of variables and must be used as a whole!")
