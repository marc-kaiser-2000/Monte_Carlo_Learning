import tensorflow as tf
from src.model.modelparameter.abstract_hyperparameter import AbstractHyperparameter

class Optimizer(AbstractHyperparameter):
    """Class implements the "Optimizer" Hyperparameter.
    Dictionary is used similar to a straegy pattern, 
    enabling implementation of further heuristics.
    """

    def __init__(self,conf, hyperparameter_manager) -> None:
        super().__init__(hyperparameter_manager)

        optimizer_options = {
        0 : _adam
        }
        self._parameter_value = optimizer_options.get(conf)

    def get_active_value(self):
        raise  NotImplementedError("Class 'Optimizer' can only initialized and not updated!")

    def get_parameter_value(self,learning_rate):
        return self._parameter_value(learning_rate)

# ==============================================
# Define Methods for 'optimizer_options' here
# ==============================================

def _adam(learning_rate):
    #learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([250001,400001],[1e-3,1e-4,1e-5])
    return tf.keras.optimizers.Adam(learning_rate=learning_rate,epsilon=1e-8)
