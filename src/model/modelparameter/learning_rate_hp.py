import tensorflow as tf
from src.model.modelparameter.abstract_hyperparameter import AbstractHyperparameter

class LearningRate(AbstractHyperparameter):
    """Class implements the "Learning Rate" Hyperparameter.
    Dictionary is used similar to a straegy pattern, 
    enabling implementation of further heuristics.
    """

    def __init__(self,conf, hyperparameter_manager) -> None:
        super().__init__(hyperparameter_manager)

        learning_rate_options = {
        0 : _schedule_pwc,
        1 : _static_0_001,
        }
        self._parameter_value = learning_rate_options.get(conf)()

    def get_active_value(self):
        raise  NotImplementedError("Class 'Optimizer' can only initialized and not updated!")

    def get_parameter_value(self):
        return self._parameter_value

# ================================================
# Define Methods for 'learning_rate_options' here
# ================================================

def _schedule_pwc():
    #learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([250001,400001],[1e-3,1e-4,1e-5])
    return tf.keras.optimizers.schedules.PiecewiseConstantDecay([250001,400001],[1e-3,1e-4,1e-5])

def _static_0_001():
    #learning_rate = 1e-3
    return 1e-3
