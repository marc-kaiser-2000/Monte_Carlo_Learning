import tensorflow as tf
from src.model.modelparameter.abstract_hyperparameter import AbstractHyperparameter

class Model(AbstractHyperparameter):
    """Class implements the "Model Topology" Hyperparameter.
    Dictionary is used similar to a straegy pattern, 
    enabling implementation of further heuristics.
    """


    def __init__(self,dim,conf, hyperparameter_manager) -> None:
        super().__init__(hyperparameter_manager)
        self.dim = dim

        model_init_options = {
        0 : _init_model_blechschmit_2021,
        }
        self._parameter_value = model_init_options.get(conf)


    def get_active_value(self):
        raise NotImplementedError("Class 'Model' can only initialized and not updated!")

    def get_parameter_value(self):
        return self._parameter_value(self.dim)

# ==============================================
# Define Methods for 'model_init_options' here
# ==============================================

def _init_model_blechschmit_2021(dim):

    activation='tanh'
    num_hidden_neurons=400
    num_hidden_layers=4
    initializer=tf.keras.initializers.GlorotUniform()


    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(dim,)))
    model.add(tf.keras.layers.BatchNormalization(epsilon=1e-6))
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_hidden_neurons,
                                activation=None,
                                use_bias=False,
                                kernel_initializer=initializer
                                ))
        model.add(tf.keras.layers.BatchNormalization(epsilon=1e-6))
        model.add(tf.keras.layers.Activation(activation))

    model.add(tf.keras.layers.Dense(1,
                                    activation=None,
                                    use_bias=False,
                                    kernel_initializer=initializer
                                    ))
    model.add(tf.keras.layers.BatchNormalization(epsilon=1e-6))

    return model
