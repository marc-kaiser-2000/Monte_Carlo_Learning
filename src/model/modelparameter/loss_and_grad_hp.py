import tensorflow as tf
from src.model.modelparameter.abstract_hyperparameter import AbstractHyperparameter

class LossAndGrad(AbstractHyperparameter):
    """Class implements the calculation of loss function and the gradient.
    Dictionary is used similar to a straegy pattern, 
    enabling implementation of further heuristics.
    """

    def __init__(self,conf, hyperparameter_manager) -> None:
        super().__init__(hyperparameter_manager)

        grand_and_loss_options = {
        0: _compute_grad_mse,
        }
        self._parameter_value = grand_and_loss_options.get(conf)

    def get_active_value(self):
        raise NotImplementedError("Class 'LossAndGrad' can only initialized and not updated!")

    def get_parameter_value(self):
        return self._parameter_value

# =================================================
# Define Methods for 'grand_and_loss_options' here
# =================================================


@tf.function
def _loss_mse(X, y, model, training=False):
    """Calculates loss with MSE.

    Args:
        X (tf.tensor): Input data.
        y (tf.tensor): True data.
        model (tf.keras.sequantial): Neural network.
        training (bool): Training flag. Defaults to False.

    Returns:
        (tf.tensor): Loss MSE
    """
    #print("Function loss_fn: tracing!")

    y_pred = model(inputs=X, training=training)
    return tf.reduce_mean(tf.math.squared_difference(y, y_pred))

@tf.function
def _compute_grad_mse(X, y, model, training=False):
    """Calculates gradient.

    Args:
        X (tf.tensor): Input data
        y (tf.tensor): True data
        model (tf.keras.sequantial): Neural network
        training (bool): Training flag. Defaults to False.

    Returns:
        loss: Calculated loss
        (tf.GradientTape().gradient): Calculated gradient
    """
    #print("Function compute_grad: tracing!")
    with tf.GradientTape() as tape:
        loss = _loss_mse(X, y, model, training)
    grad = tape.gradient(loss, model.trainable_variables)
    return loss, grad
