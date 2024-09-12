from abc import ABC, abstractmethod

class AbstractHyperparameter(ABC):
    """Abstract Hyperparameter class which must be implemented by specific
    hyperparameter.
    Functions 'get_active_value' and 'get_parameter_value' have to be overwritten
    by specific hyperparameters.

    """
    def __init__(self,hyperparameter_manager) -> None:
        self.hyperparameter_manager = hyperparameter_manager

    @abstractmethod
    def get_active_value(self):
        pass

    @abstractmethod
    def get_parameter_value(self):
        pass
