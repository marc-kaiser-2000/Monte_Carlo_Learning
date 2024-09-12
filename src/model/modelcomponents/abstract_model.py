from abc import ABC, ABCMeta, abstractmethod


class AbstractModel(ABC):
    """Abstract class for 'RootModel' and 'LeafModel'.
    Similar to composite pattern (component).
    Functions 'train' and 'evaluate' have to be overwritten by the respective classes.
    """

    def __init__(self,hyperparameter_manager,level,num_timestep) -> None:
        self.hyperparameter_manager = hyperparameter_manager
        self.level = level
        self.num_timestep = num_timestep

    @abstractmethod
    def train(self):
        """Train function.
        Must be overwritten.
        Specific implementation in classes which implement this abstract class.
        """
        pass

    @abstractmethod
    def evaluate(self):
        """Evaluate function.
        Must be overwritten.
        Specific implementation in classes which implement this abstract class."""
        pass