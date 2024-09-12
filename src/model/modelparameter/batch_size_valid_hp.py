from src.model.modelparameter.abstract_hyperparameter import AbstractHyperparameter

class BatchSizeValid(AbstractHyperparameter):
    """Class implements the "Batch Size" Hyperparameter for validation data.
    Dictionary is used similar to a straegy pattern, 
    enabling implementation of further heuristics.
    """

    def __init__(self,conf, hyperparameter_manager) -> None:
        super().__init__(hyperparameter_manager)

        batch_size_valid_options = {
        0 : self._heuristic_debug,
        1 : self._heuristic_static_25000,
        2 : self._heuristic_static_250000,
        }
        self._parameter_value = batch_size_valid_options.get(conf)()

    def get_active_value(self,level):
        return self._parameter_value[level]

    def get_parameter_value(self):
        return self._parameter_value

# =====================================================
# Define Methods for 'batch_size_valid_options' here
# =====================================================

    def _heuristic_debug(self):
        return [100 for l in range(self.hyperparameter_manager.max_level+1)]

    def _heuristic_static_25000(self):
        return [25000 for l in range(self.hyperparameter_manager.max_level+1)]

    def _heuristic_static_250000(self):
        return [250000 for l in range(self.hyperparameter_manager.max_level+1)]
