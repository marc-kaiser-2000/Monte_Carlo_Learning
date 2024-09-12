from src.model.modelparameter.abstract_hyperparameter import AbstractHyperparameter

class BatchSizeTrain(AbstractHyperparameter):
    """Class implements the "Batch Size" Hyperparameter for training data.
    Dictionary is used similar to a straegy pattern, 
    enabling implementation of further heuristics.
    """

    def __init__(self,conf,hyperparameter_manager):
        super().__init__(hyperparameter_manager)

        batch_size_train_options = {
        "0":self._heuristic_debug,
        "1":self._heuristic_constant_8192,
        "2":self._heuristic_constant_8192,
        "3":self._heuristic_constant_8192,
        "4":self._heuristic_constant_8192,
        "5":self._heuristic_constant_50000,
        "6":self._heuristic_constant_8192,
        "7":self._heuristic_constant_8192,
        "8":self._heuristic_constant_8192,
        "9":self._heuristic_constant_50000,
        }
        self._parameter_value = batch_size_train_options.get(str(conf))()

    def get_active_value(self,level,idx):
        return self._parameter_value[level][idx]

    def get_parameter_value(self):
        return self._parameter_value

# =====================================================
# Define Methods for 'batch_size_train_options' here
# =====================================================

    def _heuristic_debug(self):
        out = [[10] for level in range(self.hyperparameter_manager.max_level + 1)]
        return out

    def _heuristic_constant_8192(self):
        out = [[8192] for level in range(self.hyperparameter_manager.max_level + 1)]
        return out

    def _heuristic_constant_50000(self):
        out = [[50000] for level in range(self.hyperparameter_manager.max_level + 1)]
        return out