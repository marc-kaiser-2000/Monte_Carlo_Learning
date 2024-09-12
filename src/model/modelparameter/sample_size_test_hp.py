from src.model.modelparameter.abstract_hyperparameter import AbstractHyperparameter

class SampleSizeTest(AbstractHyperparameter):
    """Class implements the "Sample Size" Hyperparameter for test data.
    Dictionary is used similar to a straegy pattern, 
    enabling implementation of further heuristics.
    """

    def __init__(self,conf, hyperparameter_manager) -> None:
        super().__init__(hyperparameter_manager)

        batch_sample_test_options = {
        0 : self._heuristic_debug,
        1 : self._heuristic_static_320000,
        2 : self._heuristic_static_1024000,
        }
        self._parameter_value = batch_sample_test_options.get(conf)()


    def get_active_value(self):
        raise NotImplementedError("Class 'SampleSizeTest' can only initialized and not updated!")

    def get_parameter_value(self):
        return self._parameter_value

# ===================================================
# Define Methods for 'batch_sample_test_options' here
# ===================================================

    def _heuristic_debug(self):
        return 100

    def _heuristic_static_320000(self):
        return 320000
    
    def _heuristic_static_1024000(self):
        return 1024000
    