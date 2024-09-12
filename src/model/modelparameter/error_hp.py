from decimal import Decimal
from src.model.modelparameter.abstract_hyperparameter import AbstractHyperparameter

class Error(AbstractHyperparameter):
    """Class implements the "Error" Hyperparameter.
    Dictionary is used similar to a straegy pattern, 
    enabling implementation of further heuristics.
    """

    def __init__(self,idx_experiment,hyperparameter_manager) -> None:
        super().__init__(hyperparameter_manager)
        self.constants = hyperparameter_manager.buffer_constants

        error_options = {
        "0":self._heuristic_debug,
        "1":self._heuristic_return_none, 
        "2":self._heuristic_return_none,
        "3":self._heuristic_geometric_series,
        "4":self._heuristic_semi_apriori_discretization,
        "5":self._heuristic_return_none,
        "6":self._heuristic_return_none,
        "7":self._heuristic_return_none,
        "8":self._heuristic_return_none,
        "9":self._heuristic_return_none,
        }
        self._parameter_value = error_options.get(str(idx_experiment))()



    def get_active_value(self,level,idx):
        return self._parameter_value[level][idx]

    def get_parameter_value(self):
        return self._parameter_value


# ========================================
# Define Methods for 'error_options' here
# ========================================


    def _heuristic_debug(self):
        out =[ [10000,10000] for level in range(self.hyperparameter_manager.max_level+1)]
        return out

    def _heuristic_return_none(self):
        return [ [None] for level in range(self.hyperparameter_manager.max_level+1)]


    def _heuristic_geometric_series(self):

        discretization = 5
        total_error = 0.2

        def distribute_geometric_series(total_error,max_level,N):
            # Guarantees sum of individual terms to be < Total error
            return  [total_error * ((N-1)/N) * ((1/N) ** l) for l in range(max_level+1)]

        out = []

        error_dist = distribute_geometric_series(total_error,self.hyperparameter_manager.max_level,self.constants.N)
        for level in range(self.hyperparameter_manager.max_level+1):
            error = error_dist[level]
            dec_error = Decimal(str(error))
            error_digits, error_exponent = dec_error.normalize().as_tuple()[1:]
            correction = len(error_digits)-1
            level_thresholds = [10 ** exp for exp in range(correction+error_exponent+discretization-1,correction+error_exponent,-1)]
            level_thresholds.append(error)
            out.append(level_thresholds)

        return out


    def _heuristic_semi_apriori_discretization(self):
        min_error_l0 = 0.1
        min_error_ln = 0.01

        return [[min_error_l0] if l == 0  else [min_error_ln]  for l in range(self.hyperparameter_manager.max_level+1)]
    