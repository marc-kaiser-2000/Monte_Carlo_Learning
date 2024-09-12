from decimal import Decimal
from src.model.modelparameter.abstract_hyperparameter import AbstractHyperparameter

class SampleSizeTrain(AbstractHyperparameter):
    """Class implements the "Sample Size" Hyperparameter for training data.
    Dictionary is used similar to a straegy pattern, 
    enabling implementation of further heuristics.
    """


    def __init__(self,conf,hyperparameter_manager) -> None:
        super().__init__(hyperparameter_manager)
        #self._max_sample_size_l0 = 1000
        #self._max_sample_size_ln = 100
        batch_sample_train_options = {
        "0":self._heuristic_debug,
        "1":self._heuristic_static_1,
        "2":self._heuristic_static_1,
        "3":self._heuristic_exponential_increase_static, 
        "4":self._heuristic_exponential_increase_dynamic,
        "5":self._heuristic_exponential_increase_dynamic,
        "6":self._heuristic_exponential_increase_dynamic,
        "7":self._heuristic_exponential_increase_dynamic,
        "8":self._heuristic_exponential_increase_dynamic_light,
        "9":self._heuristic_exponential_increase_dynamic,
        }

        self._parameter_value = batch_sample_train_options.get(str(conf))()


    def get_active_value(self,level,idx):
        return self._parameter_value[level][idx]

    def get_parameter_value(self):
        return self._parameter_value

# =====================================================
# Define Methods for 'batch_sample_train_options' here
# =====================================================


    def _heuristic_debug(self):

        out = [[100] for level in range(self.hyperparameter_manager.max_level +1)]
        return out


    def _heuristic_static_1(self):
        out = [[1] for level in range(self.hyperparameter_manager.max_level +1)]
        return out


    def _heuristic_exponential_increase_static(self):
        out = []
        max_mc = None
        discretization = 5
        for l in range(self.hyperparameter_manager.max_level +1):



            out.append([])
            if l == 0:
                max_mc = 1000
            else:
                max_mc = 100

            if max_mc< 1:
                raise ValueError("Value of 'max_sample_size' must be greater than 1")

            mmc_decimal = Decimal(str(max_mc))
            mmc_digits, mmc_exponent = mmc_decimal.normalize().as_tuple()[1:]
            if len(mmc_digits) == 1 and mmc_digits[0] == 1:
                correction = len(mmc_digits)-1
                out[l] = [10 ** i if i >= 0 else 1 for i in range(mmc_exponent-discretization+correction+1,mmc_exponent+correction)]
                out[l].append(max_mc)
            else:
                correction = len(mmc_digits)
                out[l] = [10 ** i if i >= 0 else 1 for i in range(mmc_exponent-discretization+correction+1,mmc_exponent+correction)]
                out[l].append(max_mc)

        return out

    def _heuristic_exponential_increase_dynamic(self):
        out = []

        for l in range(self.hyperparameter_manager.max_level +1):
            out.append([])
            increasemc = True

            if l == 0:
                max_mc = 1000
            else:
                max_mc = 100
            if max_mc< 1:
                raise ValueError("Value of 'max_sample_size' must be greater than 1")

            i = 0
            while increasemc:

                new_mc = 10**i
                if new_mc < max_mc:
                    out[l].append(new_mc)
                else:
                    out[l].append(max_mc)
                    increasemc = False
                i+= 1
        return out
    
    def _heuristic_exponential_increase_dynamic_light(self):
        out = []
        max_mc = 10

        for l in range(self.hyperparameter_manager.max_level +1):
            out.append([])
            increasemc = True

            if max_mc< 1:
                raise ValueError("Value of 'max_sample_size' must be greater than 1")

            i = 0
            while increasemc:

                new_mc = 10**i
                if new_mc < max_mc:
                    out[l].append(new_mc)
                else:
                    out[l].append(max_mc)
                    increasemc = False
                i+= 1
        return out
    