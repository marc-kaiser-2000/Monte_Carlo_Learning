import json
import os

from src.model.modelparameter.sample_size_test_hp import SampleSizeTest
from src.model.modelparameter.sample_size_valid_hp import SampleSizeValid
from src.model.modelparameter.sample_size_train_hp import SampleSizeTrain
from src.model.modelparameter.batch_size_test_hp import BatchSizeTest
from src.model.modelparameter.batch_size_valid_hp import BatchSizeValid
from src.model.modelparameter.batch_size_train_hp import BatchSizeTrain
from src.model.modelparameter.constants_hp import Constants
from src.model.modelparameter.error_hp import Error
from src.model.modelparameter.gen_data_hp import GenData
from src.model.modelparameter.loss_and_grad_hp import LossAndGrad
from src.model.modelparameter.optimizer_hp import Optimizer
from src.model.modelparameter.model_topology_hp import Model
from src.model.modelparameter.learning_rate_hp import LearningRate

from src.utils.filesystem import FileSystem


class HyperparameterManager:
    """Hyperparameter Manager.
    Initializes required hyperparameter objects based on specified experiment.
    Provides methods for 'LeafModel' and 'RootModel' to access hyperparameters.

    The 'update_buffer_train()' method is overwritten by different heuristics to test different
    learning strategies.
    May be enhanced by further heuristics.
    """

    def __init__(self,experiment_repetition,experiment_name,dim) -> None:

        # Arguments
        self.experiment_repetition = experiment_repetition
        self.experiment_name = experiment_name
        self.dim = dim

        # Hyperparameter Configuration
        self.model_type = None
        self.max_level = None
        self.batch_size_test_conf = None
        self.batch_size_valid_conf = None
        self.sample_size_valid_conf = None
        self.sample_size_test_conf = None
        self.gen_data_conf = None
        self.optimizer_conf = None
        self.learning_rate_conf = None
        self.loss_and_grad_conf = None
        self.model_topology_conf = None
        self.dynamics_conf = None
        self.asset_correlation_conf = None


        # Hyperparameter Instances
        self._batch_size_train_instance = None
        self._batch_size_valid_instance = None
        self._batch_size_test_instance = None
        self._sample_size_train_instance = None
        self._sample_size_valid_instance = None
        self._sample_size_test_instance = None
        self._error_instance = None
        self._gen_data_mc_step_instance = None
        self._optimizer_instance = None
        self._loss_and_grad_instance = None
        self._model_topology_instance = None

        # Buffer Variables
        self.buffer_constants = None
        self.buffer_batch_size_train = None
        self.buffer_batch_size_valid = None
        self.buffer_batch_size_test = None
        self.buffer_sample_size_train = None
        self.buffer_sample_size_valid = None
        self.buffer_sample_size_test = None
        self.buffer_error = None
        self.buffer_gen_data_fnc = None
        self.buffer_mc_step_fnc = None
        self.buffer_optimizer = None
        self.buffer_grad_fnc = None
        self.buffer_model = None
        self.buffer_learning_rate = None


        # Working Variables for Heuristics
        self.transfer_learning = None
        self.buffer_increase_level = None
        self.current_disc_idx = None
        self.cnt_plateau = None


        self._load_run_configurations(experiment_repetition,experiment_name)
        self._check_pretrained_models_exist(experiment_name,dim)
        self._buffer_constants(dim)
        self._initialize_hyperparameter_instances(dim)
        self._set_update_values_fnc()

    def _load_run_configurations(self,experiment_repetition,experiment_name):
        with open(os.path.join(FileSystem.get_cfg_path(),f"run_config_{experiment_repetition}_{experiment_name}.json"),'r') as run_cfg_json:
            run_specific_cfg = json.load(run_cfg_json)

        self.model_type = run_specific_cfg["model_type"]
        self.max_level = run_specific_cfg["max_level"]
        self.batch_size_test_conf = run_specific_cfg["batch_size_test"]
        self.batch_size_valid_conf = run_specific_cfg["batch_size_valid"]
        self.sample_size_valid_conf = run_specific_cfg["sample_size_valid"]
        self.sample_size_test_conf = run_specific_cfg["sample_size_test"]
        self.gen_data_conf = run_specific_cfg["gen_data"]
        self.optimizer_conf = run_specific_cfg["optimizer"]
        self.learning_rate_conf = run_specific_cfg["learning_rate"]
        self.loss_and_grad_conf = run_specific_cfg["loss_and_grad"]
        self.model_topology_conf = run_specific_cfg["model_topology"]
        self.dynamics_conf = run_specific_cfg["dynamics"]
        self.asset_correlation_conf = run_specific_cfg["asset_correlation"]

    def _check_pretrained_models_exist(self,experiment_name,dim):

        self.transfer_learning = True
        for level in range(self.max_level+1):
            if not FileSystem.model_exists(experiment_name,level,dim):
                self.transfer_learning = False

    def _buffer_constants(self,dim):
        self.buffer_constants = Constants(dim,self.asset_correlation_conf)



    def _initialize_hyperparameter_instances(self,dim):

        self._batch_size_train_instance = BatchSizeTrain(self.dynamics_conf,self)
        self._batch_size_valid_instance = BatchSizeValid(self.batch_size_valid_conf,self)
        self._batch_size_test_instance = BatchSizeTest(self.batch_size_test_conf,self)
        self._sample_size_train_instance = SampleSizeTrain(self.dynamics_conf,self)
        self._sample_size_valid_instance = SampleSizeValid(self.sample_size_valid_conf,self)
        self._sample_size_test_instance = SampleSizeTest(self.sample_size_test_conf,self)
        self._error_instance = Error(self.dynamics_conf,self)
        self._gen_data_mc_step_instance =GenData(self.gen_data_conf,self)
        self._optimizer_instance = Optimizer(self.optimizer_conf,self)
        self._loss_and_grad_instance = LossAndGrad(self.loss_and_grad_conf,self)
        self._model_topology_instance = Model(dim,self.model_topology_conf,self)
        self._learning_rate_instance = LearningRate(self.learning_rate_conf,self)


    def _set_update_values_fnc(self):
        if self.transfer_learning:
            self._set_update_values_fnc_transfer()
        else:
            self._set_update_values_fnc_standard()

    def _set_update_values_fnc_standard(self):
        options_standard = {
        "0":self._heuristic_apriori_discretization,
        "1":self._heuristic_static_epochs,
        "2":self._heuristic_static_epochs,
        "3":self._heuristic_apriori_discretization, 
        "4":self._heuristic_semi_apriori_discretization,
        "5":self._heuristic_dynamic_training_plateau,
        "6":self._heuristic_dynamic_training_plateau,
        "7":self._heuristic_dynamic_training_inverted_lr,
        "8":self._heuristic_dynamic_training_plateau,
        "9":self._heuristic_dynamic_training_plateau,
        }

        self.update_buffer_train = options_standard.get(str(self.dynamics_conf))

    def _set_update_values_fnc_transfer(self):
        options_transfer = {
        "0":self._heuristic_apriori_discretization,
        "1":self._heuristic_transfer_dynamic_training_plateau,
        "2":self._heuristic_transfer_dynamic_training_plateau,
        "3":self._heuristic_transfer_dynamic_training_plateau, 
        "4":self._heuristic_transfer_dynamic_training_plateau,
        "5":self._heuristic_transfer_dynamic_training_plateau,
        "6":self._heuristic_transfer_dynamic_training_plateau,
        "7":self._heuristic_transfer_dynamic_training_plateau,
        "8":self._heuristic_transfer_dynamic_training_plateau,
        "9":self._heuristic_transfer_dynamic_training_plateau,
        }

        self.update_buffer_train = options_transfer.get(str(self.dynamics_conf))


    def init_buffer_root(self):
        """Initializes buffer variables required by 'RootModel' object during its initialization.
        """
        self.buffer_mc_step_fnc = self._gen_data_mc_step_instance.get_active_value_mc_step(0)
        self.buffer_sample_size_test = self._sample_size_test_instance.get_parameter_value()
        self.buffer_batch_size_test = self._batch_size_test_instance.get_parameter_value()



    def init_buffer_leaf(self,level):
        """Initializes buffer variables required by 'LeafModel' object during its initialization.

        Args:
            level (int): Level of the respective 'LeafModel' object.
        """
        self.buffer_gen_data_fnc = self._gen_data_mc_step_instance.get_active_value_gen_data(level)
        self.buffer_mc_step_fnc = self._gen_data_mc_step_instance.get_active_value_mc_step(level)
        self.buffer_learning_rate = self._learning_rate_instance.get_parameter_value()
        self.buffer_optimizer = self._optimizer_instance.get_parameter_value(self.buffer_learning_rate)
        self.buffer_grad_fnc = self._loss_and_grad_instance.get_parameter_value()

        if self.transfer_learning:
            self.buffer_model = FileSystem.model_load(self.experiment_name,level,self.dim)
        else:
            self.buffer_model = self._model_topology_instance.get_parameter_value()
        self.buffer_sample_size_valid = self._sample_size_valid_instance.get_active_value(level)
        self.buffer_batch_size_valid = self._batch_size_valid_instance.get_active_value(level)

    def init_buffer_train(self,level):
        """Inizialized buffer variables which are required for training.
        Function must only be used by Leaf Model in 'train()' function before 
        train loop.

        Args:
            level (int): Level of the respective 'LeafModel' object.
        """
        self.buffer_batch_size_train = self._batch_size_train_instance.get_active_value(level,0)
        self.buffer_sample_size_train = self._sample_size_train_instance.get_active_value(level,0)
        self.buffer_error = self._error_instance.get_active_value(level,0)
        self.buffer_increase_level = False

        self.cnt_plateau = 0
        self.current_disc_idx = 0

    def update_buffer_train(self,error_hist,level):
        """Updates buffered variables which are required for training.
        Function must only be used by Leaf Model in 'train()' function in 
        train loop.

        Args:
            error_hist (List[tupel]): Absolute and relative error hist.
            level (int): Level of the respective 'LeafModel' object.

        Raises:
            NotImplementedError: Method always has to be overwritten by a heuristic
        """
        raise NotImplementedError("This method has to be overrriden!")

# ==============================================
# Define Methods for 'options_standard' here
# ==============================================


    def _heuristic_static_epochs(self,error_hist,level):
        err = error_hist[-1]
        if err[0] == 750000:
            self.buffer_increase_level = True
        else:
            self.buffer_increase_level = False


    def _heuristic_apriori_discretization(self,error_hist,level):
        error = error_hist[-1][7]
        self.buffer_increase_level = False

        thresholds = self._error_instance.get_parameter_value()[level]
        for idx,th in enumerate(thresholds):
            if error >= th:
                self.buffer_sample_size_train = self._sample_size_train_instance.get_active_value(level,idx)
                return 0
        self.buffer_increase_level = True
        return 0

    def _heuristic_semi_apriori_discretization(self,error_hist,level):

        self.buffer_increase_level = False

        error_threshold = self._error_instance.get_parameter_value()[level]
        sample_size = self._sample_size_train_instance.get_parameter_value()[level]

        if error_hist[-1][7] < error_threshold:
            self.buffer_increase_level = True
            return 0

        if len(error_hist) == 1:
            return 0

        x_n = error_hist[-2][0]/1000
        y_n = error_hist[-2][7]
        x_N = error_hist[-1][0]/1000
        y_N = error_hist[-1][7]

        delta_now = (y_n - y_N) / (x_n - x_N)

        if delta_now >= 0 and self.current_disc_idx < len(sample_size)-1:
            self.current_disc_idx += 1

        else:
            self.buffer_sample_size_train = self._sample_size_train_instance.get_active_value(level,self.current_disc_idx)


    def _heuristic_dynamic_training_plateau(self,error_hist,level):
        if level == 0:
            iter_plateau = 20
        else:
            iter_plateau = 1

        self.buffer_increase_level = False

        sample_size = self._sample_size_train_instance.get_parameter_value()[level]
        if len(error_hist) == 1:
            return 0

        x_n = error_hist[-2][0]/1000
        y_n = error_hist[-2][7]
        x_N = error_hist[-1][0]/1000
        y_N = error_hist[-1][7]

        delta_now = (y_n - y_N) / (x_n - x_N)

        if delta_now >= 0:
            self.cnt_plateau += 1

        if self.cnt_plateau >= iter_plateau:
            self.cnt_plateau = 0

            if self.current_disc_idx < len(sample_size)-1:
                self.current_disc_idx += 1
                self.buffer_sample_size_train = self._sample_size_train_instance.get_active_value(level,self.current_disc_idx)
            elif self.current_disc_idx == len(sample_size)-1:
                self.current_disc_idx += 1
                self.buffer_learning_rate = 1e-4
            elif self.current_disc_idx == len(sample_size):
                self.current_disc_idx += 1
                self.buffer_learning_rate = 1e-5
            elif self.current_disc_idx == len(sample_size)+1:
                self.current_disc_idx = 0
                self.buffer_learning_rate =1e-3
                self.buffer_increase_level = True


    def _heuristic_dynamic_training_inverted_lr(self,error_hist,level):
        if level == 0:
            iter_plateau = 20
        else:
            iter_plateau = 1

        self.buffer_increase_level = False

        sample_size = self._sample_size_train_instance.get_parameter_value()[level]
        if len(error_hist) == 1:
            return 0

        x_n = error_hist[-2][0]/1000
        y_n = error_hist[-2][7]
        x_N = error_hist[-1][0]/1000
        y_N = error_hist[-1][7]

        delta_now = (y_n - y_N) / (x_n - x_N)

        if delta_now >= 0:
            self.cnt_plateau += 1

        if self.cnt_plateau >= iter_plateau:
            self.cnt_plateau = 0
            self.current_disc_idx += 1

            if self.current_disc_idx % 3 == 0:
                self.buffer_learning_rate = 1e-3

                if (self.current_disc_idx/3) < len(sample_size):
                    self.buffer_sample_size_train = self._sample_size_train_instance.get_active_value(level,(int(self.current_disc_idx/3)))
                else:
                    self.current_disc_idx = 0
                    self.buffer_increase_level = True
            else:
                idx = (self.current_disc_idx % 3) -1
                lrs = [1e-4,1e-5]
                self.buffer_learning_rate = lrs[idx]


# ==============================================
# Define Methods for 'options_transfer' here
# ==============================================


    def _heuristic_transfer_dynamic_training_plateau(self,error_hist,level):

        self.buffer_increase_level = False

        sample_size = self._sample_size_train_instance.get_parameter_value()[level]
        if len(error_hist) == 1:
            return 0

        x_n = error_hist[-2][0]/1000
        y_n = error_hist[-2][7]
        x_N = error_hist[-1][0]/1000
        y_N = error_hist[-1][7]

        delta_now = (y_n - y_N) / (x_n - x_N)

        if delta_now >= 0:

            if self.current_disc_idx < len(sample_size)-1:
                self.current_disc_idx += 1
                self.buffer_sample_size_train = self._sample_size_train_instance.get_active_value(level,self.current_disc_idx)
            elif self.current_disc_idx == len(sample_size)-1:
                self.current_disc_idx += 1

                self.buffer_learning_rate = 1e-4

            elif self.current_disc_idx == len(sample_size):
                self.current_disc_idx += 1
                self.buffer_learning_rate = 1e-5
            elif self.current_disc_idx == len(sample_size)+1:
                self.current_disc_idx = 0
                self.buffer_learning_rate =1e-3
                self.buffer_increase_level = True
    