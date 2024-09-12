import unittest 

from src.model.modelparameter.hyperparameter_manager import HyperparameterManager
from src.utils.filesystem import FileSystem


def generate_test_case(dim,experiment_repetition,experiment_name):

    class TemplateTestCase(unittest.TestCase):
        def setUp(self):
            try:
                FileSystem.init_class()
                self.hyperparameter_manager = HyperparameterManager(
                    experiment_repetition=experiment_repetition,
                    experiment_name=experiment_name,
                    dim = dim
                )
            except Exception as e:
                self.skipTest("Setup failed! Subsequent tests would also fail." )       

        def test_batch_size_train_instance_exists(self):
            self.assertIsNotNone(self.hyperparameter_manager._batch_size_train_instance)

        def test_batch_size_valid_instance_exists(self):
            self.assertIsNotNone(self.hyperparameter_manager._batch_size_valid_instance)

        def test_batch_size_test_instance_exists(self):
            self.assertIsNotNone(self.hyperparameter_manager._batch_size_test_instance)

        def test_sample_size_train_instance_exists(self):
            self.assertIsNotNone(self.hyperparameter_manager._sample_size_train_instance)

        def test_sample_size_valid_instance_exists(self):
            self.assertIsNotNone(self.hyperparameter_manager._sample_size_valid_instance)

        def test_sample_size_test_instance_exists(self):
            self.assertIsNotNone(self.hyperparameter_manager._sample_size_test_instance)

        def test_error_instance_exists(self):
            self.assertIsNotNone(self.hyperparameter_manager._error_instance)

        def test_gen_data_mc_step_instance_exists(self):
            self.assertIsNotNone(self.hyperparameter_manager._gen_data_mc_step_instance)

        def test_optimizer_instance_exists(self):
            self.assertIsNotNone(self.hyperparameter_manager._optimizer_instance)

        def test_loss_and_grad_instance_exists(self):
            self.assertIsNotNone(self.hyperparameter_manager._loss_and_grad_instance)

        def test_model_topology_instance_exists(self):
            self.assertIsNotNone(self.hyperparameter_manager._model_topology_instance)

        def test_learning_rate_instance_exists(self):
            self.assertIsNotNone(self.hyperparameter_manager._learning_rate_instance)

        
        def tearDown(self):
            self.hyperparameter_manager = None
            FileSystem._delete_outdir()

    return TemplateTestCase



class Run_Config_0_Debug(generate_test_case(100,"0","debug")):
    pass

class Run_Config_1_Debug(generate_test_case(100,"1","debug")):
    pass                         

class Run_Config_0_Debug2(generate_test_case(100,"0","debug2")):
    pass

class Run_Config_0_ML_Apriori_Training(generate_test_case(100,"0","ML_Apriori_Training")):
    pass

class Run_Config_0_SL_Benchmark(generate_test_case(100,"0","SL_Benchmark")):
    pass

class Run_Config_0_ML_Benchmark(generate_test_case(100,"0","ML_Benchmark")):
    pass

class Run_Config_0_ML_Semi_Apriori_Training(generate_test_case(100,"0","ML_Semi_Apriori_Training")):
    pass

class Run_Config_0_ML_Dynamic_Training_1(generate_test_case(100,"0","ML_Dynamic_Training_1")):
    pass

class Run_Config_0_ML_Dynamic_Training_2(generate_test_case(100,"0","ML_Dynamic_Training_2")):
    pass

class Run_Config_0_ML_Inverted_Training(generate_test_case(100,"0","ML_Inverted_Training")):
    pass


class Run_Config_0_ML_Variance_Reduction_1(generate_test_case(100,"0","ML_Variance_Reduction_1")):
    pass

class Run_Config_1_ML_Variance_Reduction_1(generate_test_case(100,"1","ML_Variance_Reduction_1")):
    pass

class Run_Config_0_ML_Transfer_Control(generate_test_case(100,"0","ML_Transfer_Control")):
    pass

class Run_Config_0_ML_Variance_Reduction_2(generate_test_case(100,"0","ML_Variance_Reduction_2")):
    pass

class Run_Config_0_ML_Euler_Scheme(generate_test_case(100,"0","ML_Euler_Scheme")):
    pass




