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

        def test_buffer_root(self):
            self.hyperparameter_manager.init_buffer_root()

            self.assertIsNotNone(self.hyperparameter_manager.buffer_mc_step_fnc)
            self.assertIsNotNone(self.hyperparameter_manager.buffer_sample_size_test)
            self.assertIsNotNone(self.hyperparameter_manager.buffer_batch_size_test)

            self.buffer_mc_step_fnc = None
            self.buffer_sample_size_test = None
            self.buffer_batch_size_test = None

        def test_buffer_leaf_for_level_0(self):
            self.hyperparameter_manager.init_buffer_leaf(0)

            self.assertIsNotNone(self.hyperparameter_manager.buffer_gen_data_fnc)
            self.assertIsNotNone(self.hyperparameter_manager.buffer_mc_step_fnc)
            self.assertIsNotNone(self.hyperparameter_manager.buffer_learning_rate)
            self.assertIsNotNone(self.hyperparameter_manager.buffer_optimizer)
            self.assertIsNotNone(self.hyperparameter_manager.buffer_grad_fnc)
            self.assertIsNotNone(self.hyperparameter_manager.buffer_model)
            self.assertIsNotNone(self.hyperparameter_manager.buffer_sample_size_valid)
            self.assertIsNotNone(self.hyperparameter_manager.buffer_batch_size_valid)

            self.buffer_gen_data_fnc = None
            self.buffer_mc_step_fnc = None
            self.buffer_learning_rate = None
            self.buffer_optimizer = None
            self.buffer_grad_fnc = None
            self.buffer_model = None
            self.buffer_sample_size_valid = None
            self.buffer_batch_size_valid = None

        def test_buffer_leaf_for_level_3(self):
            self.hyperparameter_manager.init_buffer_leaf(3)

            self.assertIsNotNone(self.hyperparameter_manager.buffer_gen_data_fnc)
            self.assertIsNotNone(self.hyperparameter_manager.buffer_mc_step_fnc)
            self.assertIsNotNone(self.hyperparameter_manager.buffer_learning_rate)
            self.assertIsNotNone(self.hyperparameter_manager.buffer_optimizer)
            self.assertIsNotNone(self.hyperparameter_manager.buffer_grad_fnc)
            self.assertIsNotNone(self.hyperparameter_manager.buffer_model)
            self.assertIsNotNone(self.hyperparameter_manager.buffer_sample_size_valid)
            self.assertIsNotNone(self.hyperparameter_manager.buffer_batch_size_valid)

            self.buffer_gen_data_fnc = None
            self.buffer_mc_step_fnc = None
            self.buffer_learning_rate = None
            self.buffer_optimizer = None
            self.buffer_grad_fnc = None
            self.buffer_model = None
            self.buffer_sample_size_valid = None
            self.buffer_batch_size_valid = None


        def test_init_buffer_train_for_level_0(self):
            self.hyperparameter_manager.init_buffer_train(0)

            self.assertIsNotNone(self.hyperparameter_manager.buffer_batch_size_train)
            self.assertIsNotNone(self.hyperparameter_manager.buffer_sample_size_train)
            #self.assertIsNotNone(self.hyperparameter_manager.buffer_error)
            self.assertEqual(self.hyperparameter_manager.buffer_increase_level,False)
            self.assertEqual(self.hyperparameter_manager.cnt_plateau,0)
            self.assertEqual(self.hyperparameter_manager.current_disc_idx,0)

            self.hyperparameter_manager.buffer_batch_size_train = None
            self.hyperparameter_manager.buffer_sample_size_train = None
            #self.hyperparameter_manager.buffer_error = None
            self.hyperparameter_manager.buffer_increase_level = None
            self.hyperparameter_manager.cnt_plateau = None
            self.hyperparameter_manager.current_disc_idx = None

        def test_init_buffer_train_for_level_3(self):
            self.hyperparameter_manager.init_buffer_train(3)

            self.assertIsNotNone(self.hyperparameter_manager.buffer_batch_size_train)
            self.assertIsNotNone(self.hyperparameter_manager.buffer_sample_size_train)
            #self.assertIsNotNone(self.hyperparameter_manager.buffer_error)
            self.assertEqual(self.hyperparameter_manager.buffer_increase_level,False)
            self.assertEqual(self.hyperparameter_manager.cnt_plateau,0)
            self.assertEqual(self.hyperparameter_manager.current_disc_idx,0)

            self.hyperparameter_manager.buffer_batch_size_train = None
            self.hyperparameter_manager.buffer_sample_size_train = None
            #self.hyperparameter_manager.buffer_error = None
            self.hyperparameter_manager.buffer_increase_level = None
            self.hyperparameter_manager.cnt_plateau = None
            self.hyperparameter_manager.current_disc_idx = None
        
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



