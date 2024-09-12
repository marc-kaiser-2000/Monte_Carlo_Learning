import unittest 
import json
import os
from src.utils.filesystem import FileSystem



def generate_test_case(dim,experiment_repetition,experiment_name):

    class TemplateTestCase(unittest.TestCase):
        def setUp(self):

            try:
                self.dim = dim
                self.experiment_repetition = experiment_repetition
                self.experiment_name = experiment_name
                FileSystem.init_class()

                with open(os.path.join(FileSystem.get_cfg_path(),f"run_config_{experiment_repetition}_{experiment_name}.json"),'r') as run_cfg_json:
                    self.run_specific_cfg = json.load(run_cfg_json)

            except Exception as e:
                self.skipTest("Setup failed! Subsequent tests would also fail." )

        def _key_exists(self,key):
            return key in self.run_specific_cfg
        
        def _value_int_in_range(self,key,min_value,max_value):

            value = self.run_specific_cfg[key]
            if (min_value <= value) and (value <= max_value):
                return True
            else:
                return False 
            
        def _value_string_model_type(self,key):
            value = self.run_specific_cfg[key]
            if (value == "multi_level") or (value == "single_level"):
                return True
            else:
                return False

        def test_model_type_key(self):
            key = "model_type"
            self.assertTrue(self._key_exists(key))

        def test_model_type_value(self):
            key = "model_type"
            self.assertTrue(self._value_string_model_type(key))

        def test_max_level_key(self):
            key = "max_level"
            self.assertTrue(self._key_exists(key))

        def test_max_level_value(self):
            key = "max_level"
            self.assertTrue(self._value_int_in_range(key,0,10))

        def test_sample_size_test_key(self):
            key = "sample_size_test"
            self.assertTrue(self._key_exists(key))

        def test_sample_size_test_value(self):
            key = "sample_size_test"
            self.assertTrue(self._value_int_in_range(key,0,2))

        def test_sample_size_valid_key(self):
            key = "sample_size_valid"
            self.assertTrue(self._key_exists(key))

        def test_sample_size_valid_value(self):
            key = "sample_size_valid"
            self.assertTrue(self._value_int_in_range(key,0,2))

        def test_batch_size_test_key(self):
            key = "batch_size_test"
            self.assertTrue(self._key_exists(key))

        def test_batch_size_test_value(self):
            key = "batch_size_test"
            self.assertTrue(self._value_int_in_range(key,0,2))

        def test_batch_size_valid_key(self):
            key = "batch_size_valid"
            self.assertTrue(self._key_exists(key))

        def test_batch_size_valid_value(self):
            key = "batch_size_valid"
            self.assertTrue(self._value_int_in_range(key,0,2))

        def test_gen_data_key(self):
            key = "gen_data"
            self.assertTrue(self._key_exists(key))

        def test_gen_data_value(self):
            key = "gen_data"
            self.assertTrue(self._value_int_in_range(key,4,6))

        def test_optimizer_key(self):
            key = "optimizer"
            self.assertTrue(self._key_exists(key))

        def test_optimizer_value(self):
            key = "optimizer"
            self.assertTrue(self._value_int_in_range(key,0,0))

        def test_learning_rate_key(self):
            key = "learning_rate"
            self.assertTrue(self._key_exists(key))

        def test_learning_rate_value(self):
            key = "learning_rate"
            self.assertTrue(self._value_int_in_range(key,0,1))

        def test_loss_and_grad_key(self):
            key = "loss_and_grad"
            self.assertTrue(self._key_exists(key))

        def test_loss_and_grad_value(self):
            key = "loss_and_grad"
            self.assertTrue(self._value_int_in_range(key,0,0))

        def test_model_topology_key(self):
            key = "model_topology"
            self.assertTrue(self._key_exists(key))

        def test_model_topology_value(self):
            key = "model_topology"
            self.assertTrue(self._value_int_in_range(key,0,0))

        def test_asset_correlation_key(self):
            key = "asset_correlation"
            self.assertTrue(self._key_exists(key))

        def test_asset_correlation_value(self):
            key = "asset_correlation"
            self.assertTrue(self._value_int_in_range(key,0,1))

        def test_dynamics_key(self):
            key = "dynamics" 
            self.assertTrue(self._key_exists(key))

        def test_dynamics_value(self):
            key = "dynamics" 
            self.assertTrue(self._value_int_in_range(key,0,9))
            

        def tearDown(self):
            self.hyperparameter_manager = None
            self.dim = None
            self.experiment_repetition = None
            self.experiment_name = None
            self.run_specific_cfg = None
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

class Run_Config_0_SL_Benchmark_L0(generate_test_case(100,"0","SL_Benchmark_L0")):
    pass

class Run_Config_0_SL_Benchmark_L5(generate_test_case(100,"0","SL_Benchmark_L5")):
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




