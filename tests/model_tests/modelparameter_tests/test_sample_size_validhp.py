import unittest

from src.utils.filesystem import FileSystem
from src.model.modelparameter.hyperparameter_manager import HyperparameterManager
from src.model.modelparameter.sample_size_valid_hp import SampleSizeValid

class TestCaseSampleSizeValid(unittest.TestCase):
    def setUp(self):
        dim = 100
        experiment_repetition = 0
        experiment_name = "debug"
        
        FileSystem.init_class()
        self.hyperparameter_manager = HyperparameterManager(
        experiment_repetition=experiment_repetition,
        experiment_name=experiment_name,
        dim = dim)
    
    def test_get_active_value_conf_0(self):
        sample_size_valid = SampleSizeValid(0,self.hyperparameter_manager)
        for level in range(self.hyperparameter_manager.max_level+1):
            self.assertEqual(sample_size_valid.get_active_value(level),1000)

    def test_get_active_value_conf_1(self):
        sample_size_valid = SampleSizeValid(1,self.hyperparameter_manager)
        for level in range(self.hyperparameter_manager.max_level+1):
            self.assertEqual(sample_size_valid.get_active_value(level),320000)

    def test_get_active_value_conf_2(self):
        sample_size_valid = SampleSizeValid(2,self.hyperparameter_manager)
        for level in range(self.hyperparameter_manager.max_level+1):
            self.assertEqual(sample_size_valid.get_active_value(level),1024000)
    
    def test_get_parameter_value(self):
        sample_size_valid = SampleSizeValid(0,self.hyperparameter_manager)
        self.assertEqual(len(sample_size_valid.get_parameter_value()),self.hyperparameter_manager.max_level+1)
        
    
    def tearDown(self):
        self.hyperparameter_manager = None
        FileSystem._delete_outdir()