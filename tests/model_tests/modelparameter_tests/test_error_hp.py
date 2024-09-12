import unittest

from src.utils.filesystem import FileSystem
from src.model.modelparameter.hyperparameter_manager import HyperparameterManager
from src.model.modelparameter.error_hp import Error

class TestCaseError(unittest.TestCase):
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
        error = Error(0,self.hyperparameter_manager)
        for level in range(self.hyperparameter_manager.max_level+1):
            self.assertEqual(error.get_active_value(level,0),10000)
            self.assertEqual(error.get_active_value(level,1),10000)

    def test_get_active_value_conf_1_2__5_9(self):
        error = Error(1,self.hyperparameter_manager)
        for level in range(self.hyperparameter_manager.max_level+1):
            self.assertIsNone(error.get_active_value(level,0))

    def test_get_active_value_conf_3(self):
        error = Error(3,self.hyperparameter_manager)
        total = 0
        for level in range(self.hyperparameter_manager.max_level+1):
            total +=error.get_active_value(level,-1)
        self.assertAlmostEqual(total,0.2,1)
    
    def test_get_active_value_conf_4(self):
        error = Error(4,self.hyperparameter_manager)
        for level in range(self.hyperparameter_manager.max_level+1):
            if level == 0:
                min_error = 0.1
            else:
                min_error = 0.01
            self.assertEqual(error.get_active_value(level,0),min_error)

    def test_get_parameter_value(self):
        error = Error(3,self.hyperparameter_manager)
        self.assertEqual(len(error.get_parameter_value()),self.hyperparameter_manager.max_level+1)
        
    def tearDown(self):
        self.hyperparameter_manager = None
        FileSystem._delete_outdir()