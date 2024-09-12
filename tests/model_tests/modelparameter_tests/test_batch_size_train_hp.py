import unittest

from src.utils.filesystem import FileSystem
from src.model.modelparameter.hyperparameter_manager import HyperparameterManager
from src.model.modelparameter.batch_size_train_hp import BatchSizeTrain

class TestCaseBatchSizeTrain(unittest.TestCase):
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
        batch_size_train = BatchSizeTrain(0,self.hyperparameter_manager)
        for level in range(self.hyperparameter_manager.max_level+1):
            self.assertEqual(batch_size_train.get_active_value(level,0),10)

    def test_get_active_value_conf_1_4__6_8(self):
        batch_size_train = BatchSizeTrain(1,self.hyperparameter_manager)
        for level in range(self.hyperparameter_manager.max_level+1):
            self.assertEqual(batch_size_train.get_active_value(level,0),8192)

    def test_get_active_value_conf_5__9(self):
        batch_size_train = BatchSizeTrain(5,self.hyperparameter_manager)
        for level in range(self.hyperparameter_manager.max_level+1):
            self.assertEqual(batch_size_train.get_active_value(level,0),50000)

    def test_get_parameter_value(self):
        batch_size_train = BatchSizeTrain(0,self.hyperparameter_manager)
        self.assertEqual(len(batch_size_train.get_parameter_value()),self.hyperparameter_manager.max_level+1)
        
    def tearDown(self):
        self.hyperparameter_manager = None
        FileSystem._delete_outdir()