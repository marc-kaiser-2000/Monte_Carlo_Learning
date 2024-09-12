import unittest

from src.utils.filesystem import FileSystem
from src.model.modelparameter.hyperparameter_manager import HyperparameterManager
from src.model.modelparameter.sample_size_train_hp import SampleSizeTrain

class TestCaseSampleSizeTrain(unittest.TestCase):
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
        sample_size_train = SampleSizeTrain(0,self.hyperparameter_manager)
        for level in range(self.hyperparameter_manager.max_level+1):
            self.assertEqual(sample_size_train.get_active_value(level,0),100)

    def test_get_active_value_conf_1_2(self):
        sample_size_train = SampleSizeTrain(1,self.hyperparameter_manager)
        for level in range(self.hyperparameter_manager.max_level+1):
            self.assertEqual(sample_size_train.get_active_value(level,0),1)

    def test_get_active_value_conf_3(self):
        sample_size_train = SampleSizeTrain(3,self.hyperparameter_manager)
        for level in range(self.hyperparameter_manager.max_level+1):
            if level == 0:
                max_mc = 1000
            else:
                max_mc = 100

            self.assertEqual(sample_size_train.get_active_value(level,0),1)
            for idx in range(1,4):
                self.assertLessEqual(1,sample_size_train.get_active_value(idx,0))
                self.assertLessEqual(sample_size_train.get_active_value(idx,0),max_mc)

            self.assertEqual(sample_size_train.get_active_value(level,4),max_mc)

    def test_get_active_value_conf_4_9(self):
        sample_size_train = SampleSizeTrain(4,self.hyperparameter_manager)
        for level in range(self.hyperparameter_manager.max_level+1):
            if level == 0:
                max_mc = 1000
            else:
                max_mc = 100
            self.assertEqual(sample_size_train.get_active_value(level,0),1)
            self.assertEqual(sample_size_train.get_active_value(level,-1),max_mc)

    def test_get_parameter_value(self):
        sample_size_train = SampleSizeTrain(0,self.hyperparameter_manager)
        self.assertEqual(len(sample_size_train.get_parameter_value()),self.hyperparameter_manager.max_level+1)
        


                    
    
    def tearDown(self):
        self.hyperparameter_manager = None
        FileSystem._delete_outdir()