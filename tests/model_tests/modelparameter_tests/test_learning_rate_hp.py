import unittest
from unittest.mock import Mock

import tensorflow as tf
from src.model.modelparameter.hyperparameter_manager import HyperparameterManager
from src.model.modelparameter.learning_rate_hp import LearningRate

class TestCaseLearningRate(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_active_value(self):
        learning_rate = LearningRate(0,Mock())
        self.assertRaises(NotImplementedError,learning_rate.get_active_value)

    def test_get_parameter_value_conf_0(self):
        learning_rate = LearningRate(0,Mock())
        self.assertIsInstance(learning_rate.get_parameter_value(),tf.keras.optimizers.schedules.PiecewiseConstantDecay)

    def test_get_parameter_value_conf_1(self):
        learning_rate = LearningRate(1,Mock())
        self.assertEqual(1e-3,learning_rate.get_parameter_value())
                    
    
    def tearDown(self):
        pass