import unittest
from unittest.mock import Mock

import tensorflow as tf
from src.model.modelparameter.optimizer_hp import Optimizer

class TestCaseOptimizer(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_active_value(self):
        optimizer = Optimizer(0,Mock())
        self.assertRaises(NotImplementedError,optimizer.get_active_value)

    def test_get_parameter_value_conf_0(self):
        optimizer = Optimizer(0,Mock())
        self.assertIsInstance(optimizer.get_parameter_value(1e-3),tf.keras.optimizers.Adam)

    def tearDown(self):
        pass