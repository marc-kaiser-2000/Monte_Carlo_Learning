import unittest
from unittest.mock import Mock

import tensorflow as tf
from src.model.modelparameter.model_topology_hp import Model,init_model_blechschmit_2021

class TestCaseModelTopology(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_active_value(self):
        model = Model(dim=100,conf=0,hyperparameter_manager=Mock())
        self.assertRaises(NotImplementedError,model.get_active_value)

    def test_get_parameter_value_conf_0(self):
        model = Model(dim=100,conf=0,hyperparameter_manager=Mock())
        self.assertIsInstance(model.get_parameter_value(),tf.keras.Sequential)

    def tearDown(self):
        pass