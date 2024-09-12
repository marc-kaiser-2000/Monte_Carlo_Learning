import unittest
from unittest.mock import Mock

import tensorflow as tf
from src.model.modelparameter.hyperparameter_manager import HyperparameterManager
from src.model.modelparameter.loss_and_grad_hp import LossAndGrad,compute_grad_mse

class TestCaseLossAndGrad(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_active_value(self):
        loss_and_grad = LossAndGrad(0,Mock())
        self.assertRaises(NotImplementedError,loss_and_grad.get_active_value)

    def test_get_parameter_value_conf_0(self):
        loss_and_grad = LossAndGrad(0,Mock())
        self.assertEqual(compute_grad_mse,loss_and_grad.get_parameter_value())

                    
    
    def tearDown(self):
        pass