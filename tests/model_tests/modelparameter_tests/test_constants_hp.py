import unittest

import tensorflow as tf
from src.model.modelparameter.constants_hp import Constants

class TestCaseConstants(unittest.TestCase):
    def setUp(self):
        self.dim = 100

    def test_get_active_value(self):
        constants = Constants(self.dim,0)
        self.assertRaises(NotImplementedError,constants.get_active_value)

    def test_get_parameter_value(self):
        constants = Constants(self.dim,0)
        self.assertRaises(NotImplementedError,constants.get_parameter_value)

    def test_init_conf_0(self):
        constants = Constants(self.dim,0)
        self.assertLessEqual(0,constants.asset_cor)
        self.assertLessEqual(constants.asset_cor,1)            

    def test_init_conf_1(self):
        constants = Constants(self.dim,1)
        self.assertLessEqual(0,constants.asset_cor)
        self.assertLessEqual(constants.asset_cor,1)   

    def test_volatility_divergence(self):
        constants = Constants(self.dim,0)
        self.assertLessEqual(constants.sigma[-1][-1],tf.constant(1.0,dtype="float32"))         
    
    def tearDown(self):
        pass