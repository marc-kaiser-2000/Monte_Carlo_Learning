import unittest
from unittest.mock import Mock

from src.utils.filesystem import FileSystem
from src.model.modelparameter.hyperparameter_manager import HyperparameterManager
from src.model.modelparameter.sample_size_test_hp import SampleSizeTest

class TestCaseSampleSizeTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_active_value(self):
        sample_size_test = SampleSizeTest(0,Mock())
        self.assertRaises(NotImplementedError,sample_size_test.get_active_value)

    def test_get_parameter_value_conf_0(self):
        sample_size_test = SampleSizeTest(0,Mock())
        self.assertEqual(100,sample_size_test.get_parameter_value())

    def test_get_parameter_value_conf_1(self):
        sample_size_test = SampleSizeTest(1,Mock())
        self.assertEqual(320000,sample_size_test.get_parameter_value())

    def test_get_parameter_value_conf_2(self):
        sample_size_test = SampleSizeTest(2,Mock())
        self.assertEqual(1024000,sample_size_test.get_parameter_value())

    def tearDown(self):
        pass
