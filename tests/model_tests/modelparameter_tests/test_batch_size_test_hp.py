import unittest
from unittest.mock import Mock

from src.utils.filesystem import FileSystem
from src.model.modelparameter.hyperparameter_manager import HyperparameterManager
from src.model.modelparameter.batch_size_test_hp import BatchSizeTest

class TestCaseBatchSizeTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_get_active_value(self):
        batch_size_test = BatchSizeTest(0,Mock())
        self.assertRaises(NotImplementedError,batch_size_test.get_active_value)

    def test_get_parameter_value_conf_0(self):
        batch_size_test = BatchSizeTest(0,Mock())
        self.assertEqual(100,batch_size_test.get_parameter_value())

    def test_get_parameter_value_conf_1(self):
        batch_size_test = BatchSizeTest(1,Mock())
        self.assertEqual(25000,batch_size_test.get_parameter_value())

    def test_get_parameter_value_conf_2(self):
        batch_size_test = BatchSizeTest(2,Mock())
        self.assertEqual(250000,batch_size_test.get_parameter_value())

    def tearDown(self):
        pass
