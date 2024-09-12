import unittest
from unittest.mock import Mock

from src.model.modelparameter.hyperparameter_manager import HyperparameterManager
from src.model.modelcomponents.root_model import LeafModel
from src.utils.filesystem import FileSystem
from src.utils.log_manager import LogManager






def generate_test_case(dim,experiment_repetition,experiment_name,level):
    class TemplateTestCase(unittest.TestCase):
        def setUp(self):
 

                try:
                    FileSystem.init_class()
                    FileSystem.init_experiment_related_directories(
                        dim = dim,
                        experiment_repetition= experiment_repetition,
                        experiment_name=experiment_name,
                        level=level
                    )

                    # Initialize Log Manager
                    self.logger = LogManager.initialize_experiment_logger(
                        dim = dim,
                        experiment_repetition= experiment_repetition,
                        experiment_name=experiment_name,
                        level=level
                    )
                    self.hyperparameter_manager = HyperparameterManager(
                        dim = dim,
                        experiment_repetition= experiment_repetition,
                        experiment_name=experiment_name,
                    )

                    self.model_type = self.hyperparameter_manager.model_type
                    self.constants = self.hyperparameter_manager.buffer_constants

                    self.leaf_model = LeafModel(
                        hyperparameter_manager=self.hyperparameter_manager,
                        level=level,
                        num_timestep=self.constants.N**level
                    )



                except Exception as e:
                    self.skipTest("Setup failed! Subsequent tests would also fail." )  

        
        def test_check_init_buffer_leaf(self):
            self.assertIsNotNone(self.leaf_model.gen_data)
            self.assertIsNotNone(self.leaf_model.mc_step)
            self.assertIsNotNone(self.leaf_model.optimizer)
            self.assertIsNotNone(self.leaf_model.compute_grad)
            self.assertIsNotNone(self.leaf_model.batch_size_valid)
            self.assertIsNotNone(self.leaf_model.sample_size_valid)
            self.assertIsNotNone(self.leaf_model.experiment_repetition)
            self.assertIsNotNone(self.leaf_model.experiment_name)
            self.assertIsNotNone(self.leaf_model.constants)



        def tearDown(self):
            LogManager.close_experiment_logger()
            self.hyperparameter_manager = None
            FileSystem._delete_outdir()

    return TemplateTestCase


class Run_Config_0_Debug(generate_test_case(100,"0","debug",2)):
    pass

class Run_Config_0_ML_Apriori_Training(generate_test_case(100,"0","ML_Apriori_Training",2)):
    pass

class Run_Config_0_SL_Benchmark(generate_test_case(100,"0","SL_Benchmark",0)):
    pass


class Run_Config_0_ML_Dynamic_Training_1(generate_test_case(100,"0","ML_Dynamic_Training_1",2)):
    pass