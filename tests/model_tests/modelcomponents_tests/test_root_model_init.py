import unittest

from src.model.modelparameter.hyperparameter_manager import HyperparameterManager
from src.model.modelcomponents.root_model import RootModel
from src.utils.filesystem import FileSystem
from src.utils.log_manager import LogManager




def generate_test_case(dim,experiment_repetition,experiment_name):
    class TemplateTestCase(unittest.TestCase):
        def setUp(self):
                self.experiment_repetition = experiment_repetition
                self.experiment_name = experiment_name
                self.dim = dim

                try:
                    FileSystem.init_class()
                    self.hyperparameter_manager = HyperparameterManager(
                        experiment_repetition=self.experiment_repetition,
                        experiment_name=self.experiment_name,
                        dim = self.dim
                    )

                    self.model_type = self.hyperparameter_manager.model_type
                    self.level = self.hyperparameter_manager.max_level
                    

                    self.root_model = RootModel(
                        hyperparameter_manager=self.hyperparameter_manager,
                        level = self.level
                    )

                    self.constants = self.root_model.constants


                except Exception as e:
                    self.skipTest("Setup failed! Subsequent tests would also fail." )  

        def test_count_component_initialization(self):


            count_component = len(self.root_model.components)
            
            if self.model_type == "multi_level":
                self.assertEqual(self.level+1,count_component)
            else:
                self.assertEqual(1,count_component)


        def test_check_component_num_timestep(self):

            for idx,c in enumerate(self.root_model.components):
                if self.model_type == "multi_level":
                    self.assertEqual(self.constants.N**idx,c.num_timestep)
                else:
                    self.assertEqual(self.constants.N**self.level,c.num_timestep)

        def test_check_component_level(self):

            for idx,c in enumerate(self.root_model.components):
                self.assertEqual(idx,c.level)

        def test_check_init_buffer_root(self):
            self.assertIsNotNone(self.root_model.mc_step)
            self.assertIsNotNone(self.root_model.sample_size_test)
            self.assertIsNotNone(self.root_model.batch_size_test)
            
        def tearDown(self):
            LogManager.close_experiment_logger()
            self.hyperparameter_manager = None
            FileSystem._delete_outdir()

    return TemplateTestCase


class Run_Config_0_Debug(generate_test_case(100,"0","debug")):
    pass

class Run_Config_0_ML_Apriori_Training(generate_test_case(100,"0","ML_Apriori_Training")):
    pass

class Run_Config_0_SL_Benchmark(generate_test_case(100,"0","SL_Benchmark")):
    pass


class Run_Config_0_ML_Dynamic_Training_1(generate_test_case(100,"0","ML_Dynamic_Training_1")):
    pass






