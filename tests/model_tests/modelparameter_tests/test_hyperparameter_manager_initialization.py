import unittest 

from src.model.modelparameter.hyperparameter_manager import HyperparameterManager
from src.utils.filesystem import FileSystem


def generate_test_case(dim,experiment_repetition,experiment_name):

    class TemplateTestCase(unittest.TestCase):
        def setUp(self):
            try:
                self.dim = dim
                self.experiment_repetition = experiment_repetition
                self.experiment_name = experiment_name
                FileSystem.init_class()
            except Exception as e:
                self.skipTest("Setup failed! Subsequent tests would also fail." )    

        def test_initialize_hyperparameter_manager(self):
            self.hyperparameter_manager = HyperparameterManager(
                experiment_repetition=self.experiment_repetition,
                experiment_name=self.experiment_name,
                dim = self.dim
            )
            

        def tearDown(self):
            self.hyperparameter_manager = None
            self.dim = None
            self.experiment_repetition = None
            self.experiment_name = None
            FileSystem._delete_outdir()
            

    return TemplateTestCase



class Run_Config_0_Debug(generate_test_case(100,"0","debug")):
    pass

class Run_Config_1_Debug(generate_test_case(100,"1","debug")):
    pass                         

class Run_Config_0_Debug2(generate_test_case(100,"0","debug2")):
    pass

class Run_Config_0_ML_Apriori_Training(generate_test_case(100,"0","ML_Apriori_Training")):
    pass

class Run_Config_0_SL_Benchmark(generate_test_case(100,"0","SL_Benchmark")):
    pass

class Run_Config_0_ML_Benchmark(generate_test_case(100,"0","ML_Benchmark")):
    pass

class Run_Config_0_ML_Semi_Apriori_Training(generate_test_case(100,"0","ML_Semi_Apriori_Training")):
    pass

class Run_Config_0_ML_Dynamic_Training_1(generate_test_case(100,"0","ML_Dynamic_Training_1")):
    pass

class Run_Config_0_ML_Dynamic_Training_2(generate_test_case(100,"0","ML_Dynamic_Training_2")):
    pass

class Run_Config_0_ML_Inverted_Training(generate_test_case(100,"0","ML_Inverted_Training")):
    pass


class Run_Config_0_ML_Variance_Reduction_1(generate_test_case(100,"0","ML_Variance_Reduction_1")):
    pass

class Run_Config_1_ML_Variance_Reduction_1(generate_test_case(100,"1","ML_Variance_Reduction_1")):
    pass

class Run_Config_0_ML_Transfer_Control(generate_test_case(100,"0","ML_Transfer_Control")):
    pass

class Run_Config_0_ML_Variance_Reduction_2(generate_test_case(100,"0","ML_Variance_Reduction_2")):
    pass

class Run_Config_0_ML_Euler_Scheme(generate_test_case(100,"0","ML_Euler_Scheme")):
    pass




