import unittest
import os
import sys
from src.utils.filesystem import FileSystem




class Initialization_Execution(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_initialization(self):
        FileSystem.init_class()
        FileSystem._delete_outdir()
            
    def tearDown(self):
        pass


class Initialization_Properties(unittest.TestCase):
    def setUp(self):
        try:
            FileSystem.init_class()
        except Exception as e:
            self.skipTest("Setup failed! Subsequent tests would also fail." )  
        
    def test_root_directory_exists(self):
        self.assertTrue(os.path.exists(FileSystem._outdir))

    def test_depth_1_directories_exist(self):
        self.assertTrue(os.path.exists(FileSystem._out_model_runner_path))
        self.assertTrue(os.path.exists(FileSystem._out_analysis_runner_path))
        self.assertTrue(os.path.exists(FileSystem._tensorboard_path))

    def test_depth_2_directories_model_runner_exist(self):
        self.assertTrue(os.path.exists(FileSystem._analytics_path))
        self.assertTrue(os.path.exists(FileSystem._validation_data_path))
        self.assertTrue(os.path.exists(FileSystem._test_data_path))
        self.assertTrue(os.path.exists(FileSystem._model_path))
        
    def test_depth_2_directories_paths_clear(self):
        self.assertFalse(os.listdir(FileSystem._model_path))
        self.assertFalse(os.listdir(FileSystem._analytics_path))

    def test_initialized_flag_set(self):
        self.assertTrue(FileSystem._outdir_initialized)

                
    def tearDown(self):
        FileSystem._delete_outdir()


class Initialization_Depth_2(unittest.TestCase):
    def setUp(self):
        try:
            FileSystem.init_class()
            self.level = 3
        except Exception as e:
            self.skipTest("Setup failed! Subsequent tests would also fail." )  
        
    def test_initialization_depth_2_analysis_runner(self):
        dimensions = [10,50,100]
        FileSystem._init_outdir_d2_analysis_runner(level=self.level)

        for level in range(self.level+1):
            self.assertTrue(os.path.exists(FileSystem._out_analysis_runner_path+FileSystem._nomenclature_d2_analysis_runner.format(level)))
           
                
    def tearDown(self):
        FileSystem._delete_outdir()

    


class Initialization_Depth_3(unittest.TestCase):
    def setUp(self):
        try:
            FileSystem.init_class()
        except Exception as e:
            self.skipTest("Setup failed! Subsequent tests would also fail." )  
        
    def test_initialization_depth_3_model_runner(self):
        dimensions = [10,50,100]
        FileSystem.init_dimension_related_directories(dimensions)

        for dim in dimensions:
            self.assertTrue(os.path.exists(FileSystem._analytics_path+FileSystem._nomenclature_d3_model_runner.format(dim)))
            self.assertTrue(os.path.exists(FileSystem._validation_data_path+FileSystem._nomenclature_d3_model_runner.format(dim)))
            self.assertTrue(os.path.exists(FileSystem._test_data_path+FileSystem._nomenclature_d3_model_runner.format(dim)))
            self.assertTrue(os.path.exists(FileSystem._model_path+FileSystem._nomenclature_d3_model_runner.format(dim)))

                
    def tearDown(self):
        FileSystem._delete_outdir()


class Initialization_Depth_4(unittest.TestCase):

    def setUp(self):
        try:
            FileSystem.init_class()
            self.dimensions = [10,50,100]
            self.experiment_repetition = "0"
            self.experiment_name = "test"
            self.level = 3
            FileSystem.init_dimension_related_directories(self.dimensions)

        except Exception as e:
            self.skipTest("Setup failed! Subsequent tests would also fail." )  
        
    def test_initialization_depth_4_model_runner(self):
        dim = self.dimensions[0]
        FileSystem.init_experiment_related_directories(dim,self.experiment_repetition,self.experiment_name,self.level)

        for l in range(self.level+1):
            self.assertTrue(os.path.exists(FileSystem._analytics_path+
                        FileSystem._nomenclature_d3_model_runner.format(dim)+
                        FileSystem._nomenclature_d4_model_runner_leaf.format(l,self.experiment_repetition,self.experiment_name)))
        self.assertTrue(os.path.exists(FileSystem._analytics_path+
                    FileSystem._nomenclature_d3_model_runner.format(dim)+
                    FileSystem._nomenclature_d4_model_runner_root.format(self.level,self.experiment_repetition,self.experiment_name)))

        
                
    def tearDown(self):
        FileSystem._delete_outdir()

