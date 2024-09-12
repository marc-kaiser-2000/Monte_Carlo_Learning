import unittest
import os 
import tensorflow as tf

from src.utils.filesystem import FileSystem

class KeepValidationTensorFunctions(unittest.TestCase):

    def setUp(self):
        try:
            FileSystem.init_class(keep_data=True)
            self.dimensions = [100]
            self.experiment_repetition = "0"
            self.experiment_name = "test_experiment"
            self.dtype = "float32"
            self.path_type = "valid"
            self.level = 3
            self.tensor_X =  tf.constant([5, 5, 5, 5, 5],dtype=self.dtype)
            self.tensor_Y =  tf.constant([1, 1, 1, 1, 1],dtype=self.dtype)


            FileSystem.init_dimension_related_directories(self.dimensions)
            FileSystem.init_experiment_related_directories(dim = self.dimensions[0],
                                                           experiment_name=self.experiment_name,
                                                           experiment_repetition=self.experiment_repetition,
                                                           level=self.level)
            
            self.x_path,self.y_path = FileSystem._test_or_validation(self.path_type,self.dimensions[0],self.experiment_repetition,self.experiment_name,self.level)
    
        except Exception as e:
            self.skipTest("Setup failed! Subsequent tests would also fail." )  
        
    def test_tensor_export(self):
        FileSystem.tensor_data_export(experiment_repetition=self.experiment_repetition,
                                    experiment_name=self.experiment_name,
                                    X=self.tensor_X,
                                    Y=self.tensor_Y,
                                    level=self.level,
                                    path_type=self.path_type,
                                    dim=self.dimensions[0]
                                    )

        x_exists = os.path.exists(self.x_path)
        y_exists = os.path.exists(self.y_path)
        
        self.assertTrue(x_exists)
        self.assertTrue(y_exists)

        os.remove(self.x_path)
        os.remove(self.y_path)

    def test_tensor_load(self):
        FileSystem.tensor_data_export(experiment_repetition=self.experiment_repetition,
                                    experiment_name=self.experiment_name,
                                    X=self.tensor_X,
                                    Y=self.tensor_Y,
                                    level=self.level,
                                    path_type=self.path_type,
                                    dim=self.dimensions[0],
                                    )
        
        X,Y = FileSystem.tensor_data_load(experiment_repetition=self.experiment_repetition,
                                    experiment_name=self.experiment_name,
                                    level=self.level,
                                    path_type=self.path_type,
                                    dim=self.dimensions[0],
                                    dtype=self.dtype
        )
        x_tensor_equal = tf.math.reduce_all(tf.equal(X,self.tensor_X))
        y_tensor_equal = tf.math.reduce_all(tf.equal(Y,self.tensor_Y))
        
        self.assertTrue(x_tensor_equal)
        self.assertTrue(y_tensor_equal)

        os.remove(self.x_path)
        os.remove(self.y_path)


    def test_tensor_data_exists_successful(self):

        FileSystem.tensor_data_export(experiment_repetition=self.experiment_repetition,
                            experiment_name=self.experiment_name,
                            X=self.tensor_X,
                            Y=self.tensor_Y,
                            level=self.level,
                            path_type=self.path_type,
                            dim=self.dimensions[0],
                            )
        
        tensors_exists = FileSystem.tensor_data_exists(experiment_name=self.experiment_name,
                            experiment_repetition=self.experiment_repetition,
                            level=self.level,
                            path_type=self.path_type,
                            dim=self.dimensions[0],
        )
        
        self.assertTrue(tensors_exists)

        os.remove(self.x_path)
        os.remove(self.y_path)

    def test_tensor_data_exists_unsuccessful_X_Y_missing(self):
        FileSystem.tensor_data_export(experiment_repetition=self.experiment_repetition,
                            experiment_name=self.experiment_name,
                            X=self.tensor_X,
                            Y=self.tensor_Y,
                            level=self.level,
                            path_type=self.path_type,
                            dim=self.dimensions[0],
                            )
        
        os.remove(self.x_path)
        os.remove(self.y_path)
        
        tensors_exists = FileSystem.tensor_data_exists(experiment_name=self.experiment_name,
                            experiment_repetition=self.experiment_repetition,
                            level=self.level,
                            path_type=self.path_type,
                            dim=self.dimensions[0],
        )
        
        self.assertFalse(tensors_exists)


        
    def test_tensor_data_exists_unsuccessful_X_missing(self):
        FileSystem.tensor_data_export(experiment_repetition=self.experiment_repetition,
                            experiment_name=self.experiment_name,
                            X=self.tensor_X,
                            Y=self.tensor_Y,
                            level=self.level,
                            path_type=self.path_type,
                            dim=self.dimensions[0],
                            )
        
        os.remove(self.x_path)
        
        
        tensors_exists = FileSystem.tensor_data_exists(experiment_name=self.experiment_name,
                            experiment_repetition=self.experiment_repetition,
                            level=self.level,
                            path_type=self.path_type,
                            dim=self.dimensions[0],
        )
        
        self.assertFalse(tensors_exists)
        self.assertFalse(os.path.exists(self.y_path))


    
    def test_tensor_data_exists_unsuccessful_Y_missing(self):
        FileSystem.tensor_data_export(experiment_repetition=self.experiment_repetition,
                            experiment_name=self.experiment_name,
                            X=self.tensor_X,
                            Y=self.tensor_Y,
                            level=self.level,
                            path_type=self.path_type,
                            dim=self.dimensions[0],
                            )
        
        os.remove(self.y_path)
        
        
        tensors_exists = FileSystem.tensor_data_exists(experiment_name=self.experiment_name,
                            experiment_repetition=self.experiment_repetition,
                            level=self.level,
                            path_type=self.path_type,
                            dim=self.dimensions[0],
        )
        
        self.assertFalse(tensors_exists)
        self.assertFalse(os.path.exists(self.x_path))
                
    def tearDown(self):
        FileSystem._delete_outdir()

class KeepNotValidationTensorFunctions(unittest.TestCase):

    def setUp(self):
        try:
            FileSystem.init_class(keep_data=False)
            self.dimensions = [100]
            self.experiment_repetition = "0"
            self.experiment_name = "test_experiment"
            self.dtype = "float32"
            self.path_type = "valid"
            self.level = 3
            self.tensor_X =  tf.constant([5, 5, 5, 5, 5],dtype=self.dtype)
            self.tensor_Y =  tf.constant([1, 1, 1, 1, 1],dtype=self.dtype)


            FileSystem.init_dimension_related_directories(self.dimensions)
            FileSystem.init_experiment_related_directories(dim = self.dimensions[0],
                                                           experiment_name=self.experiment_name,
                                                           experiment_repetition=self.experiment_repetition,
                                                           level=self.level)
            
            self.x_path,self.y_path = FileSystem._test_or_validation(self.path_type,self.dimensions[0],self.experiment_repetition,self.experiment_name,self.level)
    
        except Exception as e:
            self.skipTest("Setup failed! Subsequent tests would also fail." )  


    def test_tensor_data_exists_unsuccessful(self):

        FileSystem.tensor_data_export(experiment_repetition=self.experiment_repetition,
                            experiment_name=self.experiment_name,
                            X=self.tensor_X,
                            Y=self.tensor_Y,
                            level=self.level,
                            path_type=self.path_type,
                            dim=self.dimensions[0],
                            )
        
        tensors_exists = FileSystem.tensor_data_exists(experiment_name=self.experiment_name,
                            experiment_repetition=self.experiment_repetition,
                            level=self.level,
                            path_type=self.path_type,
                            dim=self.dimensions[0],
        )
        
        self.assertFalse(tensors_exists)
                
    def tearDown(self):
        FileSystem._delete_outdir()

class TestTensorFunctions(unittest.TestCase):

    def setUp(self):
        try:
            FileSystem.init_class(keep_data=False)
            self.dimensions = [100]
            self.experiment_repetition = "0"
            self.experiment_name = "test_experiment"
            self.dtype = "float32"
            self.path_type = "test_kpis"
            self.level = 3
            self.tensor_X =  tf.constant([5, 5, 5, 5, 5],dtype=self.dtype)
            self.tensor_Y =  tf.constant([1, 1, 1, 1, 1],dtype=self.dtype)


            FileSystem.init_dimension_related_directories(self.dimensions)
            FileSystem.init_experiment_related_directories(dim = self.dimensions[0],
                                                           experiment_name=self.experiment_name,
                                                           experiment_repetition=self.experiment_repetition,
                                                           level=self.level)
            
            self.x_path,self.y_path = FileSystem._test_or_validation(self.path_type,self.dimensions[0],self.experiment_repetition,self.experiment_name,self.level)
    
        except Exception as e:
            self.skipTest("Setup failed! Subsequent tests would also fail." )  
        
    def test_tensor_export(self):
        FileSystem.tensor_data_export(experiment_repetition=self.experiment_repetition,
                                    experiment_name=self.experiment_name,
                                    X=self.tensor_X,
                                    Y=self.tensor_Y,
                                    level=self.level,
                                    path_type=self.path_type,
                                    dim=self.dimensions[0]
                                    )

        x_exists = os.path.exists(self.x_path)
        y_exists = os.path.exists(self.y_path)
        
        self.assertTrue(x_exists)
        self.assertTrue(y_exists)

        os.remove(self.x_path)
        os.remove(self.y_path)

    def test_tensor_load(self):
        FileSystem.tensor_data_export(experiment_repetition=self.experiment_repetition,
                                    experiment_name=self.experiment_name,
                                    X=self.tensor_X,
                                    Y=self.tensor_Y,
                                    level=self.level,
                                    path_type=self.path_type,
                                    dim=self.dimensions[0],
                                    )
        
        X,Y = FileSystem.tensor_data_load(experiment_repetition=self.experiment_repetition,
                                    experiment_name=self.experiment_name,
                                    level=self.level,
                                    path_type=self.path_type,
                                    dim=self.dimensions[0],
                                    dtype=self.dtype
        )
        x_tensor_equal = tf.math.reduce_all(tf.equal(X,self.tensor_X))
        y_tensor_equal = tf.math.reduce_all(tf.equal(Y,self.tensor_Y))
        
        self.assertTrue(x_tensor_equal)
        self.assertTrue(y_tensor_equal)

        os.remove(self.x_path)
        os.remove(self.y_path)


    def test_tensor_data_exists_successful(self):

        FileSystem.tensor_data_export(experiment_repetition=self.experiment_repetition,
                            experiment_name=self.experiment_name,
                            X=self.tensor_X,
                            Y=self.tensor_Y,
                            level=self.level,
                            path_type=self.path_type,
                            dim=self.dimensions[0],
                            )
        
        tensors_exists = FileSystem.tensor_data_exists(experiment_name=self.experiment_name,
                            experiment_repetition=self.experiment_repetition,
                            level=self.level,
                            path_type=self.path_type,
                            dim=self.dimensions[0],
        )
        
        self.assertTrue(tensors_exists)

        os.remove(self.x_path)
        os.remove(self.y_path)

    def test_tensor_data_exists_unsuccessful_X_Y_missing(self):
        FileSystem.tensor_data_export(experiment_repetition=self.experiment_repetition,
                            experiment_name=self.experiment_name,
                            X=self.tensor_X,
                            Y=self.tensor_Y,
                            level=self.level,
                            path_type=self.path_type,
                            dim=self.dimensions[0],
                            )
        
        os.remove(self.x_path)
        os.remove(self.y_path)
        
        tensors_exists = FileSystem.tensor_data_exists(experiment_name=self.experiment_name,
                            experiment_repetition=self.experiment_repetition,
                            level=self.level,
                            path_type=self.path_type,
                            dim=self.dimensions[0],
        )
        
        self.assertFalse(tensors_exists)


class ModelFunctions(unittest.TestCase):

    def setUp(self):
        try:
            FileSystem.init_class()
            self.dimensions = [10,50,100]
            self.dim = self.dimensions[0]
            self.experiment_repetition = "1"
            self.experiment_name = "ML_Dynamic_Training"
            self.level = 3
            FileSystem.init_dimension_related_directories(self.dimensions)
            FileSystem.init_experiment_related_directories(dim = self.dimensions[0],
                                                           experiment_name=self.experiment_name,
                                                           experiment_repetition=self.experiment_repetition,
                                                           level=self.level)

        except Exception as e:
            self.skipTest("Setup failed! Subsequent tests would also fail." )
        
        self.models = []
        for level in range(self.level+1):
            path = os.path.dirname(os.path.abspath(__file__))+ "/models"+FileSystem._nomenclature_model.format(self.experiment_name,level)

            self.models.append(
                tf.keras.models.load_model(path,compile = False)
            )

    def test_model_export(self):

        for level,model in enumerate(self.models):
            FileSystem.model_export(model=model,
                                    experiment_name=self.experiment_name,
                                    dim = self.dim,
                                    level=level)
            path = FileSystem._model_path + FileSystem._nomenclature_d3_model_runner.format(self.dim)+ FileSystem._nomenclature_model.format(self.experiment_name,level)
            model_exists = os.path.exists(path)
            self.assertTrue(model_exists)
            os.remove(path)

    def test_model_load(self):
        
        for level,model in enumerate(self.models):
            FileSystem.model_export(model=model,
                                    experiment_name=self.experiment_name,
                                    dim = self.dim,
                                    level=level)
            
        for level in range(self.level+1):
            FileSystem.model_load(self.experiment_name,
                                  level=level,
                                  dim=self.dim)
            path = FileSystem._model_path + FileSystem._nomenclature_d3_model_runner.format(self.dim)+ FileSystem._nomenclature_model.format(self.experiment_name,level)
            os.remove(path)

    def test_model_exists_successful(self):
        for level,model in enumerate(self.models):
            FileSystem.model_export(model=model,
                                    experiment_name=self.experiment_name,
                                    dim = self.dim,
                                    level=level)
            
        for level in range(self.level+1):
            model_exists = FileSystem.model_exists(self.experiment_name,
                                  level=level,
                                  dim=self.dim)
            self.assertTrue(model_exists)

            path = FileSystem._model_path + FileSystem._nomenclature_d3_model_runner.format(self.dim)+ FileSystem._nomenclature_model.format(self.experiment_name,level)
            os.remove(path)


    def test_model_exists_unsuccessful(self):
         for level in range(self.level+1):
            model_exists = FileSystem.model_exists(self.experiment_name,
                                  level=level,
                                  dim=self.dim)
            self.assertFalse(model_exists)

    def tearDown(self):
        FileSystem._delete_outdir()