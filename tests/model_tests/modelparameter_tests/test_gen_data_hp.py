import unittest

import tensorflow as tf

from src.model.modelparameter.constants_hp import Constants

from src.model.modelparameter.gen_data_hp import gen_data_tf_const
from src.model.modelparameter.gen_data_hp import gen_data_python_const
from src.model.modelparameter.gen_data_hp import gen_data_opt_time,gen_data_opt_space
from src.model.modelparameter.gen_data_hp import gen_data_milstein_default_l0,gen_data_milstein_default_ln
from src.model.modelparameter.gen_data_hp import gen_data_milstein_control_variate
from src.model.modelparameter.gen_data_hp import gen_data_euler_default_l0,gen_data_euler_default_ln


class TestCaseGenDataConf4(unittest.TestCase):
    
    def setUp(self):
        self.constants = Constants(dim=100,asset_correlation_conf=0)

    def test_level_0_shape_x(self):
        level = 0
        num_timestep = self.constants.N ** level
        mc_samples = 1 
        n_points = 20

        X,y = gen_data_milstein_default_l0(
            const = self.constants,
            n_points = n_points,
            mc_samples = mc_samples,
            num_timestep = num_timestep
        )
        self.assertEqual(X.shape[0],n_points)
        self.assertEqual(X.shape[1],self.constants.dim)

    def test_level_0_shape_y(self):
        level = 0
        num_timestep = self.constants.N ** level 
        mc_samples = 1 
        n_points = 20
        
        X,y = gen_data_milstein_default_l0(
            const = self.constants,
            n_points = n_points,
            mc_samples = mc_samples,
            num_timestep = num_timestep
        )
        self.assertEqual(y.shape[0],n_points)
        self.assertEqual(y.shape[1],1)

    def test_level_0_value_y(self):
        level = 0
        n_points = 100
        mc_samples = 100
        num_timestep = self.constants.N ** level
        X,y = gen_data_milstein_default_l0(
            const = self.constants,
            n_points = n_points,
            mc_samples = mc_samples,
            num_timestep = num_timestep
        )

        self.assertLessEqual(6,tf.reduce_mean(y).numpy())
        self.assertLessEqual(tf.reduce_mean(y).numpy(),10)

    def test_level_1_shape_x(self):
        level = 1
        n_points = 20
        mc_samples = 2
        num_timestep = self.constants.N ** level
        X,y = gen_data_milstein_default_ln(
            const = self.constants,
            n_points = n_points,
            mc_samples = mc_samples,
            num_timestep = num_timestep
        )
        self.assertEqual(X.shape[0],n_points)
        self.assertEqual(X.shape[1],self.constants.dim)

    def test_level_1_shape_x(self):
        level = 1
        n_points = 20
        mc_samples = 1
        num_timestep = self.constants.N ** level 
        X,y = gen_data_milstein_default_ln(
            const = self.constants,
            n_points = n_points,
            mc_samples = mc_samples,
            num_timestep = num_timestep
        )
        self.assertEqual(y.shape[0],n_points)
        self.assertEqual(y.shape[1],1)

    def test_level_1_value_y(self):
        """Function 'gen_data_l0' test value of 'y'"""
        level = 1
        n_points = 100
        mc_samples = 100
        num_timestep = self.constants.N ** level
        X,y = gen_data_milstein_default_ln(
            const = self.constants,
            n_points = n_points,
            mc_samples = mc_samples,
            num_timestep = num_timestep
        )

        self.assertGreaterEqual(tf.reduce_mean(y).numpy(),0)
        self.assertLessEqual(tf.reduce_mean(y).numpy(),1)    

    
    def tearDown(self):
        self.constants = None

class TestCaseGenDataConf5(unittest.TestCase):
    def setUp(self):
        self.constants = Constants(dim=100,asset_correlation_conf=0)
        self.n_points = 20
        self.mc_samples = 1 

    def test_level_0_shape_x(self):
        level = 0
        num_timestep = self.constants.N ** level
        mc_samples = 1 
        n_points = 20

        X,y = gen_data_milstein_control_variate(
            const = self.constants,
            n_points = n_points,
            mc_samples = mc_samples,
            num_timestep = num_timestep
        )
        self.assertEqual(X.shape[0],n_points)
        self.assertEqual(X.shape[1],self.constants.dim)

    def test_level_0_shape_y(self):
        level = 0
        num_timestep = self.constants.N ** level 
        mc_samples = 1 
        n_points = 20
        
        X,y = gen_data_milstein_control_variate(
            const = self.constants,
            n_points = n_points,
            mc_samples = mc_samples,
            num_timestep = num_timestep
        )
        self.assertEqual(y.shape[0],n_points)
        self.assertEqual(y.shape[1],1)

    def test_level_0_value_y(self):
        level = 0
        n_points = 100
        mc_samples = 100
        num_timestep = self.constants.N ** level
        X,y = gen_data_milstein_control_variate(
            const = self.constants,
            n_points = n_points,
            mc_samples = mc_samples,
            num_timestep = num_timestep
        )

        self.assertLessEqual(6,tf.reduce_mean(y).numpy())
        self.assertLessEqual(tf.reduce_mean(y).numpy(),10)

    def tearDown(self):
        self.constants = None
    



class TestCaseGenDataConf6(unittest.TestCase):

    def setUp(self):
        self.constants = Constants(dim=100,asset_correlation_conf=0)
        self.n_points = 20
        self.mc_samples = 1 

    def test_level_0_shape_x(self):
        level = 0
        num_timestep = self.constants.N ** level
        mc_samples = 1 
        n_points = 20

        X,y = gen_data_euler_default_l0(
            const = self.constants,
            n_points = n_points,
            mc_samples = mc_samples,
            num_timestep = num_timestep
        )
        self.assertEqual(X.shape[0],n_points)
        self.assertEqual(X.shape[1],self.constants.dim)

    def test_level_0_shape_y(self):
        level = 0
        num_timestep = self.constants.N ** level 
        mc_samples = 1 
        n_points = 20
        
        X,y = gen_data_euler_default_l0(
            const = self.constants,
            n_points = n_points,
            mc_samples = mc_samples,
            num_timestep = num_timestep
        )
        self.assertEqual(y.shape[0],n_points)
        self.assertEqual(y.shape[1],1)

    def test_level_0_value_y(self):
        level = 0
        n_points = 100
        mc_samples = 100
        num_timestep = self.constants.N ** level
        X,y = gen_data_euler_default_l0(
            const = self.constants,
            n_points = n_points,
            mc_samples = mc_samples,
            num_timestep = num_timestep
        )

        self.assertLessEqual(6,tf.reduce_mean(y).numpy())
        self.assertLessEqual(tf.reduce_mean(y).numpy(),10)

    def test_level_1_shape_x(self):
        level = 1
        n_points = 20
        mc_samples = 2
        num_timestep = self.constants.N ** level
        X,y = gen_data_euler_default_ln(
            const = self.constants,
            n_points = n_points,
            mc_samples = mc_samples,
            num_timestep = num_timestep
        )
        self.assertEqual(X.shape[0],n_points)
        self.assertEqual(X.shape[1],self.constants.dim)

    def test_level_1_shape_x(self):
        level = 1
        n_points = 20
        mc_samples = 1
        num_timestep = self.constants.N ** level 
        X,y = gen_data_euler_default_ln(
            const = self.constants,
            n_points = n_points,
            mc_samples = mc_samples,
            num_timestep = num_timestep
        )
        self.assertEqual(y.shape[0],n_points)
        self.assertEqual(y.shape[1],1)

    def test_level_1_value_y(self):
        """Function 'gen_data_l0' test value of 'y'"""
        level = 1
        n_points = 100
        mc_samples = 100
        num_timestep = self.constants.N ** level
        X,y = gen_data_euler_default_ln(
            const = self.constants,
            n_points = n_points,
            mc_samples = mc_samples,
            num_timestep = num_timestep
        )

        self.assertGreaterEqual(tf.reduce_mean(y).numpy(),0)
        self.assertLessEqual(tf.reduce_mean(y).numpy(),1)   

    def tearDown(self):
        self.constants = None
