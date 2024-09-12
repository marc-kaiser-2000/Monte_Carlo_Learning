from datetime import datetime

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.model.modelcomponents.abstract_model import AbstractModel
from src.utils.filesystem import FileSystem
from src.utils.log_manager import LogManager
from src.utils.plot import Plot

class LeafModel(AbstractModel):
    """Leaf due to graph like interpretation.
    Similar to composite pattern without recursive composition (Leaf).
    Trains a neural network.
    """

    def __init__(self, hyperparameter_manager,level,num_timestep) -> None:

        # Validation Data
        self.Xvalid = None
        self.Yvalid = None

        # Hyperparemeters Train
        self.batch_size_train = None
        self.sample_size_train = None
        self.increase_level = None

        # Error and Loss History
        self.error_hist = []
        self.loss_hist = []

        # Initialize Buffer Variables
        hyperparameter_manager.init_buffer_leaf(level)

        # Load Buffered Variables 
        self.gen_data = hyperparameter_manager.buffer_gen_data_fnc
        self.mc_step = hyperparameter_manager.buffer_mc_step_fnc
        self.optimizer = hyperparameter_manager.buffer_optimizer
        self.compute_grad = hyperparameter_manager.buffer_grad_fnc
        self.model = hyperparameter_manager.buffer_model
        self.batch_size_valid = hyperparameter_manager.buffer_batch_size_valid
        self.sample_size_valid = hyperparameter_manager.buffer_sample_size_valid
        self.experiment_repetition = hyperparameter_manager.experiment_repetition
        self.experiment_name = hyperparameter_manager.experiment_name 
        self.constants = hyperparameter_manager.buffer_constants

        #Logger 
        self.logger = LogManager.get_experiment_logger()

        super().__init__(hyperparameter_manager=hyperparameter_manager,
            level = level,
            num_timestep = num_timestep)


    def _initalize_train_hyperparameter(self,level):
        """Initializes the required hyperparameter for training a neural network.

        Args:
            level (int): Sublevel given by related 'RootModel'
        """
        self.hyperparameter_manager.init_buffer_train(level)
        self.batch_size_train = self.hyperparameter_manager.buffer_batch_size_train
        self.sample_size_train = self.hyperparameter_manager.buffer_sample_size_train
        self.increase_level = self.hyperparameter_manager.buffer_increase_level


    def _update_train_hyperparameter(self,error_hist,level):
        """Updates the required hyperparameter for training a neural network.

        Args:
            error_hist (List[tupel]): Absolute and relative history of validation training error.
            level (int): Sublevel given by related 'RootModel'
        """
        self.hyperparameter_manager.update_buffer_train(error_hist,level)
        self.batch_size_train = self.hyperparameter_manager.buffer_batch_size_train
        self.sample_size_train = self.hyperparameter_manager.buffer_sample_size_train
        self.increase_level = self.hyperparameter_manager.buffer_increase_level
        self.optimizer.learning_rate = self.hyperparameter_manager.buffer_learning_rate

    def _get_increase_level(self):
        return self.increase_level

    def train(self):
        """Trains a neural network.
        Due to adapted composite structure this function is called by 'RootModel' objects.
        Function may be separated into two phases.
        At first validation data is generated and hyperparameters are initialized.
        Hereafter the training loop is entred.
        After a fix number of epochs, the model is validate (early-stoppin like behavior).
        """
        ### Logger Functions: Train ###
        self.logger.info(f"Leaf Model Level {self.level}: 'train' Start")
        t_start_train = datetime.now()

        self.validation_data()

        ### Logger Functions: Loop ###
        self.logger.info(f"Leaf Model Level {self.level}: Loop Start")        
        t_start_loop = datetime.now()

        self._initalize_train_hyperparameter(self.level)

        # Initialize header of output
        self.logger.info('  Iter        Loss   L1_rel   L2_rel   Linf_rel |   L1_abs   L2_abs   Linf_abs  |    Time  Stepsize  MC Samples  Startpoints per Sample| Current  Peak ')

        i = 0
        log_interval = 1000
        #log_interval = 1
        while not self._get_increase_level():

            i += 1

            ### Perform training step ###
            loss = self.train_step(sample_size=self.sample_size_train,
                                   batch_size=self.batch_size_train)

            self.loss_hist.append(loss)

            if i % log_interval == 0:
                ### Compute Model Prediction ###
                Ypred = self.model(self.Xvalid, training=False)
                t_current_train = datetime.now()
                t_delta_train = t_current_train - t_start_train 


                ### Compute Errors ###
                abs_error = tf.abs(Ypred - self.Yvalid)
                rel_error = abs_error/self.Yvalid
                L2_rel = tf.sqrt(tf.reduce_mean(tf.pow(rel_error, 2))).numpy()
                L1_rel = tf.reduce_mean(tf.abs(rel_error)).numpy()
                Linf_rel = tf.reduce_max(tf.abs(rel_error)).numpy()

                L2_abs = tf.sqrt(tf.reduce_mean(tf.pow(abs_error, 2))).numpy()
                L1_abs = tf.reduce_mean(tf.abs(abs_error)).numpy()
                Linf_abs = tf.reduce_max(tf.abs(abs_error)).numpy()


                t_current_train = datetime.now()
                t_delta_train = t_current_train - t_start_train 


                ### Logging Functions: GPU Usage ### 
                if tf.config.experimental.list_physical_devices('GPU'):
                    gpu_info = tf.config.experimental.get_memory_info('GPU:0')
                    current = gpu_info['current']
                    peak = gpu_info['peak']
                else:
                    current = 0
                    peak = 0

                ### Logging Functions: Train ###                
                t_current_train = datetime.now()
                t_delta_train = t_current_train - t_start_train

                stepsize = self.optimizer.learning_rate.numpy()
                err = (i, loss.numpy(), L1_rel, L2_rel, Linf_rel, L1_abs, L2_abs, Linf_abs, t_delta_train.total_seconds(), stepsize,self.sample_size_train,self.batch_size_train,current,peak)
                self.error_hist.append(err)
                self.logger.info('{:5d} {:12.4f} {:8.4f} {:8.4f}   {:8.4f} | {:8.4f} {:8.4f}   {:8.4f}  |  {:6.1f}  {:6.2e}  {:8d}  {:8d}| {:11d}  {:11d}'.format(*err))


                ### Update Runparameter Values ###
                self._update_train_hyperparameter(self.error_hist,self.level)


                t_current_train = datetime.now()
                t_delta_train = t_current_train - t_start_train 

        FileSystem.model_export(self.model,self.experiment_name,self.level,self.constants.dim)

        self.Xvalid = None
        self.Yvalid = None
        ### Create Analytics ###
        self.create_analytics()

        ### Logger Functions: Loop ###
        self.logger.info(f"Leaf Model Level {self.level}: Loop End")
        t_end_loop= datetime.now()
        t_duration_loop = t_end_loop - t_start_loop
        self.logger.info(f"Duration: {t_duration_loop}")

        ### Logger Functions: Train ### 
        self.logger.info(f"Leaf Model Level {self.level}: 'train' End")
        t_end_train= datetime.now()
        t_duration_train = t_end_train - t_start_train
        self.logger.info(f"Duration: {t_duration_train}")


    def evaluate(self,input):
        """Evaluas trained neural network for a given input 
        and returns generated output.
        Due to adapted composite structure this function is called by 'RootModel'

        Args:
            input (tf.tensor): Input signal.

        Returns:
            tf.tensor: Estimated output generated by neural network.
        """
        ### Model Evaluate ###
        out = self.model(input,training=False)
        return  out

    def validation_data(self):
        """Generates validation data used during training.
        Exports generated data.
        When validation data already exists it is imoprted and generation is skipped.
        """
        self.logger.info(f"Leaf Model Level {self.level}: 'validation_data' Start")
        t_start_validation_data = datetime.now()
        ### Check if Validation Data exists ### 
        if FileSystem.tensor_data_exists(self.experiment_repetition,self.experiment_name,self.level,"valid",self.constants.dim):
            self.logger.info(f"Leaf Model Level {self.level}: Load Validation Data")

            X, y = FileSystem.tensor_data_load(
                experiment_repetition= self.experiment_repetition,
                experiment_name=self.experiment_name,
                level=self.level,
                dtype = self.constants.DTYPE,
                path_type = "valid",
                dim = self.constants.dim)
        else:
            self.logger.info(f"Leaf Model Level {self.level}: Generate Validation Data")

            X = self.constants.a + tf.random.uniform((self.batch_size_valid,self.constants.dim), dtype=self.constants.DTYPE) * (self.constants.b-self.constants.a)
            y = tf.zeros(
                shape=(self.batch_size_valid,1),
                dtype=self.constants.DTYPE
            )

            for i in range(self.sample_size_valid):
                if (i+1) % 1000 == 0:
                    print(f"MC Samples: {i+1} / {self.sample_size_valid}")

                y = self.mc_step(
                    init_S0 = X,
                    const = self.constants,
                    n_points = self.batch_size_valid,
                    mc_samples = self.sample_size_valid,
                    num_timestep = self.num_timestep,
                    y = y
                )

            FileSystem.tensor_data_export(
                experiment_repetition= self.experiment_repetition,
                experiment_name=self.experiment_name,
                X = X,
                Y = y,
                level = self.level,
                dim = self.constants.dim,
                path_type = "valid")

        self.Xvalid = X
        self.Yvalid = y

        self.logger.info(f"Leaf Model Level {self.level}: 'validation_data' End")
        t_end_validation_data = datetime.now()
        t_duration_validation_data = t_end_validation_data - t_start_validation_data
        self.logger.info(f"Duration: {t_duration_validation_data}")

    @tf.function
    def train_step(self,sample_size,batch_size):
        """Executes a single training epoch.
        Generates training data.
        Computes loss and gradients.
        Applies gradient.

        Args:
            sample_size (int): Monte Carlo sample size for 'gen_data'
            batch_size (int): Batch size for forward and backward pass

        Returns:
            tf.tensor : tensor of the calculated loss for this epoch. 
        """
        #print("Function train_step: tracing!")
        # Draw batch of random paths

        X,y = self.gen_data(
            const = self.constants,
            n_points = batch_size,
            mc_samples = sample_size,
            num_timestep = self.num_timestep
        )

        # Compute the loss and the gradient
        loss, grad = self.compute_grad(X, y, self.model, training=True)

        # Perform gradient step
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        return loss

    def create_analytics(self):
        """Function generates plots of absolute and relative error generated by 
        during validation.
        """

        out_path = FileSystem.get_model_runner_experiment_leaf_path(
            dim = self.constants.dim,
            experiment_repetition=self.experiment_repetition,
            experiment_name=self.experiment_name,
            level=self.level
        )

        xrange =  np.arange(len(self.error_hist))*self.error_hist[0][0]
        Plot.plot_model_train_line(
            x = xrange,
            y = [e[2:5] for e in self.error_hist],
            path = out_path+"/Errors_OptionPricing_Rel.pdf")

        Plot.plot_model_train_line(
            x = xrange,
            y = [e[5:8] for e in self.error_hist],
            path = out_path+"/Errors_OptionPricing_Abs.pdf"
            )
