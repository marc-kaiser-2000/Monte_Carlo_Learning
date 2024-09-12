from datetime import datetime

import tensorflow as tf
import numpy as np

from src.model.modelcomponents.abstract_model import AbstractModel
from src.model.modelcomponents.leaf_model import LeafModel

from src.utils.log_manager import LogManager
from src.utils.filesystem import FileSystem
from src.utils.plot import Plot


class RootModel(AbstractModel):
    """Root due to graph like interpretation.
    Similar to composite pattern without recursive composition (Composite).
    Trains one or multiple neural networks by creating 'LeafModel' objects.
    """

    def __init__(self,hyperparameter_manager,level) -> None:

        # Set Experiment Key and Name
        self.experiment_repetition = hyperparameter_manager.experiment_repetition
        self.experiment_name = hyperparameter_manager.experiment_name

        # Load Buffered Constants
        self.constants = hyperparameter_manager.buffer_constants

        # Initialize Buffer Variables
        hyperparameter_manager.init_buffer_root()

        # Load Buffered Variables
        self.mc_step = hyperparameter_manager.buffer_mc_step_fnc
        self.sample_size_test = hyperparameter_manager.buffer_sample_size_test
        self.batch_size_test = hyperparameter_manager.buffer_batch_size_test

        # Create Experiment related Directories
        FileSystem.init_experiment_related_directories(
            dim = self.constants.dim,
            experiment_repetition= self.experiment_repetition,
            experiment_name=self.experiment_name,
            level=level
        )

        # Initialize Log Manager
        self.logger = LogManager.initialize_experiment_logger(
            dim = self.constants.dim,
            experiment_repetition=self.experiment_repetition,
            experiment_name=self.experiment_name,
            level=level
        )

        # Leaf Components
        self.components = None
        self.init_components(hyperparameter_manager,level)

        super().__init__(hyperparameter_manager,level,self.constants.N**level)

    def init_components(self,hyperparameter_manager,level):
        """Intializes related 'LeafModel' objects depending on 'model_type'

        Args:
            hyperparameter_manager (HyperparameterManager): Instance
            level (int): Level of respective experiment.

        Raises:
            KeyError: Specified 'model_type' is not existing.
        """
        self.components = []
        if hyperparameter_manager.model_type == "multi_level":
            for sub_level in range(level+1):
                self.components.append(LeafModel(hyperparameter_manager=hyperparameter_manager,
                                level = sub_level,
                                num_timestep=self.constants.N**sub_level
                                )
                        )
        elif hyperparameter_manager.model_type == "single_level":
            self.components.append(LeafModel(hyperparameter_manager=hyperparameter_manager,
                               level = 0,
                               num_timestep=self.constants.N**level
                                )
                        )
        else:
            raise KeyError("Key not implemented")




    def train(self):
        """Trains 'RootModel' by training all related 'LeafModels'
        Similar approach as in composite pattern.
        """
        ### Logging Functions: Total ###
        t_start_total = datetime.now()
        self.logger.info(f"Root Model Level {self.level} : Run Start")

        ### Model Train ###
        for c in self.components:
            c.train()

        ### Logging Functions: Total ###
        t_end_total = datetime.now()
        self.logger.info(f"Root Model Level {self.level} : Run End")
        duration_total = t_end_total - t_start_total
        self.logger.info(f"Run Duration : {duration_total}")

    def evaluate(self):
        """Evaluates 'RootModel' (composite of one or multiple 'LeafModel' instances) by
        by evaluating a slice of the solution graphically in 'evaluate_plot'.
        A KPI based test is executed in 'evaluate_kpis'.
        """

        self.evaluate_plot()

        self.evaluate_kpis()


    def init_xgrid(self,idx0,idx1,ngrid = 100):
        """Initializes the x and y axis (plane) for a three dimensional plot.

        Args:
            idx0 (int): First dimension of basket which is plotted.
            idx1 (int): Second dimension of basket which is plotted.
            ngrid (int, optional): Discretization of x- and y-axis respectively. Defaults to 100.

        Returns:
            _type_: _description_
        """
        # Determine remaining indices
        idx = np.arange(self.constants.dim)
        idx_remaining = np.setdiff1d(idx,[idx0,idx1])
        # Set remaining values to midpoints
        val_remaining = np.array((self.constants.a+self.constants.b)/2)

        # Meshgrid for plots
        xspace = np.linspace(self.constants.a[idx0], self.constants.b[idx0], ngrid + 1, dtype=self.constants.DTYPE)
        yspace = np.linspace(self.constants.a[idx1], self.constants.b[idx1], ngrid + 1, dtype=self.constants.DTYPE)
        X,Y = np.meshgrid(xspace, yspace)

        # Append remaining values
        Z = np.repeat(val_remaining[idx_remaining].reshape(1,-1), (ngrid+1)**2,axis=0)
        Xgrid = np.vstack([X.flatten(),Y.flatten()]).T
        Xgrid = np.hstack([Xgrid,Z])

        Xgrid = tf.convert_to_tensor(Xgrid,dtype=self.constants.DTYPE)
        return X,Y,Xgrid


    def evaluate_plot(self):
        """The one or multiple related 'LeafModels' are evaluated.
        Function plots a two-dimansional slice solution provided by trained models.
        Additionally the relative error of this slice is also plotted. 
        """
        self.logger.info(f"Root Model Level {self.level} : 'evaluate_plot' Start")

        #Note: It is okay to initialize xgrid here, since it is deterministic.
        ngrid = 100
        X_surface,Y_surface,X_test = self.init_xgrid(
            idx0 = 0,
            idx1 = 1,
            ngrid = ngrid)

        batch_size_test_calculated, dim = X_test.shape

        if FileSystem.tensor_data_exists(self.experiment_repetition,self.experiment_name,self.level,"test_plot",self.constants.dim):
            self.logger.info(f"Root Model Level {self.level}: Load Test Plot Data")
            X_test, y_test = FileSystem.tensor_data_load(
                experiment_repetition= self.experiment_repetition,
                experiment_name=self.experiment_name,
                level = self.level,
                dtype = self.constants.DTYPE,
                path_type = "test_plot",
                dim = self.constants.dim
            )
        else:
            self.logger.info(f"Root Model Level {self.level}: Generate Test Plot Data")
            y_test = tf.zeros(
                shape=(batch_size_test_calculated,1),
                dtype=self.constants.DTYPE
            )

            for i in range(self.sample_size_test):
                if (i+1) % 1000 == 0:
                    print(f"MC Samples: {i+1} / {self.sample_size_test}")

                y_test = self.mc_step(
                    init_S0 = X_test,
                    const = self.constants,
                    n_points = batch_size_test_calculated,
                    mc_samples = self.sample_size_test,
                    num_timestep = self.num_timestep,
                    y = y_test
                )

            FileSystem.tensor_data_export(
                experiment_repetition= self.experiment_repetition,
                experiment_name=self.experiment_name,
                X = X_test,
                Y = y_test,
                level = self.level,
                dim = self.constants.dim,
                path_type = "test_plot")

        Y_pred = tf.zeros(
                shape=(batch_size_test_calculated,1),
                #shape = ()
                dtype=self.constants.DTYPE
                )

        for component in self.components:
            Y_pred  += component.evaluate(X_test)


        #Calculate Metrics
        uabs_error =tf.abs(Y_pred -  y_test)
        urel_error =uabs_error/y_test
        Urel = urel_error.numpy().reshape(ngrid+1,ngrid+1)

        # Slice of solution
        U = Y_pred.numpy().reshape(ngrid+1,ngrid+1)

        out_path = FileSystem.get_model_runner_experiment_root_path(
            dim = self.constants.dim,
            experiment_repetition=str(self.experiment_repetition),
            experiment_name=self.experiment_name,
            level=self.hyperparameter_manager.max_level
        )

        Plot.plot_model_result_surface(
            X = X_surface,
            Y = Y_surface,
            Z = U,
            path = out_path + "/Sol_OptionPricing.pdf"
        )

        Plot.plot_model_result_surface(
            X = X_surface,
            Y = Y_surface,
            Z = Urel,
            path = out_path + "/RelError_OptionPricing.pdf"
        )


    def evaluate_kpis(self):
        """The one or multiple related 'LeafModels' are evaluated.
        Function compares test data with solution given by trained models and calculates error.
        """
        ### Logging Functions: Build ###
        self.logger.info(f"Root Model Level {self.level} : 'evaluate_kpis' Start")

        if FileSystem.tensor_data_exists(self.experiment_repetition,self.experiment_name,self.level,"test_kpis",self.constants.dim):
            self.logger.info(f"Root Model Level {self.level}: Load Test KPIs Data")
            X_test, y_test = FileSystem.tensor_data_load(
                experiment_repetition= self.experiment_repetition,
                experiment_name=self.experiment_name,
                level = self.level,
                dtype = self.constants.DTYPE,
                path_type = "test_kpis",
                dim = self.constants.dim
            )
        else:
            self.logger.info(f"Root Model Level {self.level}: Generate Test KPIs Data")
            X_test = self.constants.a + tf.random.uniform((self.batch_size_test,self.constants.dim), dtype=self.constants.DTYPE) * (self.constants.b-self.constants.a)
            y_test = tf.zeros(
                shape=(self.batch_size_test,1),
                dtype=self.constants.DTYPE
            )

            for i in range(self.sample_size_test):
                if (i+1) % 1000 == 0:
                    print(f"MC Samples: {i+1} / {self.sample_size_test}")
                y_test = self.mc_step(
                    init_S0 = X_test,
                    const = self.constants,
                    n_points = self.batch_size_test,
                    mc_samples = self.sample_size_test,
                    num_timestep = self.num_timestep,
                    y = y_test
                )


            FileSystem.tensor_data_export(
                experiment_repetition= self.experiment_repetition,
                experiment_name=self.experiment_name,
                X = X_test,
                Y = y_test,
                level = self.level,
                dim = self.constants.dim,
                path_type = "test_kpis")

        Y_pred = tf.zeros(
                shape=(self.batch_size_test,1),
                dtype=self.constants.DTYPE
                )

        for component in self.components:
            Y_pred  += component.evaluate(X_test)

        aggregated_absolute_error = self.get_aggregated_absolute_error()

        self.logger.info('Aggregated Error:')
        self.logger.info('|  L1_rel   L2_rel   Linf_rel |   L1_abs   L2_abs   Linf_abs|')
        self.logger.info('|{:8.4f} {:8.4f}   {:8.4f} | {:8.4f} {:8.4f}   {:8.4f}  |'.format(*aggregated_absolute_error))

        #Calculate Metrics
        abs_error =tf.abs(Y_pred -  y_test)
        rel_error =abs_error/y_test

        L2_rel = tf.sqrt(tf.reduce_mean(tf.pow(rel_error, 2))).numpy()
        L1_rel = tf.reduce_mean(tf.abs(rel_error)).numpy()
        Linf_rel = tf.reduce_max(tf.abs(rel_error)).numpy()

        L2_abs = tf.sqrt(tf.reduce_mean(tf.pow(abs_error, 2))).numpy()
        L1_abs = tf.reduce_mean(tf.abs(abs_error)).numpy()
        Linf_abs = tf.reduce_max(tf.abs(abs_error)).numpy()

        err = (L1_rel, L2_rel, Linf_rel, L1_abs, L2_abs, Linf_abs)
        self.logger.info('Test Error:')
        self.logger.info('|  L1_rel   L2_rel   Linf_rel |   L1_abs   L2_abs   Linf_abs|')
        self.logger.info('|{:8.4f} {:8.4f}   {:8.4f} | {:8.4f} {:8.4f}   {:8.4f}  |'.format(*err))

    def get_aggregated_absolute_error(self):
        """Gets the aggregated error of all related leaf models

        Returns:
            list[float]: aggregated errpr
        """

        #Rel L1 | Rel L2 | Rel LInf| Abs L1 | Abs L2 | Abs LInf
        aggregated_error = [0,0,0,0,0,0]

        for c in self.components:
            last_validation = c.error_hist[-1]
            aggregated_error[0] += last_validation[2]
            aggregated_error[1] += last_validation[3]
            aggregated_error[2] += last_validation[4]
            aggregated_error[3] += last_validation[5]
            aggregated_error[4] += last_validation[6]
            aggregated_error[5] += last_validation[7]
        return tuple(aggregated_error)
