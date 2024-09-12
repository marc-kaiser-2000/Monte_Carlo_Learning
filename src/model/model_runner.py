"""Cotains ModelRunner-Class: Sequentially Executes 
"""
import tensorflow as tf

from src.model.modelcomponents.root_model import RootModel
from src.model.modelparameter.hyperparameter_manager import HyperparameterManager
from src.utils.filesystem import FileSystem

class ModelRunner:
    """Sequentially executes specified experiments
    """

    def __init__(self,tf_eager_mode = False, debug = False) -> None:

        if tf_eager_mode:
            tf.config.run_functions_eagerly(True)
        self.debug = debug


    def run(self):
        """Starts exeuction of specified Monte Carlo learning experiments.
        """

        # === Possibility to include multiple models ===
        dimensions = [10,100,50]

        FileSystem.init_dimension_related_directories(dimensions)

        for dim in dimensions:
            if self.debug:

                hyperparameter_manager = HyperparameterManager(0,"debug",dim)
                aggregate_model = RootModel(hyperparameter_manager = hyperparameter_manager,
                                                level=hyperparameter_manager.max_level)
                aggregate_model.train()
                aggregate_model.evaluate()

                """
                hyperparameter_manager = HyperparameterManager(1,"debug",dim)
                aggregate_model = RootModel(hyperparameter_manager = hyperparameter_manager,
                                                level=hyperparameter_manager.max_level)
                aggregate_model.train()
                aggregate_model.evaluate()

                hyperparameter_manager = HyperparameterManager(0,"debug2",dim)
                aggregate_model = RootModel(hyperparameter_manager = hyperparameter_manager,
                                                level=hyperparameter_manager.max_level)
                aggregate_model.train()
                aggregate_model.evaluate()
                """
            else:
                
                # Thesis chapter 6.2 Singlelevel Benchmark L3
                hyperparameter_manager = HyperparameterManager(0,"SL_Benchmark",dim)
                aggregate_model = RootModel(hyperparameter_manager = hyperparameter_manager,
                                                level=hyperparameter_manager.max_level)
                aggregate_model.train()
                aggregate_model.evaluate()

                # Thesis chapter 6.2 Singlelevel Benchmark L0
                hyperparameter_manager = HyperparameterManager(0,"SL_Benchmark_L0",dim)
                aggregate_model = RootModel(hyperparameter_manager = hyperparameter_manager,
                                                level=hyperparameter_manager.max_level)
                aggregate_model.train()
                aggregate_model.evaluate()
                
                # Thesis chapter 6.2 Singlelevel Benchmark L5
                hyperparameter_manager = HyperparameterManager(0,"SL_Benchmark_L5",dim)
                aggregate_model = RootModel(hyperparameter_manager = hyperparameter_manager,
                                                level=hyperparameter_manager.max_level)
                aggregate_model.train()
                aggregate_model.evaluate()

                # Thesis chapter 6.3 Multilevel Benchmark
                hyperparameter_manager = HyperparameterManager(0,"ML_Benchmark",dim)
                aggregate_model = RootModel(hyperparameter_manager = hyperparameter_manager,
                                                level=hyperparameter_manager.max_level)
                aggregate_model.train()
                aggregate_model.evaluate()

                # Thesis chapter 6.4 Multilevel Apriori Training
                hyperparameter_manager = HyperparameterManager(0,"ML_Semi_Apriori_Training",dim)
                aggregate_model = RootModel(hyperparameter_manager = hyperparameter_manager,
                                                level=hyperparameter_manager.max_level)
                aggregate_model.train()
                aggregate_model.evaluate()

                # Thesis chapter 6.5 Multilevel Dynamic Training
                hyperparameter_manager = HyperparameterManager(0,"ML_Dynamic_Training_1",dim)
                aggregate_model = RootModel(hyperparameter_manager = hyperparameter_manager,
                                                level=hyperparameter_manager.max_level)
                aggregate_model.train()
                aggregate_model.evaluate()

                # Thesis chapter 6.6 Multilevel Dynamic Training Control
                hyperparameter_manager = HyperparameterManager(0,"ML_Dynamic_Training_2",dim)
                aggregate_model = RootModel(hyperparameter_manager = hyperparameter_manager,
                                                level=hyperparameter_manager.max_level)
                aggregate_model.train()
                aggregate_model.evaluate()

                # Thesis chapter 6.7 Multilevel Transfer Learning
                hyperparameter_manager = HyperparameterManager(1,"ML_Dynamic_Training_2",dim)
                aggregate_model = RootModel(hyperparameter_manager = hyperparameter_manager,
                                                level=hyperparameter_manager.max_level)
                aggregate_model.train()
                aggregate_model.evaluate()

                # Thesis chapter 6.7 Multilevel Transfer Learning (Control)
                hyperparameter_manager = HyperparameterManager(0,"ML_Transfer_Control",dim)
                aggregate_model = RootModel(hyperparameter_manager = hyperparameter_manager,
                                                level=hyperparameter_manager.max_level)
                aggregate_model.train()
                aggregate_model.evaluate()

                # Thesis chapter 6.8 Multilevel Variance Reduction
                hyperparameter_manager = HyperparameterManager(0,"ML_Variance_Reduction_1",dim)
                aggregate_model = RootModel(hyperparameter_manager = hyperparameter_manager,
                                                level=hyperparameter_manager.max_level)
                aggregate_model.train()
                aggregate_model.evaluate()

                """
                # Experiments which were not added to the thesis

                hyperparameter_manager = HyperparameterManager(0,"ML_Apriori_Training",dim)
                aggregate_model = RootModel(hyperparameter_manager = hyperparameter_manager,
                                                level=hyperparameter_manager.max_level)
                aggregate_model.train()
                aggregate_model.evaluate()
      
                hyperparameter_manager = HyperparameterManager(0,"ML_Inverted_Training",dim)
                aggregate_model = RootModel(hyperparameter_manager = hyperparameter_manager,
                                                level=hyperparameter_manager.max_level)
                aggregate_model.train()
                aggregate_model.evaluate()

                hyperparameter_manager = HyperparameterManager(0,"ML_Euler_Scheme",dim)
                aggregate_model = RootModel(hyperparameter_manager = hyperparameter_manager,
                                                level=hyperparameter_manager.max_level)
                aggregate_model.train()
                aggregate_model.evaluate()

                hyperparameter_manager = HyperparameterManager(0,"ML_Variance_Reduction_2",dim)
                aggregate_model = RootModel(hyperparameter_manager = hyperparameter_manager,
                                                level=hyperparameter_manager.max_level)
                aggregate_model.train()
                aggregate_model.evaluate()

                hyperparameter_manager = HyperparameterManager(1,"ML_Variance_Reduction_1",dim)
                aggregate_model = RootModel(hyperparameter_manager = hyperparameter_manager,
                                                level=hyperparameter_manager.max_level)
                aggregate_model.train()
                aggregate_model.evaluate()
                """