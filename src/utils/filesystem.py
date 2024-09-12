"""Interface to filesystem
"""
import os
import shutil

import tensorflow as tf

class FileSystem:
    """Class provides and aggregates functions to use the file system.
    """

    # Nomenclature for file description

    _nomenclature_valid_data_X = "/Repetition_{}_Name_{}_X_Level_{}.dat"
    _nomenclature_valid_data_Y = "/Repetition_{}_Name_{}_Y_Level_{}.dat"
    _nomenclature_test_kpis_data_X = "/KPIs_Repetition_{}_Name_{}_X_Level_{}.dat"
    _nomenclature_test_kpis_data_Y = "/KPIs_Repetition_{}_Name_{}_Y_Level_{}.dat"
    _nomenclature_test_plot_data_X = "/Plot_Repetition_{}_Name_{}_X_Level_{}.dat"
    _nomenclature_test_plot_data_Y = "/Plot_Repetition_{}_Name_{}_Y_Level_{}.dat"
    _nomenclature_model = "/Experiment_{}_Model_Level_{}.keras"
    _nomenclature_analysis_plot = "/Analysis_{}_Gen_Data_Conf_{}.pdf"
    _nomenclature_d3_model_runner = "/dim_{}"
    _nomenclature_d4_model_runner_leaf = "/Leaf_Level_{}_Repetition_{}_Name_{}"
    _nomenclature_d4_model_runner_root = "/Root_Level_{}_Repetition_{}_Name_{}"
    _nomenclature_d2_analysis_runner = "/Level_{}"

    # Source File Paths
    _file_abs_path = None
    _cfg_abs_path = None
    _test_abs_path = None

    # Boolean Working Variables
    _outdir_initialized = None
    _use_exported_tensors = None

    # Root Directory
    _outdir = None

    # Depth 1 Directories
    _out_model_runner_path = None
    _out_analysis_runner_path = None
    _tensorboard_path = None

    # Depth 2 Directories 'out_model_runner'
    _analytics_path = None
    _validation_data_path = None
    _test_data_path = None
    _model_path = None





    def __init__(self) -> None:
        pass

    @classmethod
    def init_class(cls,keep_data = False):
        """Init function for class.

        Args:
            keep_data (bool, optional): Determines if validation data is reused if existing.
            Defaults to False.
        """

        cls._reset_variables()

        dirname=os.path.dirname
        cls._file_abs_path = dirname(dirname(os.path.abspath(__file__)))
        cls._cfg_abs_path = os.path.join(cls._file_abs_path , "configs","run_configs")
        cls._init_outdir("./outdir")
        cls._use_exported_tensors = keep_data


    @classmethod
    def _reset_variables(cls) -> None:
        # Source File Paths
        cls._file_abs_path = None
        cls._cfg_abs_path = None
        cls._test_abs_path = None

        # Boolean Working Variables
        cls._outdir_initialized = None
        cls._use_exported_tensors = None

        # Root Directory
        cls._outdir = None

        # Depth 1 Directories
        cls._out_model_runner_path = None
        cls._out_analysis_runner_path = None
        cls._tensorboard_path = None

        # Depth 2 Directories 'out_model_runner'
        cls._analytics_path = None
        cls._validation_data_path = None
        cls._test_data_path = None
        cls._model_path = None


    @classmethod
    def _init_outdir(cls,outdir) -> None:
        cls._outdir = os.path.abspath(outdir)

        #Create Roote
        os.makedirs(cls._outdir,exist_ok=True)

        cls._init_outdir_d1()
        cls._outdir_initialized = True

    @classmethod
    def _init_outdir_d1(cls) -> None:
        # Initialize Paths
        cls._out_model_runner_path = cls._outdir + "/out_model_runner"
        cls._out_analysis_runner_path = cls._outdir + "/out_analysis_runner"
        cls._tensorboard_path = cls._outdir +"/tensorboard"

        # Create Paths Depth 1
        os.makedirs(cls._out_model_runner_path,exist_ok=True)
        os.makedirs(cls._out_analysis_runner_path, exist_ok= True)
        os.makedirs(cls._tensorboard_path, exist_ok= True)

        # Clear Paths
        cls._clear(cls._tensorboard_path)
        cls._clear(cls._out_analysis_runner_path)

        cls._init_outdir_d2_model_runner()
        # d2 for analysis runner is called through client
        # d2 for tensorboard does not exist



    @classmethod
    def _init_outdir_d2_model_runner(cls) -> None:
        # Initialize Paths
        cls._analytics_path = cls._out_model_runner_path + "/analytics_path"
        cls._validation_data_path = cls._out_model_runner_path +"/validation_data_path"
        cls._test_data_path = cls._out_model_runner_path +"/test_data_path"
        cls._model_path = cls._out_model_runner_path + "/model"

        # Create Paths Depth 2
        os.makedirs(cls._analytics_path,exist_ok=True)
        os.makedirs(cls._validation_data_path,exist_ok=True)
        os.makedirs(cls._test_data_path,exist_ok=True)
        os.makedirs(cls._model_path,exist_ok=True)

        # Clear Paths
        cls._clear(cls._analytics_path)
        cls._clear(cls._model_path)


        # d3 for all paths is created through the client

    @classmethod
    def _init_outdir_d2_analysis_runner(cls,level) -> None:
        #Initialize Paths Level 2 in "/out_model_runner"
        for l in range(level +1):
            os.makedirs(cls._out_analysis_runner_path
                         + cls._nomenclature_d2_analysis_runner.format(l))

    @classmethod
    def _init_outdir_d3_model_runner(cls,dimensions) -> None:
        for dim in dimensions:
            os.makedirs(cls._analytics_path+
                        cls._nomenclature_d3_model_runner.format(dim),exist_ok=True)
            os.makedirs(cls._validation_data_path+
                        cls._nomenclature_d3_model_runner.format(dim),exist_ok=True)
            os.makedirs(cls._test_data_path+
                        cls._nomenclature_d3_model_runner.format(dim),exist_ok=True)
            os.makedirs(cls._model_path+
                        cls._nomenclature_d3_model_runner.format(dim),exist_ok=True)

    @classmethod
    def _init_outdir_d4_model_runner(cls,dim,experiment_repetition,experiment_name,level):
        for l in range(level+1):
            os.makedirs(cls._analytics_path+
                        cls._nomenclature_d3_model_runner.format(dim)+
                        cls._nomenclature_d4_model_runner_leaf.format(l,experiment_repetition,experiment_name))
        os.makedirs(cls._analytics_path+
                    cls._nomenclature_d3_model_runner.format(dim)+
                    cls._nomenclature_d4_model_runner_root.format(level,experiment_repetition,experiment_name))

    @classmethod
    def _test_or_validation(cls,path_type,dim,experiment_repetition,experiment_name,level) -> str:
        if path_type == "test_kpis":
            path_data_x = cls._test_data_path +cls._nomenclature_d3_model_runner.format(dim) + cls._nomenclature_test_kpis_data_X.format(experiment_repetition,experiment_name,level)
            path_data_y = cls._test_data_path +cls._nomenclature_d3_model_runner.format(dim) + cls._nomenclature_test_kpis_data_Y.format(experiment_repetition,experiment_name,level)
        elif path_type == "test_plot":
            path_data_x = cls._test_data_path +cls._nomenclature_d3_model_runner.format(dim) + cls._nomenclature_test_plot_data_X.format(experiment_repetition,experiment_name,level)
            path_data_y = cls._test_data_path +cls._nomenclature_d3_model_runner.format(dim) + cls._nomenclature_test_plot_data_Y.format(experiment_repetition,experiment_name,level)
        elif path_type == "valid":
            path_data_x = cls._validation_data_path +cls._nomenclature_d3_model_runner.format(dim) + cls._nomenclature_valid_data_X.format(experiment_repetition,experiment_name,level)
            path_data_y = cls._validation_data_path +cls._nomenclature_d3_model_runner.format(dim) + cls._nomenclature_valid_data_Y.format(experiment_repetition,experiment_name,level)
        else:
            raise NotImplementedError("This type is not implemented! Choose either 'test' or 'valid'!")
        return path_data_x,path_data_y

    @classmethod
    def _check_is_initialized(cls):
        if not cls._outdir_initialized:
            raise ValueError("The 'outdir' is not initialized!")

    @staticmethod
    def _clear(path) -> None:
        for element in os.listdir(path):

            if os.path.isfile(path+"/"+element):
                os.remove(path+"/"+element)
            else:
                shutil.rmtree(path+"/"+element)

    @classmethod
    def _delete_outdir(cls) -> None:
        """Deletes the outdir in its current location.
        """
        cls._clear(cls._outdir)

        if os.getcwd() != os.path.dirname(cls._outdir):
            os.remove(cls._outdir)

    ########################
    ### Public Functions ###
    ########################



    @classmethod
    def init_dimension_related_directories(cls,dimensions) -> None:
        """Initializes directories related to basket dimensions.

        Args:
            dimensions (List): List of basket dimensions.
        """
        cls._check_is_initialized()
        cls._init_outdir_d3_model_runner(dimensions)

    @classmethod
    def init_analysis_related_directories(cls,level) -> None:
        """Initializes directories related to multilevel learning level.

        Args:
            level (int): Experiment level
        """
        cls._check_is_initialized()
        cls._init_outdir_d2_analysis_runner(level)

    @classmethod
    def init_experiment_related_directories(cls,dim,experiment_repetition,experiment_name,level) -> None:
        """Initializes directories related to specific experiments.

        Args:
            dim (int): Experiment basket dimension
            experiment_repetition (int): Experiment Repetition
            experiment_name (str): Experiment name
            level (int): Experiment level        
        """
        cls._init_outdir_d4_model_runner(
            dim = dim,
            experiment_repetition= experiment_repetition,
            experiment_name= experiment_name,
            level=level
        )

    @classmethod
    def get_tensorboard_path(cls) -> str:
        """Getter method for 'tensorboard' path.

        Returns:
            str: 'tensorboard' path
        """
        cls._check_is_initialized()
        return cls._tensorboard_path

    @classmethod
    def get_model_runner_analytics_path(cls) -> str:
        """Getter method for 'analytics' path in 'model_runner'.

        Returns:
            str: 'analytics' path.
        """
        cls._check_is_initialized()
        return cls._analytics_path

    @classmethod
    def get_model_runner_analytics_path_dim(cls,dim) -> str:
        """Getter method for 'analytics' path with specific dimension in 'model_runner'.

        Args:
            dim (int): basket dimension

        Returns:
            str: 'analytics' path with specific dimension
        """
        cls._check_is_initialized()
        return cls._analytics_path + cls._nomenclature_d3_model_runner.format(dim)

    @classmethod
    def get_model_runner_experiment_root_path(cls,dim,experiment_repetition,experiment_name,level) -> str:
        """Getter method for experiment specific 'root' directory

        Args:
            dim (int): Experiment basket dimension
            experiment_repetition (int): Experiment repetition
            experiment_name (str): Experiment name
            level (int): Experiment level  

        Returns:
            str: Experiment 'root' directory
        """
        return cls._analytics_path+cls._nomenclature_d3_model_runner.format(dim)+cls._nomenclature_d4_model_runner_root.format(level,experiment_repetition,experiment_name)


    @classmethod
    def get_nomenclature_d3_model_runner_dim(cls,dim) -> str:
        """Getter method for 'dim' string for experiments

        Args:
            dim (int): Experiment basket dimension

        Returns:
            str: Nomenclature of 'dim'
        """
        return cls._nomenclature_d3_model_runner.format(dim)


    @classmethod
    def get_nomenclature_d4_model_runner_root(cls,experiment_repetition,experiment_name,level) -> str:
        """Getter method for nomenclature of 'root' directory for experiments.

        Args:
            experiment_repetition (int): Experiment repetition
            experiment_name (str): Experiment name
            level (int): Experiment level  

        Returns:
            str: Nomenclature of 'root' directory
        """
        return cls._nomenclature_d4_model_runner_root.format(level,experiment_repetition,experiment_name)

    @classmethod
    def get_nomenclature_analysis_plot(cls,analysis_method,gen_data_conf):
        """Getter method for nomenclature of analysis method.

        Args:
            analysis_method (str): Analysis Method
            gen_data_conf (int): GenData configuration

        Returns:
            str: Nomenclature of analysis method
        """
        return cls._nomenclature_analysis_plot.format(analysis_method,gen_data_conf)

    @classmethod
    def get_model_runner_experiment_leaf_path(cls,dim,experiment_repetition,experiment_name,level) -> str:
        """Getter method for experiment specific 'leaf' directory

        Args:
            dim (int): Experiment basket dimension
            experiment_repetition (int): Experiment repetition
            experiment_name (str): Experiment name
            level (int): Experiment level  

        Returns:
            str: Experiment 'leaf' directory
        """
        return cls._analytics_path+cls._nomenclature_d3_model_runner.format(dim)+cls._nomenclature_d4_model_runner_leaf.format(level,experiment_repetition,experiment_name)


    @classmethod
    def get_analysis_runner_path(cls) -> str:
        """Getter method for 'analysis_runner' path

        Returns:
            str: 'analysis_runner' path
        """
        cls._check_is_initialized()
        return cls._out_analysis_runner_path

    @classmethod
    def get_analysis_runner_level_path(cls,level) -> str:
        """Getter method for 'analysis_runner' path with respect to level

        Args:
            level (int): Experiment level

        Returns:
            str: 'analysis_runner' path with respect to level
        """
        cls._check_is_initialized()
        return cls._out_analysis_runner_path + cls._nomenclature_d2_analysis_runner.format(level)

    @classmethod
    def get_cfg_path(cls) -> str:
        """Getter method for 'config' path

        Returns:
            str: 'config' path
        """
        cls._check_is_initialized()
        return cls._cfg_abs_path

    @classmethod
    def get_test_path(cls) -> str:
        """Getter method for 'test' path

        Returns:
            str: 'test' path
        """
        cls._check_is_initialized()
        return cls._test_abs_path

    @classmethod
    def _get_model_path(cls,experiment_name,level,dim) -> str:
        cls._check_is_initialized()
        return cls._model_path + cls._nomenclature_d3_model_runner.format(dim)+ cls._nomenclature_model.format(experiment_name,level)

    @classmethod
    def model_exists(cls,experiment_name,level,dim) -> bool:
        """Checks if a model file exists.

        Args:
            experiment_name (str): Experiment name
            level (int): Experiment level
            dim (int): Experiment basket dimension

        Returns:
            bool: Flag model exists
        """
        cls._check_is_initialized()
        exists = os.path.exists(cls._get_model_path(experiment_name,level,dim))
        return exists

    @classmethod
    def model_load(cls,experiment_name,level,dim) -> tf.keras.Model:
        """Loads model from file.

        Args:
            experiment_name (str): Experiment name
            level (int): Experiment level
            dim (int): Experiment basket dimension
        Returns:
            tf.keras.Model: Model instance.
        """
        cls._check_is_initialized()
        return tf.keras.models.load_model(cls._get_model_path(experiment_name,level,dim),compile = False)

    @classmethod
    def model_export(cls,model,experiment_name,level,dim) -> None:
        """Export model to file

        Args:
            model (tf.keras.Model): Model instance.
            experiment_name (str): Experiment name
            level (int): Experiment level
            dim (int): Experiment basket dimension
        """
        cls._check_is_initialized()
        model.save(cls._get_model_path(experiment_name,level,dim))


    @classmethod
    def tensor_data_exists(cls,experiment_repetition,experiment_name,level,path_type,dim) -> bool:
        """Checks if tensor files exist.

        Args:
            experiment_repetition (int): Experiment repitition
            experiment_repetition (str): Experiment name
            level (int): Experiment level
            path_type (str): 'test' or 'valid' data
            dim (int): Experiment basket dimension

        Returns:
            bool: Flag tensor exists
        """
        path_data_x,path_data_y = cls._test_or_validation(path_type,dim,experiment_repetition,experiment_name,level)
        x_exists = os.path.exists(path_data_x)
        y_exists = os.path.exists(path_data_y)

        if x_exists and y_exists and (cls._use_exported_tensors or path_type != 'valid'):
            return True

        else:
            if x_exists:
                os.remove(path_data_x)
            if y_exists:
                os.remove(path_data_y)
            return False

    @classmethod
    def tensor_data_load(cls,experiment_repetition,experiment_name,level,dtype,path_type,dim) -> tf.Tensor:
        """Loads tensors from file.

        Args:
            experiment_repetition (int): Experiment repetition
            level (int): Experiment level
            dtype (str): Tensor data type
            path_type (str): 'test' or 'valid' data
            dim (int): Experiment basket dimension

        Returns:
            tf.Tensor: Input and true output data
        """
        path_data_x,path_data_y = cls._test_or_validation(path_type,dim,experiment_repetition,experiment_name,level)
        X = tf.io.read_file(path_data_x)
        Y = tf.io.read_file(path_data_y)

        X = tf.io.parse_tensor(X,out_type=dtype)
        Y = tf.io.parse_tensor(Y,out_type=dtype)

        return X,Y

    @classmethod
    def tensor_data_export(cls,experiment_repetition,experiment_name,X,Y,level,path_type,dim) -> None:
        """Exports tensors to file

        Args:
            experiment_repetition (int): Experiment Repetition
            X (tf.Tensor): Input tensor
            Y (tf.Tensor): True output tensor
            level (int): Experiment level
            path_type (str): 'test' or 'valid' data
            dim (int): Experiment basket dimensions
        """
        path_data_x,path_data_y = cls._test_or_validation(path_type,dim,experiment_repetition,experiment_name,level)
        x_serialized = tf.io.serialize_tensor(X)
        y_serialized = tf.io.serialize_tensor(Y)

        tf.io.write_file(path_data_x,x_serialized)
        tf.io.write_file(path_data_y,y_serialized)
