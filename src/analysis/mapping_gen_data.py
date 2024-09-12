import tensorflow as tf

from src.model.modelparameter.gen_data_hp import gen_data_tf_const
from src.model.modelparameter.gen_data_hp import gen_data_python_const
from src.model.modelparameter.gen_data_hp import gen_data_opt_time,gen_data_opt_space
from src.model.modelparameter.gen_data_hp import gen_data_milstein_default_l0,gen_data_milstein_default_ln
from src.model.modelparameter.gen_data_hp import gen_data_milstein_control_variate_l0
from src.model.modelparameter.gen_data_hp import gen_data_euler_default_l0,gen_data_euler_default_ln


class MappingGenData:
    """Class maps the individual 'gen_data' configurations and provides a unified interface.
    """

    def __init__(self) -> None:
        self.options = {
                    1:self._execute_gen_data_conf_1,
                    2:self._execute_gen_data_conf_2,
                    3:self._execute_gen_data_conf_3,
                    4:self._execute_gen_data_conf_4,
                    5:self._execute_gen_data_conf_5,
                    6:self._execute_gen_data_conf_6
            }

    def execute_gen_data(self,conf,constants,level,batch_size,sample_size,num_timestep,dim):
        """Executes a specified configuration of the 'gen_data' method.

        Args:
            conf (int): GenData configuration
            constants (Constants): Instance providing the required variables.
            level (int): Level of time discretization
            batch_size (int): Batch Size for data generation
            sample_size (int): Sample size for data generation
            num_timestep (int): Number of steps per time interval
            dim (int): Size of option basket.
        """
        
        gen_data =self.options.get(conf)
        gen_data(constants,level,batch_size,sample_size,num_timestep,dim)

    def get_no_gen_data_confs(self):
        """Gets the number of 'gen_data' configurations

        Returns:
            int: Size of 'options'
        """
        return len(self.options)

    def _execute_gen_data_conf_1(self,constants,level,batch_size,sample_size,num_timestep,dim):
        if level == 0:
            X,y = gen_data_opt_time(
                            const = constants,
                            level = level,
                            n_points = batch_size,
                            mc_samples = sample_size,
                            num_timestep = num_timestep
            )
        else:
            X,y = gen_data_opt_space(
                            const = constants,
                            level = level,
                            n_points = batch_size,
                            mc_samples = sample_size,
                            num_timestep = num_timestep
            )

    def _execute_gen_data_conf_2(self,constants,level,batch_size,sample_size,num_timestep,dim):
        X,y = gen_data_tf_const(
            level = tf.constant(level,dtype = "int32"),
            n_points = tf.constant(batch_size,dtype = "int32"),
            mc_samples = tf.constant(sample_size,dtype = "int32"),
            num_timestep = tf.constant(num_timestep,dtype = "int32"),
            dim = tf.constant(dim,dtype = "int32"),
        )

    def _execute_gen_data_conf_3(self,constants,level,batch_size,sample_size,num_timestep,dim):
        X,y = gen_data_python_const(
            const = constants,
            level = level,
            n_points = batch_size,
            mc_samples = sample_size,
            num_timestep = num_timestep
        )


    def _execute_gen_data_conf_4(self,constants,level,batch_size,sample_size,num_timestep,dim):
        if level == 0:
            X,y = gen_data_milstein_default_l0(
                const = constants,
                n_points = batch_size,
                mc_samples = sample_size,
                num_timestep = num_timestep
            )
        else:
            X,y = gen_data_milstein_default_ln(
                const = constants,
                n_points = batch_size,
                mc_samples = sample_size,
                num_timestep = num_timestep
            )

    def _execute_gen_data_conf_5(self,constants,level,batch_size,sample_size,num_timestep,dim):
        if level == 0:
            X,y = gen_data_milstein_control_variate_l0(
                const = constants,
                n_points = batch_size,
                mc_samples = sample_size,
                num_timestep = num_timestep
            )
        else:
            X,y = gen_data_milstein_default_ln(
                const = constants,
                n_points = batch_size,
                mc_samples = sample_size,
                num_timestep = num_timestep
            )

    def _execute_gen_data_conf_6(self,constants,level,batch_size,sample_size,num_timestep,dim):
        if level == 0:
            X,y = gen_data_euler_default_l0(
                const = constants,
                n_points = batch_size,
                mc_samples = sample_size,
                num_timestep = num_timestep
            )
        else:
            X,y = gen_data_euler_default_ln(
                const = constants,
                n_points = batch_size,
                mc_samples = sample_size,
                num_timestep = num_timestep
            )
