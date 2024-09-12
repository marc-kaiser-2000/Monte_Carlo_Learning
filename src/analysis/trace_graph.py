import tensorflow as tf

from src.utils.log_manager import LogManager
from src.utils.filesystem import FileSystem
from src.model.modelparameter.constants_hp import Constants
from src.analysis.mapping_gen_data import MappingGenData

class TraceGraph:
    """Analysis Method used to generate a graph of a @tf.function.
    Class encapsulates method and provides auxiliary functions.
    """

    def __init__(self):
        self._batch_size = 5
        self._sample_size = 200
        self._mapping_gen_data = MappingGenData()
        self._tensorboard_path = FileSystem.get_tensorboard_path()

    def _execute_level(self,level):
        options = {
            0:True,
            1:False,
            2:False,
            3:True,
        }
        return options.get(level)

    def run(self,level,conf,dimensions):
        """Executes analysis method:
        Generates and exports graph of @tf.function

        Args:
            level (int): Level of time discretization
            conf (int): GenData configuration
            dimensions (List[int]): List of sizes of option basket
        """
        logger = LogManager.get_analysis_logger()
        logger.info(f"Run 'TraceGraph' on Configuration {conf}")

        dim = dimensions[-1]
        constants = Constants(dim,0)

        for sublevel in range(level+1):

            if not self._execute_level(sublevel):
                continue
            
            tf.summary.trace_on(graph=True, profiler=True,profiler_outdir=self._tensorboard_path)

            self._mapping_gen_data.execute_gen_data(
                    conf=conf,
                    constants=constants,
                    level = sublevel,
                    batch_size=self._batch_size,
                    sample_size=self._sample_size,
                    num_timestep=constants.N**sublevel,
                    dim = dim
                )
            writer =  tf.summary.create_file_writer(self._tensorboard_path)
            with writer.as_default():
                tf.summary.trace_export(
                    name=f"GenData_Configuration_{conf}_Level_{sublevel}",
                    step = 1
                    )
