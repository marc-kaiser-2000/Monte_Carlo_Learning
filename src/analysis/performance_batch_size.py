import numpy as np
from datetime import datetime


from src.analysis.mapping_gen_data import MappingGenData
from src.model.modelparameter.constants_hp import Constants
from src.utils.plot import Plot
from src.utils.filesystem import FileSystem
from src.utils.log_manager import LogManager

class BatchSizePerformance:
    """Analysis Method used to analyze execution duration under increasing batch size. 
    Class encapsulates method.
    """

    _mapping_gen_data = MappingGenData()


    def run(self,level,conf,batch_size_start,batch_size_end,step_size,dimensions):
        """Executes analysis method:
        Analyzes duration of function execution with increasing batch size.

        Args:
            level (int): Level of time discretization
            conf (int): GenData configuration
            batch_size_start (int): Minumum batch size
            batch_size_end (int): Maximum batch size
            step_size (int): Stepsize used for discretization between 'batch_size_start' and 'batch_size_end'.
            dimensions (List[int]): List of sizes of option basket
        """
        logger = LogManager.get_analysis_logger()
        logger.info(f"Run 'BatchSizePerformance' on Configuration {conf}")

        for sublevel in range(level+1):
            logger.info(f"Level {sublevel}")
            batch_size_interval = np.arange(batch_size_start,batch_size_end,step_size)
            batch_size_interval = batch_size_interval.tolist()
            gen_data_execution_time = []

            for dim in dimensions:

                gen_data_execution_time.append([])

                constants = Constants(dim=dim,asset_correlation_conf=0)
                for batch_size in batch_size_interval:

                    for no_execution in range(2):
                        t_start = datetime.now()
                        self._mapping_gen_data.execute_gen_data(
                            conf=conf,
                            constants=constants,
                            level = sublevel,
                            batch_size=batch_size,
                            sample_size=1,
                            num_timestep=constants.N**sublevel,
                            dim = dim
                        )
                        t_end = datetime.now()
                        if no_execution == 1:
                            gen_data_execution_time[-1].append((t_end-t_start).total_seconds())

            path = FileSystem.get_analysis_runner_level_path(sublevel) + FileSystem.get_nomenclature_analysis_plot("Batch Size",conf)
            Plot.plot_analysis_results_line(x_data=batch_size_interval,
                                            y_data=gen_data_execution_time,
                                            x_label=f"Batch Size",
                                            y_label="Time T in Seconds",
                                            title = f"GenData Conf {conf} Execution Duration",
                                            path=path)
