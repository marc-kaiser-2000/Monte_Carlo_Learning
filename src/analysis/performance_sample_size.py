import numpy as np
from datetime import datetime

from src.analysis.mapping_gen_data import MappingGenData
from src.model.modelparameter.constants_hp import Constants
from src.utils.plot import Plot
from src.utils.filesystem import FileSystem
from src.utils.log_manager import LogManager

class SampleSizePerformance:
    """Analysis Method used to analyze execution duration for increasing sample size.
    Class encapsulates method and provides auxiliary functions.
    """

    _mapping_gen_data = MappingGenData()

    def _execute_gen_data_conf(self,conf):
        options = {
            1: False, # THIS CALL WILL CAUSE A GPU EXHAUSTION AND CRASH! VALIDATE BY CHANGING: False ==> True
            2: True,
            3: False, # THIS CALL WILL CAUSE A GPU EXHAUSTION AND CRASH! VALIDATE BY CHANGING: False ==> True
            4: True,
            5: False, # Similar to 4, due to comp complexity analysis avoided
            6: False, # Similar to 4, due to comp complexity analysis avoided
        }
        return options.get(conf)

    def _execute_level(self,level):
        options = {
            0:True,
            1:False,
            2:False,
            3:True,
        }
        return options.get(level)


    def run(self,level,conf,sample_size_minimum,sample_size_maximum,dimensions):
        """Executes analysis method:
        Analyzes duration of function execution for minimum and maximum sample size.

        Args:
            level (int): Level of time discretization
            conf (int): GenData configuration
            sample_size_minimum (int): Minimum sample size investigated
            sample_size_maximum (int): Maximum sample size investigated
            dimensions (List[int]): List of sizes of option basket
        """


        if not self._execute_gen_data_conf(conf):
            return

        logger = LogManager.get_analysis_logger()
        logger.info(f"Run 'SampleSizePerformance' on Configuration {conf}")

        for sublevel in range(level+1):

            if not self._execute_level(sublevel):
                continue
            logger.info(f"Level {sublevel}")

            sample_size_interval = [sample_size_minimum,sample_size_maximum]
            gen_data_execution_time_absolute = []
            gen_data_execution_time_relative = []

            for dim in dimensions:

                gen_data_execution_time_absolute.append([])
                gen_data_execution_time_relative.append([])

                constants = Constants(dim=dim,asset_correlation_conf=0)
                for sample_size in sample_size_interval:

                    t_start = datetime.now()
                    self._mapping_gen_data.execute_gen_data(
                        conf=conf,
                        constants=constants,
                        level = sublevel,
                        batch_size=1,
                        sample_size=sample_size,
                        num_timestep=constants.N**sublevel,
                        dim = dim
                    )
                    t_end = datetime.now()
                    gen_data_execution_time_absolute[-1].append((t_end-t_start).total_seconds())
                    gen_data_execution_time_relative[-1].append((t_end-t_start).total_seconds()/sample_size)

            path = FileSystem.get_analysis_runner_level_path(sublevel) + FileSystem.get_nomenclature_analysis_plot("Sample_Size_Abs",conf)
            Plot.plot_analysis_results_bar(
                x_mc_samples=sample_size_interval,
                y_data=gen_data_execution_time_absolute,
                title=f"GenData Conf {conf} Execution Duration",
                y_label="Time T in Seconds",
                path = path
            )
            path = FileSystem.get_analysis_runner_level_path(sublevel) + FileSystem.get_nomenclature_analysis_plot("Sample_Size_Rel",conf)
            Plot.plot_analysis_results_bar(
                x_mc_samples=sample_size_interval,
                y_data=gen_data_execution_time_relative,
                title=f"GenData Conf {conf} Duration for a single Sample",
                y_label="Time T in Seconds",
                path = path
            )
