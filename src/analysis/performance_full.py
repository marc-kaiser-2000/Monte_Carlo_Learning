from datetime import datetime

from src.utils.filesystem import FileSystem
from src.analysis.mapping_gen_data import MappingGenData
from src.model.modelparameter.constants_hp import Constants
from src.utils.log_manager import LogManager

class FullPerformance:
    """Analysis Method used to analyze execution duration 
    Class encapsulates method and provides auxiliary functions.
    """
    _mapping_gen_data = MappingGenData()

    def _execute_gen_data_conf(self,conf):
        options = {
            1: False, # THIS CALL WILL CAUSE A GPU EXHAUSTION AND CRASH! VALIDATE BY CHANGING: False ==> True
            2: True,
            3: False, # THIS CALL WILL CAUSE A GPU EXHAUSTION AND CRASH! VALIDATE BY CHANGING: False ==> True
            4: True,
            5: True,
            6: True,
        }
        return options.get(conf)

    def run(self,level,conf,batch_size,sample_size,dimensions):
        """Executes analysis method:
        Analyzes duration of function execution with heavy computational requirements (full load).

        Args:
            level (int): Level of time discretization
            conf (int): GenData configuration
            batch_size (int): Batch size
            sample_size (int): Sampel size
            dimensions (List[int]): List of sizes of option basket
        """
        dim = dimensions[-1]
        constants = Constants(dim,0)

        if not self._execute_gen_data_conf(conf):
            return

        logger = LogManager.get_analysis_logger()
        logger.info(f"Run 'FullPerformance' on Configuration {conf}")

        #level_interval = [0,level]

        for sublevel in range(level+1):

            logger.info("Start full perfomance execution!")
            for x in range(2):
            
                t_start_execution = datetime.now()
                self._mapping_gen_data.execute_gen_data(
                    conf=conf,
                    constants=constants,
                    level = sublevel,
                    batch_size=batch_size,
                    sample_size=sample_size,
                    num_timestep=constants.N**sublevel,
                    dim = dim
                )
            if x == 1:
                logger.info("End full perfomance execution!")
                t_end_execution = datetime.now()
                duration = t_end_execution-t_start_execution
                logger.info(f"Duration full performance execution {duration}.")
