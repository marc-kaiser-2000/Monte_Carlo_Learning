import tensorflow as tf

from src.utils.log_manager import LogManager
from src.utils.filesystem import FileSystem

from src.analysis.performance_batch_size import BatchSizePerformance
from src.analysis.performance_sample_size import SampleSizePerformance
from src.analysis.performance_full import FullPerformance
from src.analysis.trace_graph import TraceGraph
from src.analysis.mapping_gen_data import MappingGenData

class AnalysisRunner:
    """Interface class for for analysis sub-package.
    Executes multiple specified analysis methods.
    """

    def __init__(self,debug = False) -> None:
        self.debug = debug
        self.level = 3
        self.dimensions = [10,50,100]
        self.no_gen_data_confs = MappingGenData().get_no_gen_data_confs()

        self.trace_graph=TraceGraph()
        self.batch_size_performance = BatchSizePerformance()
        self.sample_size_performance = SampleSizePerformance()
        self.full_performance = FullPerformance()

        FileSystem.init_analysis_related_directories(self.level)


    def run(self):
        """Function executes specified analysis methods
        """
        logger = LogManager.initialize_analysis_logger()

        logger.info("Start Analysis")

        for conf in range(1,self.no_gen_data_confs+1,1):
            logger.info(f"GenData Configuration {conf}")

            if self.debug:
                self.trace_graph.run(
                    level=self.level,
                    conf=conf,
                    dimensions=self.dimensions,
                )
                
                self.batch_size_performance.run(
                    level=self.level,
                    conf=conf,
                    batch_size_start=1,
                    batch_size_end=10,
                    step_size=1,
                    dimensions=self.dimensions
                )
                self.sample_size_performance.run(
                    level=self.level,
                    conf=conf,
                    sample_size_minimum=10,
                    sample_size_maximum=100,
                    dimensions=self.dimensions
                )
                self.full_performance.run(
                    level=self.level,
                    conf=conf,
                    batch_size=500,
                    sample_size=10,
                    dimensions=self.dimensions
                )
                

            else:
                """
                self.trace_graph.run(
                    level=self.level,
                    conf=conf,
                    dimensions=self.dimensions,
                    writer = self.writer,
                    tensorboard_path = self.tensorboard_path
                )
                """
                self.batch_size_performance.run(
                    level=self.level,
                    conf=conf,
                    batch_size_start=1000,
                    batch_size_end=260000,
                    step_size=1000,
                    dimensions=self.dimensions
                )
                self.sample_size_performance.run(
                    level=self.level,
                    conf=conf,
                    sample_size_minimum=320000,
                    sample_size_maximum=1024000,
                    dimensions=self.dimensions
                )
                self.full_performance.run(
                    level=self.level,
                    conf=conf,
                    batch_size=50000,
                    sample_size=1000,
                    dimensions=self.dimensions
                )
        logger.info("End Analysis")
