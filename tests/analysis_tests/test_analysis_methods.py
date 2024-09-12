import unittest
from unittest.mock import Mock

from src.analysis.performance_batch_size import BatchSizePerformance
from src.analysis.performance_sample_size import SampleSizePerformance
from src.analysis.performance_full import FullPerformance
from src.analysis.trace_graph import TraceGraph
from src.analysis.mapping_gen_data import MappingGenData
from src.analysis.analysis_runner import AnalysisRunner

from src.utils.filesystem import FileSystem
from src.utils.log_manager import LogManager


class TraceGraph(unittest.TestCase):


    def _count_true_in_fnc(self,fnc_pointer,iterator)->int:
        true_count = 0
        for i in range(iterator):
            if fnc_pointer(i):
                true_count += 1

        return true_count


    def setUp(self):

        self.mock = Mock()

        FileSystem.init_class()

        self.analysis_runner = AnalysisRunner(debug=True)
        self.no_gen_data_confs = self.analysis_runner.no_gen_data_confs
        self.trace_graph = self.analysis_runner.trace_graph
        self.trace_graph._mapping_gen_data.execute_gen_data = self.mock


    def test_call_count_execute_gen_data(self):
        iterator = self.analysis_runner.level
        fnc_pointer = self.analysis_runner.trace_graph._execute_level
        no_levels = self._count_true_in_fnc(fnc_pointer,iterator+1)

        self.trace_graph.run(
                level=self.analysis_runner.level,
                conf=1,
                dimensions=self.analysis_runner.dimensions,
            )


        self.assertTrue(self.mock.call_count == no_levels)
        
    def tearDown(self):
        LogManager.close_analysis_logger()
        FileSystem._delete_outdir()


class BatchSize(unittest.TestCase):



    def setUp(self):

        self.mock = Mock()
        FileSystem.init_class()

        self.analysis_runner = AnalysisRunner(debug=True)
        self.analysis_runner.batch_size_performance._mapping_gen_data.execute_gen_data = self.mock


    def test_call_count_execute_gen_data(self):
        dimensions = [10,50]
        level = 1
        conf = 1
        batch_size_start = 2
        batch_size_end = 3
        step_size = 1
        
        executions_per_bs = 2

        estimated_cnt = len(dimensions)*(level+1)*step_size*executions_per_bs
        
        self.analysis_runner.batch_size_performance.run( 
                    level=level,
                    conf=conf,
                    batch_size_start=batch_size_start,
                    batch_size_end=batch_size_end,
                    step_size=step_size,
                    dimensions=dimensions)

        self.assertTrue(self.mock.call_count == estimated_cnt)
        
    def tearDown(self):
        LogManager.close_analysis_logger()
        FileSystem._delete_outdir()


class SampleSize(unittest.TestCase):


    def _count_true_in_fnc(self,fnc_pointer,iterator)->int:
        true_count = 0
        for i in range(iterator):
            if fnc_pointer(i):
                true_count += 1

        return true_count


    def setUp(self):

        self.mock = Mock()
        FileSystem.init_class()

        self.analysis_runner = AnalysisRunner(debug=True)
        self.sample_size_performance = self.analysis_runner.sample_size_performance
        self.sample_size_performance._mapping_gen_data.execute_gen_data = self.mock


    def test_call_count_execute_gen_data_conf_1(self):
        dimensions = [10,50]
        level = 1
        conf = 1
        sample_size_minimum = 1
        sample_size_maximum = 2


        no_levels = self._count_true_in_fnc(self.sample_size_performance._execute_level,level + 1)
        exec_conf = self.sample_size_performance._execute_gen_data_conf(conf)
        
        if exec_conf:
            estimated_cnt = len(dimensions)*no_levels*4
        else:
            estimated_cnt = 0
        self.analysis_runner.sample_size_performance.run( 
                    level=level,
                    conf=conf,
                    sample_size_minimum=sample_size_minimum,
                    sample_size_maximum=sample_size_minimum,
                    dimensions=dimensions)

        self.assertTrue(self.mock.call_count == estimated_cnt)

    def test_call_count_execute_gen_data_conf_2(self):
        dimensions = [10,50]
        level = 1
        conf = 2
        sample_size_minimum = 1
        sample_size_maximum = 2


        no_levels = self._count_true_in_fnc(self.sample_size_performance._execute_level,level + 1)
        exec_conf = self.sample_size_performance._execute_gen_data_conf(conf)
        
        if exec_conf:
            estimated_cnt = len(dimensions)*no_levels*2
        else:
            estimated_cnt = 0
        self.analysis_runner.sample_size_performance.run( 
                    level=level,
                    conf=conf,
                    sample_size_minimum=sample_size_minimum,
                    sample_size_maximum=sample_size_minimum,
                    dimensions=dimensions)

        self.assertTrue(self.mock.call_count == estimated_cnt)
        
    def tearDown(self):
        LogManager.close_analysis_logger()
        FileSystem._delete_outdir()


class FullPerformance(unittest.TestCase):



    def setUp(self):

        self.mock = Mock()
        FileSystem.init_class()

        self.analysis_runner = AnalysisRunner(debug=True)
        self.full_performance = self.analysis_runner.full_performance
        self.full_performance._mapping_gen_data.execute_gen_data = self.mock


    def test_call_count_execute_gen_data_conf_1(self):
        dimensions = [10,50]
        level = 1
        conf = 1        

        exec_conf = self.full_performance._execute_gen_data_conf(conf)
        
        if exec_conf:
            estimated_cnt = 2
        else:
            estimated_cnt = 0
        
        self.full_performance.run( 
                    level=level,
                    conf=conf,
                    batch_size = 500,
                    sample_size = 10,
                    dimensions=dimensions)

        self.assertTrue(self.mock.call_count == estimated_cnt)

    def test_call_count_execute_gen_data_conf_2(self):
        dimensions = [10,50]
        level = 3
        conf = 2        

        exec_conf = self.full_performance._execute_gen_data_conf(conf)
        
        if exec_conf:
            estimated_cnt = (level+1)*2 
        else:
            estimated_cnt = 0
        
        self.full_performance.run( 
                    level=level,
                    conf=conf,
                    batch_size = 500,
                    sample_size = 10,
                    dimensions=dimensions)

        self.assertTrue(self.mock.call_count == estimated_cnt)
        
    def tearDown(self):
        LogManager.close_analysis_logger()
        FileSystem._delete_outdir()

    