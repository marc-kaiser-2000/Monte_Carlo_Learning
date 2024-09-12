import unittest
from unittest.mock import Mock

from src.utils.filesystem import FileSystem
from src.utils.log_manager import LogManager

from src.analysis.performance_batch_size import BatchSizePerformance
from src.analysis.performance_sample_size import SampleSizePerformance
from src.analysis.performance_full import FullPerformance
from src.analysis.trace_graph import TraceGraph
from src.analysis.mapping_gen_data import MappingGenData
from src.analysis.analysis_runner import AnalysisRunner


class AnRunnerSampleSize(unittest.TestCase):
    def setUp(self):
        
        FileSystem.init_class()
        
        self.mock = Mock()

        self.analysis_runner = AnalysisRunner()
        self.analysis_runner.batch_size_performance.run = lambda *args, **kwargs:None
        self.analysis_runner.full_performance.run = lambda *args, **kwargs:None
        self.analysis_runner.trace_graph.run = lambda *args, **kwargs:None
        self.analysis_runner.sample_size_performance.run = self.mock   

    def test_execution(self):
        self.analysis_runner.run()
        self.mock.assert_called()        

    def tearDown(self):
        LogManager.close_analysis_logger()
        FileSystem._delete_outdir()

class AnRunnerBatchSize(unittest.TestCase):
    def setUp(self):
        
        FileSystem.init_class()

        self.mock = Mock()

        self.analysis_runner = AnalysisRunner()
        self.analysis_runner.batch_size_performance.run = self.mock  
        self.analysis_runner.full_performance.run = lambda *args, **kwargs:None
        self.analysis_runner.trace_graph.run = lambda *args, **kwargs:None
        self.analysis_runner.sample_size_performance.run = lambda *args, **kwargs:None   

    def test_execution(self):
        self.analysis_runner.run()
        self.mock.assert_called()        

    def tearDown(self):
        LogManager.close_analysis_logger()
        FileSystem._delete_outdir()

class AnRunnerFullPerformance(unittest.TestCase):
    def setUp(self):

        FileSystem.init_class()
        
        self.mock = Mock()

        self.analysis_runner = AnalysisRunner()
        self.analysis_runner.batch_size_performance.run = lambda *args, **kwargs:None
        self.analysis_runner.full_performance.run = self.mock 
        self.analysis_runner.trace_graph.run = lambda *args, **kwargs:None
        self.analysis_runner.sample_size_performance.run = lambda *args, **kwargs:None  

    def test_execution(self):
        self.analysis_runner.run()
        self.mock.assert_called()        

    def tearDown(self):
        LogManager.close_analysis_logger()
        FileSystem._delete_outdir()

class AnRunnerTraceGraph(unittest.TestCase):
    def setUp(self):

        FileSystem.init_class()
        
        self.mock = Mock()

        self.analysis_runner = AnalysisRunner(debug=True)
        self.analysis_runner.batch_size_performance.run = lambda *args, **kwargs:None
        self.analysis_runner.full_performance.run = lambda *args, **kwargs:None
        self.analysis_runner.trace_graph.run = self.mock  
        self.analysis_runner.sample_size_performance.run = lambda *args, **kwargs:None

    def test_execution(self):
        self.analysis_runner.run()
        self.mock.assert_called()        

    def tearDown(self):
        LogManager.close_analysis_logger()
        FileSystem._delete_outdir()