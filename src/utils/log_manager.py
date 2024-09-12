"""Interface to logger
"""
import logging

from logging.handlers import RotatingFileHandler
from logging import StreamHandler
from src.utils.filesystem import FileSystem

class LogManager:
    """Class is build upon the 'logging' python library.
    The provided methods implement create loggers for the specific sub-packagess.
    """
    _active_dim = None
    _active_experiment_repetition = None
    _active_experiment_name = None
    _active_level = None

    @classmethod
    def initialize_experiment_logger(cls,dim,experiment_repetition,experiment_name,level) -> None:
        """Initializes logger class for experiments.

        Args:
            dim (int): Experiment basket dimension
            experiment_key (int): Experiment key
            experiment_name (str): Experiment name
            level (int): Experiment level  
        """

        cls._active_dim = dim
        cls._active_experiment_repetition = experiment_repetition
        cls._active_experiment_name = experiment_name
        cls._active_level = level

        path = FileSystem.get_model_runner_experiment_root_path(
            dim = dim,
            experiment_repetition= experiment_repetition,
            experiment_name=experiment_name,
            level=level
        )

        logger = cls._create_logger(
            path = path,
            filename = "/experiment.log"
        )
        return logger

    @classmethod
    def get_experiment_logger(cls):
        path = FileSystem.get_model_runner_experiment_root_path(
            dim = cls._active_dim,
            experiment_repetition= cls._active_experiment_repetition,
            experiment_name=cls._active_experiment_name,
            level=cls._active_level
        )
        return logging.getLogger(path)
    
    @classmethod
    def close_experiment_logger(cls):
        logger = cls.get_experiment_logger()
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)

    @classmethod
    def initialize_analysis_logger(cls) -> logging.Logger:
        """Initializes logger class for analysis.
        """
        path = FileSystem.get_analysis_runner_path()

        logger = cls._create_logger(
            path = path,
            filename = "/analysis.log"
        )
        return logger
    
    @classmethod
    def get_analysis_logger(cls):
        path = FileSystem.get_analysis_runner_path()
        return logging.getLogger(path)
    
    @classmethod
    def close_analysis_logger(cls):
        logger = cls.get_analysis_logger()
        handlers = logger.handlers[:]
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)

    @staticmethod
    def _create_logger(path,filename) -> logging.Logger:

        logger = logging.getLogger(path)
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        file_handler = RotatingFileHandler(
            filename=path+filename,
            maxBytes=1024*1024,
            backupCount=5,
            encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)        
        file_handler.setFormatter(formatter)

        console_handler = StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger