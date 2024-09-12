"""Main Module

Executes the tests, analysis and/or experiments. Depending on configuration.
"""

import sys
import os
import argparse

from src.utils.filesystem import FileSystem
from src.model.model_runner import ModelRunner
from src.analysis.analysis_runner import AnalysisRunner


#tensorboard --logdir framework/tensorboard

def setup(keep_data) -> bool:
    """Initializes required functionalities for experiments.

    Args:
        keep_data (bool): Determines if validation data is reused if existing. Defaults to False.
    """
    FileSystem.init_class(keep_data)


def execute_analysis(debug):
    """Executes analysis.
    """
    analysis_runner = AnalysisRunner(
        debug= debug
    )
    analysis_runner.run()

def execute_model_runs(debug):
    """Executes experiments.
    """
    model_runner = ModelRunner(
        tf_eager_mode= False,
        debug=debug
    )
    model_runner.run()



def main(debug=False, include_analysis=False,skip_experiments = False,keep_data=False):
    """Main function.

    Args:
        debug (bool): Determines if debug configuration is used during execution. Defaults to False.
        include_analysis (bool): Determines if analysis is executed. Defaults to False.
        skip_experiments (bool): Determines if experiments are skipped. Defaults to False.
        keep_data (bool): Determines if validation data is reused if existing. Defaults to False.
    """

    setup(keep_data)

    if include_analysis:
        execute_analysis(debug)

    if not skip_experiments:
        execute_model_runs(debug)


if __name__ == '__main__':
    dirname = os.path.dirname
    file_abs_path = dirname(dirname(os.path.abspath(__file__)))
    sys.path.append(file_abs_path)


    parser = argparse.ArgumentParser(
        prog="Monte Carlo Learning",
        description="This framework implements configurable Monte Carlo learning experiments!",

    )
    parser.add_argument('-d','--debug',action="store_true")
    parser.add_argument('-a','--analysis',action="store_true")
    parser.add_argument('-s','--skip_experiments',action="store_true")
    parser.add_argument('-k','--keep_data',action="store_true")
    args = parser.parse_args()

    main(debug = args.debug,
        include_analysis = args.analysis,
        skip_experiments=args.skip_experiments,
        keep_data=args.keep_data)
