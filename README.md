# Deep_ML_Learning

This software was designed and implemented during the developers matster's thesis at the Frankfurt University of Applied Sciences.
An Nvidia A100-SXM40-80GB GPU was provided for development purposes during the thesis.

This software enables the execution of Monte Carlo learning experiments.
These can be seperated into singlelevel Monte Carlo (SLMC) and multilevel Monte Carlo (MLMC) learning experiments.
The required theoretical background for SLMC learning is given by Beck et al. (https://arxiv.org/abs/1806.00421).
Additionally the review (https://arxiv.org/abs/2102.11802) and provided code (https://github.com/janblechschmidt/PDEsByNNs) of Blechschmidt and Ernst was also used.
MLMC learning was invented by Gerstner et al. (https://arxiv.org/abs/2102.08734).

This software enables the sequential exeuction of Monte Carlo learning experiments and the individual configuration of the hyperparameters.
While only a small set of possible hyperparameters was implemented, due to the design of software, the available hyperparameters can be enhanced to the required needs.
Provided features are:
+ Easy configuration of additional hyperparameters
+ Comparability of SLMC and MLMC approaches (as long as the remaining hyperparameters are equal)
+ Transfer Learning (not useful for all hyperparameter changes)



## Installation


Clone the repository:

    https://github.com/marc-kaiser-2000/Monte_Carlo_Learning.git


Install Python version 3.10.12 or create a virual environment with this version.
If the software shall be executed on a GPU CUDA version 12.2 is required.
Then install the requirements for the GPU.

    pip install -r requirements_gpu.txt

If the software is executed on a CPU, install the requirements for the CPU

    pip install -r requirements_cpu.txt

It is recommended to use the software with a GPU, since computations in the current setting are numerically heavy.
The experiments were conducted usign a Nvidia A100-SXM40-80GB and took in total about 1.5 weeks to be executed.
Hence, the software should only be executed on a CPU only in debug mode.

## Usage

The software is started using the following commands. 

On Windows:
    
    python main.py

And on Linux:

    python3 main.py

This command will execute all experiments specified in the 'modelrunner.py' file.
The main file may be called with additional parameters
+ -d / --debug  : Executes the experiments ('model' sub-package) and/or 'analysis' sub-package in debug mode.
+ -a / --analysis : Executes the 'analysis' sub-package.
+ -s / --skip_experiments : Disables execution of the experiments and thus 'model' sub-package.
+ -k / --keep_data : Reuses already generated validation data for experiments, if data exists.

After enhancing existing hyperparameters, it is useful to check if nothing in the code has unintentionally been broken.
The unit-tests can therefore be executed using the following command:

    python -m unittest discover -v -s . -p test*.py

The created outputs can be found in the 'out_dir'.
More specifically:

+ out_analysis_runner : Contains the results of the 'analysis' sub-package
+ out_model_runner : Contains the results of the 'model'sub-package
+ tensorboard : Contains the outputs of the 'analysis' sub-package which require the tensorboard.

The outputs can be interpreted as follows.
The 'analysis_runner.py' tests the execution of the 'GenData' configurations until level 3 and saves the results in the 'out_analysis_runner' directory.
For each level exists a directory.
The PDFs provided represent the execution duration with either increasing sample size or batch size.
The 'analysis.log' displays logged events and the duration of a function execution under heavy computational requirements.

The 'out_model_runner' directory is also further subdivided.
The 'model' directory contains the trained and exported models for a respective basket size.
The 'test_data_path' and 'validation_data_path' store the exported test and validation data.
The 'analysis_path' directory is also separated into the respective basket size.
For each executed experiment $L$ Leaf directories are created.
These directories contain the absolute and relative errors generated during the validation in a early-stopping like manner.
Each experiment has a Root directory.
This directory contains a slice of the solution, the calculated relative error and the 'experiment.log' of the experiment.

The contents of the 'tensorboard' can be investigated using

    tensorboard --logdir outdir/tensorboard

This will start a webserver on this directory, which parses the generated files.
This visualization toolkit is provided by TensorFlow.


## Enhancments

How can this framework be enhanced to enable individual experiment configurations?

In the source 'configs/runconfigs' exists a JSON file for each experiment.
Given the file 'run_config_0_debug.json' , where '0' is the experiment repretition and 'debug' the experiment name.
Increasing the repetition and keepting the same name (run_config_1_debug.json) enables the usage of transfer learning.

```json
    {
    "model_type" : "multi_level", 
    "max_level" : 3,
    "sample_size_test" : 0,
    "sample_size_valid" : 0,
    "batch_size_test" : 0,
    "batch_size_valid" : 0, 
    "gen_data" : 4,
    "optimizer" : 0,
    "learning_rate" : 1,
    "loss_and_grad" : 0,
    "model_topology" : 0,
    "asset_correlation" : 0,
    "dynamics" : 0
    
}
```
Each key-value pair, except for the first two, refers to one file in the 'modelparameter' directory.
The 'model_type' is an option and can switch between 'multi_level' and 'single_level'.
The 'max_level' defines the number of neural networks for 'multilevel' experiments.
For 'singlelevel' experiments it only defines the time discretization refinement.
For each other pair a dedicated hyperparameter (hp) file exists in the 'modelparameter' directory.
Further configurations can here be added.
The 'asset_correlation' refers to the 'constants_hp.py' in which the correlation matrix is defined.
The 'dynamics' key refers to 'batch_size_train_hp.py','sample_size_train_hp.py', 'error_size_hp.py' and the heuristic in the 'hyperparameter_manager.py'.
For these only a combined key exists, since they strongly depent on each other.

These '*\_hp.py' files may be enhanced by additional configurations.
Additional '\_*hp.py' files can also be added, by creating a similar structured file and inheriting from the 'abstract_hyperparameter.py'.
The new hyperparameter has then needs to be configured in the 'hyperparameter_manager.py'
It furthermore seems reasonable to then also enhance the tests to reduce the chances for an error.


## Project Structure 

The following tree shows the directory structure of this project.

```plaintext
.
├── src/
│   ├── analysis/
│   │   ├── analysis_runner.py
│   │   └── ...
│   ├── configs/
│   │   └── runconfigs/
│   │       ├── conf_1.json
│   │       └── ...
│   ├── model/
│   │   ├── modelcomponents/
│   │   │   ├── abstract_model.py
│   │   │   ├── leaf_model.py
│   │   │   └── root_model.py
│   │   ├── modelprameters/
│   │   │   ├── abstract_hyperparameter.py
│   │   │   ├── hyperparameter_manager.py
│   │   │   └── ...
│   │   └── model_runner.py
│   └── utils/
│       └── ...
├── tests/
│   ├── analysis_tests/
│   │   └── ...
│   ├── config_tests/
│   │   └── ...
│   ├── model_tests/
│   │   └── ...
│   └── utils_tests/
│       └── ...
└── (outdir)/
    ├── out_analysis_runner/
    │   ├── Level_0/
    │   │   ├── Analysis_Batch_Size_Gen_Data_Conf_*.pdf
    │   │   ├── Analysis_Sample_Size_Abs_Gen_Data_Conf_*.pdf
    │   │   └── Analysis_Sample_Size_Rel_Gen_Data_Conf_*.pdf
    │   ├── Level_1/
    │   │   └── ...
    │   ├── Level_2/
    │   │   └── ...
    │   ├── Level_3/
    │   │   └── ...
    │   └── analysis.log
    ├── out_model_runner/
    │   ├── analytics_path/
    │   │   ├── dim_10/
    │   │   │   ├── Leaf_Level_0_Repetition_0_Name_debug/
    │   │   │   │   ├── Error_OptionPricing_Abs.pdf
    │   │   │   │   └── Error_OptionPricing_Rel.pdf
    │   │   │   ├── ...
    │   │   │   ├── Leaf_Level_3_Repetition_0_Name_debug/
    │   │   │   │   └── ...
    │   │   │   └── Root_Level_3_Repetition_3_Name_debug/
    │   │   │       ├── experiment.log
    │   │   │       ├── RelError_OptionPricing.pdf
    │   │   │       └── Sol_OptionPricing.pdf
    │   │   ├── dim_50
    │   │   └── dim_100
    │   ├── model/
    │   │   └── ...
    │   ├── test_data_path/
    │   │   └── ...
    │   └── validation_data_path/
    │       └── ...
    └── tensorboard/
        └── ...
```

## License
This project is licensed under the MIT License - see the LICENSE file for details


