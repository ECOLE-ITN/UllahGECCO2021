[![DOI](https://zenodo.org/badge/404800705.svg)](https://zenodo.org/badge/latestdoi/404800705)



# Comparison of Moment-Generating Function of Improvement and Expected Improvement Criterion for Robust Bayesian Optimization
Contains code for the empirical comparison of two acquisition functions, namely the so-called Moment-Generating Function of the Improvement (MGFI) and
the Expected Improvement Criterion (EIC). The performance comparison is based on the scenarios of optimization under uncertainty aka Robust Optimization, where the focus
is to find solutions immune to the uncertainty in the decision/search variables. In the comparison, EIC serves as a baseline.

# Introduction
This code is based on our paper, titled [A New Acquisition Function for Robust Bayesian
Optimization of Unconstrained Problems](https://dl.acm.org/doi/pdf/10.1145/3449726.3463206) (Ullah, Wang, Menzel, Sendhoff & Bäck, 2021), and can be used to reproduce
the experimental setup and results mentioned in the paper. The code is produced in Python 3.7.0. The main packages utilized in this code are presented in the next section which deals with technical requirements. 

The code is saved in the directory, which is titled `Baseline Comparison`. This directory contains four other directories, which are
named after four test problems considered in the paper. The name of these directroeis are `Branin`, `Eight-Dimensional`, `Ten-Dimensional`, and `Three-Dimensional` respectively. 
Each of these directories contains further two sub-directories, namely `EIC` and `MGFI`.
Logically, these sub-directories contain the implementation of the corresponding acquisition functions.
The implementation of the acquisition function is based on the noise level chosen, where the noise level describes the scale of uncertainty, e.g., how
much uncertain each decision/search variables is? In the paper, we deal with three increasing noise levels.
The nomenclature of the subdirectories in `EIC` and `MGFI` is based on these three noise levels, namely `Noise Level1`, `Noise Level2`, and `Noise Level3` respectively.
Finally, each noise level implementation contains two files, namely `Utils.py` and `RBO.ipynb`.
As the name suggests, the `Utils.py` contains the utility functions to run Robust Bayesian Optimization (RBO), which is ececuted in `RBO.ipynb`.
In the following, we describe the technical requirements as well the instructions to run the code in a sequential manner.


## Requiremnts
In this code, we make sue of four python packages (among others), which are presented below in the table.
In particular, `pyDOE2` can be utilized for sampling plans and Design of Experiment (DoE).
For the same purpose, `SMT` package may also be used.
We employ the so-called `Latin Hypercube Sampling` based on the `SMT` package.  
For the purpose of numerical optimization in the code, e.g., to maximize the acquisition function, we utilize the famous `L-BFGS` algorithm based on `SciPy` package.
Finally, the main purpose of the `scikit-learn` package is to construct the Kriging surrogate, as well as data manipulation/wrangling in genera 
All four required packages can be installed by executing `pip install -r requirements.txt` from the main directory via the command line.

| Package | Description |
| --- | --- |
| pyDOE2 | For sampling plans and Design of Experiment (DoE).  |
| SciPy |For numerical optimization based on L-BFGS-B algorithm. |
| SMT |Surrogate Modeling Toolbox utilized for the implementation of Branin Function. |
| scikit-learn | For constructing the Kriging surrogate, as well as data manipulation. |

In the following, we describe how to reproduce the experimental setup and results mentioned in our paper.

## 1. Setting up the Test Problem
The first step in the paper is to determine the test problem, and noise level (the scale of uncertainty).
The scale of uncertainty is based on three different levels as stated earlier, which represent 5, 10, and 20 % deviation in the nominal values of the decision variables.
Based on the chosen test problem and noise level, we setup the global parameters in our code which characterize the search space, e.g., bounds.
The global parameters have to be set manually in the file `RBO.ipynb` based on the test problem, and noise level.
Note that same parameters setting should be present for both acquisition functions.

## 2. Choose Initial Samples based on DoE
After the global parameters have been set, we utilize the method `DOE` from `Utils.py` to retrieve the initial sampling locations.
The sampling locations are based on the `Latin Hypercube Sampling` scheme, which can be implemented by the `SMT` or the `pyDOE2` package.

## 3. Run Robust Bayesian Optimization
After retrieving the sampling coordinates to observe the function response, we have to use the `bayesian_optimisation` method in `RBO.ipynb`
to run the RBO. This refers to observing the function response, as well constructing and updating the Kriging surrogate in a sequential manner based
on the chosen acquisition function. This method returns the robust optimum, which can be stored if desirable. RBO is run for 25 independent runs
and the averaged results are compared.

## Citation
## Paper Reference
Sibghat Ullah, Hao Wang, Stefan Menzel, Bernhard Sendhoff, and Thomas Bäck. 2021. A New Acquisition Function for Robust Bayesian Optimization
of Unconstrained Problems. In 2021 Genetic and Evolutionary Computation Conference Companion (GECCO ’21 Companion), July 10–14, 2021, Lille,
France. 
## BibTex Reference
`@inproceedings{ullah2021new,`\
  `title={A new acquisition function for robust Bayesian optimization of unconstrained problems},`\
  `author={Ullah, Sibghat and Wang, Hao and Menzel, Stefan and Sendhoff, Bernhard and B{\"a}ck, Thomas},`\
  `booktitle={Proceedings of the Genetic and Evolutionary Computation Conference Companion},`\
  `pages={1344--1345},`\
  `year={2021}`\
`}`

## Acknowledgements
This research has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement number 766186 (ECOLE).
