[![DOI](https://zenodo.org/badge/404800705.svg)](https://zenodo.org/badge/latestdoi/404800705)



# Comparison of Moment-Generating Function of Improvement and Expected Improvement Criterion for Robust Bayesian Optimization
Contains code for the empirical comparison of two acquisition functions, namely the so-called Moment-Generating Function of the Improvement (MGFI) and
the Expected Improvement Criterion (EIC). The performance comparison is based on the scenarios of optimization under uncertainty aka Robust Optimization, where the focus
is to find solutions immune to the uncertainty in the decision/search variables. In the comparison, EIC serves as a baseline.

# Introduction
This code is based on our paper, titled [A New Acquisition Function for Robust Bayesian
Optimization of Unconstrained Problems](https://dl.acm.org/doi/pdf/10.1145/3449726.3463206) (Ullah, Wang, Menzel, Sendhoff & Bäck, 2021), and can be used to reproduce
the experimental setup and results. The code is produced in Python 3.7.0. The main packages utilized in this code are presented in the next section which deals with technical requirements. 

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

| Package | Description |
| --- | --- |
| pyDOE2 | For sampling plans and Design of Experiment (DoE)  |
| SciPy |For numerical optimization based on L-BFGS-B algorithm |
| SMT |Surrogate Modeling Toolbox utilized for the implementation of Branin Function |
| scikit-learn | For constructing the Kriging surrogate, as well as data manipulation |



For this project to run you need:
* Python >= 3.5
* Numpy 1.20.0
* pyDOE2 1.3.0
* Scipy 1.6.1
* SMT 1.0.0
* Scikit-learn 0.24.0 

## References:
<a id="1">[1]</a> 
Ullah, Sibghat, et al. "A New Acquisition Function for Robust Bayesian Optimization of Unconstrained Problems." (2021).
