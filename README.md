[![DOI](https://zenodo.org/badge/404800705.svg)](https://zenodo.org/badge/latestdoi/404800705)



# Comparison of Moment-Generating Function of Improvement and Expected Improvement Criterion for Robust Bayesian Optimization
Contains code for the empirical comparison of two acquisition functions, namely the so-called Moment-Generating Function of the Improvement (MGFI) and
the Expected Improvement Criterion (EIC). The performance comparison is based on the scenarios of optimization under uncertainty aka Robust Optimization, where the focus
is to find solutions immune to the uncertainty in the decision/search variables. In the comparison, EIC serves as a baseline.

# Introduction
This code is based on our paper [A New Acquisition Function for Robust Bayesian
Optimization of Unconstrained Problems](https://dl.acm.org/doi/pdf/10.1145/3449726.3463206) (Ullah, Wang, Menzel, Sendhoff & Bäck, 2021) and can be used to reproduce
the experimental setup and results.


## Requiremnts
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
