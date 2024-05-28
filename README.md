# Dynamic Bayesian Optimization Criterion for Data Staleness

This repository contains the implementation of the criterion used by W-DBO, an fast algorithm for Dynamic Bayesian Optimization (DBO).
This criterion aims to decrease the time needed to evaluate stale data points of a dataset to optimize continuously a function that is changing
over time.

This package is used for the full W-DBO algorithm (wdbo name?).

## Content of the Project

Data staleness is a major issue in DBO. To maximize the number of queries on the function that we want to optimize, the number of observations used by the Gaussian Process (GP) to do so needs to be limited to avoid the inference time of the GP to become too large (because it is of the order cube of the number of observations). This criterion helps to identify data points that no longer bring useful informations for the future queries of the GP to optimize our function over time.

Other methods to identify stale data exist but are not on the same range of complexity as W-DBO. The criterion has been written in C++ using the linear algebra library Eigen. It is available for python programmer thanks to pybind11 as bindings package.

Thanks to this criterion, W-DBO outperforms the other algorithms for DBO.

## How to use the package

The package is unavailable on macos platforms. This is related to the fact that neither Apple clang nor clang++'s own stdlib have the Bessel functions implemented, which are essential for the criterion if the Matérn kernel is used. We recommend to use another platform using docker for example.

On windows and Linux it's simple: the package is a simple function binded using pybind11. To use the criterion you just need to download it using `pip`

```bash
pip install wdbo_criterion
```

2 Kernels can be used: RBF and Matérn. You can then use the criterion by calling the function as follows:

```python
import wdbo_criterion
import numpy as np

dimension = 5 # dimension of each point
size_dataset = 4 # number of observations
lamb = 0.72971242 # lambda of kernels
variance = 0.3 # variance of the sample noise
l_s = 0.2908291 # Space Lengthscale
l_t = 0.18401412 # Time Lengthscale
t0 = 0.3383822445869446 # current time
nu_t = 1.5

normalize_criterion = 1
verbose = 0 # 1 if you want to print the values passed

inputs = np.array([
 [0.9687505, 0.33303383, 0.91202209, 0.05732997, 0.4390582],
 [0.80951052, 0.80318733, 0.22482374, 0.37230322, 0.98726084],
 [0.86916179, 0.46339954, 0.21277571, 0.07380144, 0.56478391],
 [0.40247319, 0.9790963, 0.98341625, 0.54941297, 0.21665171]])

time = np.array([0.04599568, 0.09191627, 0.09606597, 0.06932814])
y = np.array([-1.39654819, -1.27302667, -1.39127814,  2.10939348])

kernel_space = wdbo_criterion.RBFKernel(l_s)
kernel_time = wdbo_criterion.MaternKernel(l_t, nu_t)

result = wdbo_criterion.wasserstein_criterion(
    inputs, y, time, size_dataset, dimension, lamb, variance, kernel_space, kernel_time, t0, verbose, normalize_criterion)

```

The package assumes that `numpy` arrays and matrices are used to provide the dataset because pybind11 that provides the bindings is optimized for this specific translation. Eigen requires contiguous arrays so maybe you'll need to use the function `np.ascontiguousarray`.

The variable `result` will be an numpy array where each element $0 \leq i < n$ is the criterion associated with the element `inputs[i]`, `y[i]` and `time[i]`.

## Troubleshootings

If, by installing the library using pip, you get an `libwdbo_bayesian.so is not found`, run the command

```
export LD_LIBRARY_PATH=/home/<user>/miniconda3/envs/pybind/lib/<python_version>/site-packages/:$LD_LIBRARY_PATH
```

## Context

This project was my bachelor project (2024) at EPFL. It has been supervised by Anthony Bardou from the Information and Network Dynamics Lab (INDY) at EPFL, directed by Prof. Patrick Thiran and Matthias Grossglauser.

The project taught me a lot in the field of Bayesian Optimization (BO) and DBO. Many aspects of computer science in engineering has been involved (memory management, auto-differentiation, numerical methods) and I enjoyed working on the state of the art of DBO by implementing them and seeing the improvements made by this W-DBO algorithm and it's criterion.

## License

wdbo_criterion is provided under a BSD-style license that can be found in the LICENSE file. By using, distributing, or contributing to this project, you agree to the terms and conditions of this license.

## Links

- Pypi repository of wdbo_criterion: https://pypi.org/project/wdbo-criterion/
- Pypi repository of (name full algo) : TODO
- ref papers?
