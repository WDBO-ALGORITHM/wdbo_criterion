# Dynamic Bayesian Optimization Criterion for Data Staleness

This repository contains the C++ implementation of the criterion used in W-DBO, a Dynamic Bayesian Optimization algorithm. It is a well-suited optimizer for a dynamic black-box function. This criterion aims to decrease the time needed to evaluate stale data points in a dataset to optimize continuously a function that is changing over time.

## Content and Context

Some algorithms have been proposed to extend Bayesian Optimization (BO) to time-varying functions. This adaptation is known as Dynamic Bayesian Optimization (DBO). To achieve great performance, W-DBO from [1] uses a criterion that quantifies how relevant an observation is for the future predictions of the Gaussian Process. By evolving in time, the optimum of the function changes, and observations of the function become less relevant due to their lack of information on its future values. In a long-period time optimization, keeping all of them will impact the performance of the algorithm. In fact, the sampling frequency will decrease due to the growth of the Gaussian Process' inference time. To remove them rapidly, the criterion should be calculated efficiently in a low-level programming language. By speeding up the computations using C++ to calculate the criterion, W-DBO shown great improvements over state of the art solutions. As stated before, this repository contains (i) the C++ implementation of this criterion. The criterion is computed using the linear algebra library Eigen. It is available for python programmer thanks to Pybind11

This project has been the bachelor project (2024) of [Giovanni Ranieri](https://flxinxout.github.io) at EPFL. It has been supervised by [Anthony Bardou](https://abardou.github.io/) from the Information and Network Dynamics Lab (INDY) at EPFL, directed by Prof. Patrick Thiran and Matthias Grossglauser.

The project taught me a lot in the field of Bayesian Optimization (BO) and DBO. Many aspects of computer science in engineering have been involved (memory management, auto-differentiation, numerical methods) and I enjoyed working on the state of the art of DBO by implementing them and seeing the improvements made by W-DBO with the criterion.

## W-DBO and Note To Developers

The criterion is used in W-DBO, available as a Python package on this [link](https://github.com/WDBO-ALGORITHM/wdbo_algo). To use W-DBO, the developer do not need to call the criterion with parameters (everything is done behind the scene during the optimization, removing stale observations are simply done by some call of a function). For interested readers, [1] servs as mathematical reference for the overall algorithm W-DBO.

## How to use the package

The package is unavailable on MacOS platforms. Neither Apple clang nor clang++'s own stdlib have the Bessel functions implemented, which are essential for the criterion. We recommend to use another platform using Docker or a VM.

On windows and Linux it's simple: the package is a simple function binded using Pybind11. You just need to download it using `pip`

```bash
pip install wdbo-criterion
```

The criterion needs 2 kernels: 1 for the spatial dimensions and 1 for the time dimension. 2 Kernels can be used: RBF (Squared exponential) and Matérn. The `lamb` has to be specified (> 0), the sample noise too (if you provide less than 1e-5, it will be set to 1e-5). Normalized criterion is set to 1 to be consistent with the results of [1].

Here is an example:

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
verbose = 0 # 1 for debug mode

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

The package assumes that `numpy` arrays and matrices are used to provide the dataset because Pybind11 that provides the bindings is optimized for this specific translation. Eigen requires contiguous arrays so maybe you'll need to use the function `np.ascontiguousarray`.

The variable `result` will be an numpy array where each element $0 \leq i < n$ is the criterion associated with the element `inputs[i]`, `y[i]` and `time[i]`. To use an ARD Kernel:

```python
import wdbo_criterion
import numpy as np

dimension = 5 # dimension of each point
size_dataset = 4 # number of observations
lamb = 0.72971242 # lambda of kernels
variance = 0.3 # variance of the sample noise
l_ss = np.array([0.2, 0.3, 1, 0.1, 0.2]) # Space Lengthscales
l_t = 0.18401412 # Time Lengthscale
t0 = 0.3383822445869446 # current time
nu_t = 1.5

normalize_criterion = 1
verbose = 0 # 1 for debug mode

inputs = np.array([
 [0.9687505, 0.33303383, 0.91202209, 0.05732997, 0.4390582],
 [0.80951052, 0.80318733, 0.22482374, 0.37230322, 0.98726084],
 [0.86916179, 0.46339954, 0.21277571, 0.07380144, 0.56478391],
 [0.40247319, 0.9790963, 0.98341625, 0.54941297, 0.21665171]])

time = np.array([0.04599568, 0.09191627, 0.09606597, 0.06932814])
y = np.array([-1.39654819, -1.27302667, -1.39127814,  2.10939348])

kernel_space = wdbo_criterion.ARDKernel(l_ss)
kernel_time = wdbo_criterion.MaternKernel(l_t, nu_t)

result = wdbo_criterion.wasserstein_criterion(
    inputs, y, time, size_dataset, dimension, lamb, variance, kernel_space, kernel_time, t0, verbose, normalize_criterion)

```

## Troubleshootings

If, by installing the library using pip, you get a `libwdbo_bayesian.so is not found`, run the command

```
export LD_LIBRARY_PATH=<path to your python packages installed>:$LD_LIBRARY_PATH
```

## License

wdbo-criterion is provided under a BSD-style license that can be found in the LICENSE file. By using, distributing, or contributing to this project, you agree to the terms and conditions of this license.

## Links

- Pypi repository of the criterion: https://pypi.org/project/wdbo-criterion/
- Pypi repository of W-DBO : https://pypi.org/project/wdbo-algo/

## References

[1] Bardou, A., Thiran, P., & Ranieri, G. (2024). This Too Shall Pass: Removing Stale Observations in Dynamic Bayesian Optimization. arXiv preprint arXiv:2405.14540.
