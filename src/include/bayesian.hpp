#ifndef BAYESIAN_H
#define BAYESIAN_H

#include <vector>
#include <array>

#include "bayesian_types.hpp"
#include "kernel/kernel.hpp"
#include "kernel/kernel_functions.hpp"
#include "criterion.hpp"
#include "kernel/kernel_types.hpp"

using Eigen::VectorXd;
using std::vector;

/**
 * @brief Main function. It's the function called using pybind11 from Python code.
 * 
 * @param dataset_x the dataset of points in space
 * @param y_vector the vector of function values 
 * @param time_vec the vector of time
 * @param num_obs the number of observations (number of data points)
 * @param dim the dimension of the data points
 * @param lambda lambda multiplying the spatial and time kernel
 * @param variance the noise
 * @param kernel_space the type of spatial kernel
 * @param kernel_time the type of the time kernel
 * @param t0 the current time
 * @param verbose 1 => print the values passed as params, 0 => no print
 * @param normalize_criterion 1 => normalized criterion, 0 => no normalization
 * @return kernel_vec 
 */
kernel_vec wasserstein_criterion(Eigen::Ref<compatible_storage_order_matrix> dataset_x, Eigen::Ref<kernel_vec> y_vector,
                                 Eigen::Ref<kernel_vec> time_vec,
                                 const int num_obs,
                                 const int dim,
                                 const double lambda,
                                 double variance,
                                 KernelParams &kernel_space,
                                 KernelParams &kernel_time,
                                 const double t0,
                                 const int verbose,
                                 const int normalize_criterion);

// compute the criterion with respect to one data point
kernel_vec wasserstein_critetion_point(kernel_array &kernel_D, const double lambda,
                                       const double variance, Eigen::Ref<kernel_vec> &y_vec, kernel_array &coeff_c_plus,
                                       const int num_obs, const int normalize_criterion);

#endif