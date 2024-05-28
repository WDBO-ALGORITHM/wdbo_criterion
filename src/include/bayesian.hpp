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

kernel_vec wasserstein_critetion_point(kernel_array &kernel_D, const double lambda,
                                       const double variance, Eigen::Ref<kernel_vec> &y_vec, kernel_array &coeff_c_plus,
                                       const int num_obs, const int normalize_criterion);

#endif