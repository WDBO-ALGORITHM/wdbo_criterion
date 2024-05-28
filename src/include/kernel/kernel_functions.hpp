#ifndef KERNEL_FUNCTIONS_H
#define KERNEL_FUNCTIONS_H

#include <cmath>

#include "../bayesian_types.hpp"
#include "../helper/helper_functions.hpp"

// --------------------------------------------------------------------------------------------------
// EXPONENTIAL KERNEL
// --------------------------------------------------------------------------------------------------

double square_exp_kernel_space(const double norm, const double l_s);

double square_exp_kernel_time(const double t, const double l_t);

double square_exp_conv_space_kernel(const double norm_squared, const double d, const double l_s);

void square_exp_conv_time_kernel(const double t0, kernel_array &ti_minus_tj,
                                 kernel_array &ti_plus_tj, const double l_t,
                                 kernel_array &result);
// -----------------------------------------------------------------------------------------------
// Matérn
// -----------------------------------------------------------------------------------------------

double matern_kernel_space(double norm_square, const double nu, const double l_s);

double matern_kernel_time(double t, const double nu, const double l_t);

double matern_conv_space_kernel(double norm_squared, const double d, const double nu, const double l_s);

void matern_conv_time_kernel(const double t0, const double nu_t, const double l_t,
                             const int obs, kernel_array &ti_minus_tj,
                             kernel_array &ti_plus_tj, kernel_array &param_polynome,
                             kernel_array &results_derivatives, kernel_array &results);

#endif