#ifndef KERNEL_FUNCTIONS_H
#define KERNEL_FUNCTIONS_H

#include <cmath>

#include "../bayesian_types.hpp"
#include "../helper/helper_functions.hpp"

// --------------------------------------------------------------------------------------------------
// EXPONENTIAL KERNEL
// Please have a look at the main paper (readme) for more informations.
// --------------------------------------------------------------------------------------------------

/**
 * @brief Square exponential kernel for space dimensions
 * 
 * @param norm the norm of the data vector
 * @param l_s space lengthscale
 * @return double
 */
double square_exp_kernel_space(const double norm, const double l_s);

/**
 * @brief Square exponential kernel for time dimension
 * 
 * @param norm the time
 * @param l_s time lengthscale
 * @return double
 */
double square_exp_kernel_time(const double t, const double l_t);

/**
 * @brief Formula of the convolution of the Square exponential kernel with itself for space dimensions
 * 
 * @param norm the norm squared of the data vector
 * @param l_s space lengthscale
 * @param d the dimension of the data
 * @return double
 */
double square_exp_conv_space_kernel(const double norm_squared, const double d, const double l_s);

/**
 * @brief ARD kernel for space dimensions.
 * 
 * @param xi_minus_xj vector representing the difference between 2 data vectors
 * @param d the dimension of the data
 * @param sigma the kernel matrix
 * @return double 
 */
double ard_space_kernel(kernel_vec &xi_minus_xj, const int d, kernel_array &sigma);

/**
 * @brief ARD kernel convoluted with itself for space dimensions.
 * 
 * @param xi_minus_xj vector representing the difference between 2 data vectors
 * @param d the dimension of the data
 * @param sigma the kernel matrix
 * @param determinant the determinant of @sigma
 * @return double 
 */
double ard_space_conv_kernel(kernel_vec &xi_minus_xj, const int d, kernel_array &sigma, double determinant);

/**
 * @brief  Formula of the convolution of the Square exponential kernel with itself for time dimension
 * 
 * @param t0 the current time
 * @param ti_minus_tj a matrix with the difference between each pair of times t1 t2
 * @param ti_plus_tj a matrix with the sum between each pair of times t1 t2
 * @param l_t the time lengthscale
 * @param result matrix with all the evaluations of the function for each pair of times
 */
void square_exp_conv_time_kernel(const double t0, kernel_array &ti_minus_tj,
                                 kernel_array &ti_plus_tj, const double l_t,
                                 kernel_array &result);
// -----------------------------------------------------------------------------------------------
// Matérn
// -----------------------------------------------------------------------------------------------

/**
 * @brief Matern kernel for space dimensions
 * 
 * @param norm_square the norm squared of a data vector
 * @param nu parameter of kernel
 * @param l_s spatial lengthscale
 * @return double 
 */
double matern_kernel_space(double norm_square, const double nu, const double l_s);

/**
 * @brief Matern kernel for time dimension
 * 
 * @param t time
 * @param nu parameter of kernel
 * @param l_t time lengthscale
 * @return double 
 */
double matern_kernel_time(double t, const double nu, const double l_t);

/**
 * @brief Matern kernel convoluted with itself for the space dimensions
 * 
 * @param norm_squared the norm squared of a data vector
 * @param d the number of spatial dimensions
 * @param nu parameter of the kernel
 * @param l_s spatial lengthscales
 * @return double 
 */
double matern_conv_space_kernel(double norm_squared, const double d, const double nu, const double l_s);

/**
 * @brief Matern kernel convoluted with itself for the time dimensions
 * 
 * @param t0 the current time
 * @param nu_t parameter of the kernel 
 * @param l_t time lengthscale
 * @param obs the number of data vectors
 * @param ti_minus_tj  a matrix with the difference between each pair of times t1 t2
 * @param ti_plus_tj  a matrix with the sums between each pair of times t1 t2
 * @param param_polynome 
 * @param results_derivatives result of P_function
 * @param results 
 */
void matern_conv_time_kernel(const double t0, const double nu_t, const double l_t,
                             const int obs, kernel_array &ti_minus_tj,
                             kernel_array &ti_plus_tj, kernel_array &param_polynome,
                             kernel_array &results_derivatives, kernel_array &results);

#endif