#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <memory>

#include "../derivative/derivative_smart.hpp"

// ----------------------------------------------------------------------------------------
/*
For more informations, please have a look at the main paper related to this project.
It helps to understand the meaning of these functions, especially for the differentiation.
*/
// ----------------------------------------------------------------------------------------

/**
 * @brief Check that a matrix is PSD
 * 
 * @param array the matrix
 * @return true if the matrix is PSD
 * @return false if the matrix is not PSD
 */
bool isPsd(kernel_array &array);

bool isPositiveHalfInteger(double x);

/**
 * @brief Implementation of the Gamma function
 * 
 * @param num it's parameter
 * @return double : the Gamma function evaluated at "num"
 */
double gamma_fct(double num);

/**
 * @brief Implementation of the Error function
 * 
 * @param num it's parameter
 * @return double : the Error function evaluated at "num"
 */
double erf_fct(double num);


/**
 * @brief Implementation of the Bessel function
 * 
 * @param nu parameter 1
 * @param x parameter 2
 * @return double : the Bessel function evaluated at "x" with parameter "nu"
 */
double modif_bessel_fct(double nu, double x);

/**
 * @brief Exponential coefficient for matern kernel
*/
double matern_conv_time_coeff(const int p, const double t0,
                              const double l_t, const double t);

/**
 * @brief Compute the C coefficients of the matern kernel
 */
double C_k1_k2(const int k1, const int k2, const int p, const double l_t);

/**
 * @brief This function computes the derivatives of the P_polynome for matern kernel
 * @param results an array with all derivatives with respect to time of the P_polynome.
 */
void P_function(const int k1, const int k2, const int p, const double l_t,
                kernel_array &time_diff, kernel_array &param_polynome,
                const int obs, kernel_array &results);

/**
 * @brief Returns the abstraction of the polynome we want to differentiate for the matern kernel
 * @return std::shared_ptr<Derivative> 
 */
std::shared_ptr<Derivative> P_polynome(const int k1, const int k2, const int p,
                                       kernel_array &param, const int obs);

#endif