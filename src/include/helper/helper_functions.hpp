#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <memory>

#include "../derivative/derivative_smart.hpp"

bool isPsd(kernel_array &array);

bool isPositiveHalfInteger(double x);

double gamma_fct(double num);

double erf_fct(double num);

double modif_bessel_fct(double nu, double x);

double matern_conv_time_coeff(const int p, const double t0,
                              const double l_t, const double t);

double C_k1_k2(const int k1, const int k2, const int p, const double l_t);

void P_function(const int k1, const int k2, const int p, const double l_t,
                kernel_array &time_diff, kernel_array &param_polynome,
                const int obs, kernel_array &results);

std::shared_ptr<Derivative> P_polynome(const int k1, const int k2, const int p,
                                       kernel_array &param, const int obs);

#endif