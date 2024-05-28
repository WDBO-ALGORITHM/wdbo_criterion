#include <iostream>
#include <cmath>

#include "../../include/helper/helper_functions.hpp"
#include "../../include/bayesian_types.hpp"
#include "../../include/kernel/kernel_functions.hpp"

/* for convenience */
#define M_PI 3.14159265358979323846

// --------------------------------------------------------------------------------------------------
// EXPONENTIAL KERNEL
// --------------------------------------------------------------------------------------------------

/*
Space Square exponential kernel
*/
double square_exp_kernel_space(const double norm, const double l_s)
{
    return std::exp(-norm / (2.0 * l_s * l_s));
}

/*
Time Square exponential kernel
*/
double square_exp_kernel_time(const double t, const double l_t)
{
    return std::exp(-std::pow(t, 2) / (2.0 * l_t * l_t));
}

/*
(k_S \conv k_S) : convolution of 2 space square exponential kernels
*/
double square_exp_conv_space_kernel(const double norm_squared, const double d, const double l_s)
{
    double first = std::pow(M_PI, d / 2.0) * std::pow(l_s, d);
    double exp = std::exp(-norm_squared / (4.0 * l_s * l_s));

    return std::min(1.0, first * exp);
}

/*
(k_T \conv k_T) : convolution of 2 time square exponential kernels

The function take advantage of vectorization. Since the kernel is a Gram matrix, we apply the
the function for each element providing all elements are ones.
*/
void square_exp_conv_time_kernel(const double t0, kernel_array &ti_minus_tj,
                                 kernel_array &ti_plus_tj, const double l_t,
                                 kernel_array &result)
{
    double a = std::sqrt(M_PI) * l_t / 2.0;

    ti_minus_tj = ti_minus_tj.unaryExpr([&l_t](double t)
                                        { return std::exp(-(std::pow(t, 2.0) / (2.0 * l_t * l_t))); });

    ti_plus_tj = ti_plus_tj.unaryExpr([&l_t, &t0](double t)
                                      { return 1 - erf_fct((2.0 * t0 - t) / (2.0 * l_t)); });

    result = a * ti_minus_tj.cwiseProduct(ti_plus_tj);
}

// -----------------------------------------------------------------------------------------------
// Matérn
// -----------------------------------------------------------------------------------------------

/*
Space Matern kernel
*/
double matern_kernel_space(double norm_square, const double nu, const double l_s)
{
    if (norm_square < 0)
    {
        norm_square = 0.0;
    }
    double norm_p = std::sqrt(norm_square);
    if (norm_p <= 1e-5)
    {
        norm_p = 1e-5;
    }

    double squ = std::sqrt(2.0 * nu);
    double x = (norm_p * squ) / l_s;
    double bessel = modif_bessel_fct(nu, x);
    double frac = (std::pow(2.0, 1.0 - nu)) / gamma_fct(nu);
    return frac * std::pow(x, nu) * bessel;
}

/*
Time Matern kernel
*/
double matern_kernel_time(double t, const double nu, const double l_t)
{
    if (t <= 1e-5)
    {
        t = 1e-5;
    }

    double squ = std::sqrt(2.0 * nu);
    double x = (t * squ) / l_t;
    double bessel = modif_bessel_fct(nu, x);
    double frac = (std::pow(2.0, 1.0 - nu)) / gamma_fct(nu);
    return frac * std::pow(x, nu) * bessel;
}

/*
(k_S \conv k_S) : convolution of 2 space matern kernels
*/
double matern_conv_space_kernel(double norm_squared, const double d, const double nu, const double l_s)
{
    if (norm_squared < 0)
    {
        norm_squared = 0;
    }
    double a = (std::sqrt(2.0 * nu)) / l_s;
    double norm = std::sqrt(norm_squared);

    if (norm <= 1e-5)
    {
        norm = 1e-5;
    }

    double x = norm * a;
    double bessel = modif_bessel_fct((2.0 * nu) + (d / 2.0), x);

    double norm_pow = std::pow(norm, (2.0 * nu) + (d / 2.0));

    double b = std::pow(a, (2.0 * nu) - (d / 2.0));

    double gamma_nu = std::pow(gamma_fct(nu), 2.0);
    double gamma_2 = gamma_fct(2.0 * nu + d);
    double gamma_3 = std::pow(gamma_fct(nu + (d / 2.0)), 2.0);
    double c = std::pow(2.0, (d / 2.0) - (2.0 * nu) + 1) * std::pow(M_PI, d / 2.0);

    double frac = (c * gamma_3) / (gamma_2 * gamma_nu);

    return std::min(1.0, frac * b * norm_pow * bessel);
}

/*
(k_T \conv k_T) : convolution of 2 time matern kernels

The function take advantage of vectorization. Since the kernel is a Gram matrix, we apply the
the function for each element providing all elements are ones.
*/
void matern_conv_time_kernel(const double t0, const double nu_t, const double l_t,
                             const int obs, kernel_array &ti_minus_tj,
                             kernel_array &ti_plus_tj, kernel_array &param_polynome,
                             kernel_array &results_derivatives, kernel_array &results)
{
    // already checked that nu_t is a positive half integer
    int p = nu_t - 0.5;

    // exponential coefficients
    ti_plus_tj = ti_plus_tj.unaryExpr([&p, &t0, &l_t](double t)
                                      { return matern_conv_time_coeff(p, t0, l_t, t); });
    for (int k1 = 0; k1 <= p; ++k1)
    {
        for (int k2 = 0; k2 <= p; ++k2)
        {
            results_derivatives = results_derivatives.unaryExpr([](double x)
                                                                { return 0.0; });
            P_function(k1, k2, p, l_t, ti_minus_tj, param_polynome, obs, results_derivatives);
            results += (C_k1_k2(k1, k2, p, l_t) * results_derivatives.cwiseProduct(ti_plus_tj));
        }
    }
}