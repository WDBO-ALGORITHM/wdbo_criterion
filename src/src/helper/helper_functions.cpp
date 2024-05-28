#include <iostream>
#include <cmath>
#include <vector>
#include <memory>

#include "../../include/derivative/derivative_smart.hpp"
#include "../../include/bayesian_types.hpp"

using std::vector;

bool isPsd(kernel_array &array)
{
    if (!array.isApprox(array.transpose()))
    {
        return false;
    }
    const auto ldlt = array.template selfadjointView<Eigen::Upper>().ldlt();
    if (ldlt.info() == Eigen::NumericalIssue || !ldlt.isPositive())
    {
        return false;
    }
    return true;
}

bool isPositiveHalfInteger(double x)
{
    if (x <= 0)
        return false;

    double fractional_part = x - std::floor(x);
    return std::abs(fractional_part - 0.5) < 1e-10;
}

double gamma_fct(double num)
{
    return std::tgamma(num);
}

double erf_fct(double num)
{
    return std::erf(num);
}

double modif_bessel_fct(double nu, double x)
{
    return std::cyl_bessel_k(nu, x);
}

double matern_conv_time_coeff(const int p, const double t0,
                              const double l_t, const double t)
{
    return std::exp(-(std::sqrt(2.0 * p + 1.0) * (2.0 * t0 - t)) / l_t);
}

/*
Compute the C coefficients of matern kernel
*/
double C_k1_k2(const int k1, const int k2, const int p, const double l_t)
{
    // since k1 and k2 are [0, p], then p - k1 or p - k2 >= 0
    double p_test_up, p_test_down, p_k1, p_k2, p_min_k1, p_min_k2, k1_test, k2_test = 0.0;

    const double last_term = std::pow((2 * std::sqrt(2 * p + 1)) / (l_t), 2 * p - k1 - k2 - 1);

    p_test_up = (p == 0) ? 1 : gamma_fct((double)(p + 1.0));
    p_test_down = (p == 0) ? 1 : gamma_fct((double)(2 * p + 1.0));
    p_k1 = (p + k1 == 0) ? 1 : gamma_fct((double)(p + k1 + 1.0));
    p_k2 = (p + k2 == 0) ? 1 : gamma_fct((double)(p + k2 + 1.0));
    p_min_k1 = (p - k1 == 0) ? 1 : gamma_fct((double)(p - k1 + 1.0));
    p_min_k2 = (p - k2 == 0) ? 1 : gamma_fct((double)(p - k2 + 1.0));
    k1_test = (k1 == 0) ? 1 : gamma_fct((double)(k1 + 1.0));
    k2_test = (k2 == 0) ? 1 : gamma_fct((double)(k2 + 1.0));

    const double first_term = std::pow((p_test_up) / (p_test_down), 2.0);
    const double middle_term = (p_k1 * p_k2) / (k1_test * k2_test * p_min_k1 * p_min_k2);

    return first_term * middle_term * last_term;
}

/*
Returns the polynomial to differentiate for matern kernel
*/
std::shared_ptr<Derivative> P_polynome(const int k1, const int k2, const int p,
                                       kernel_array &param, const int obs)
{
    kernel_array b = kernel_array::Zero(obs, obs);
    return std::shared_ptr<Derivative>(new Product(std::shared_ptr<Derivative>(new ExpMonome(param, p - k2)),
                                                   std::shared_ptr<Derivative>(new ExpMonome(b, p - k1))));
}

/*
Compute the derivatives of the P_polynome for matern kernel
*/
void P_function(const int k1, const int k2, const int p, const double l_t,
                kernel_array &time_diff, kernel_array &param_polynome,
                const int obs, kernel_array &results)
{
    double power_param = 0.0;
    std::shared_ptr<Derivative> derivation_polynome = P_polynome(k1, k2, p, time_diff, obs);
    kernel_array derivative_value = kernel_array::Zero(obs, obs);
    vector<std::shared_ptr<Derivative>> grads_vec(2 * p - k1 - k2 + 1);
    for (int i = 0; i <= 2 * p - k1 - k2; ++i)
    {
        if (i == 0)
        {
            grads_vec.at(i) = derivation_polynome;
        }
        else
        {
            grads_vec.at(i) = (grads_vec.at(i - 1)->derivative(obs));
        }
    }

    for (int k3 = 0; k3 <= 2 * p - k1 - k2; ++k3)
    {
        power_param = std::pow(l_t / (2.0 * std::sqrt(2.0 * p + 1)), k3);
        derivative_value = grads_vec.at(k3)->evaluate(param_polynome);
        results += (power_param * derivative_value);
    }
}