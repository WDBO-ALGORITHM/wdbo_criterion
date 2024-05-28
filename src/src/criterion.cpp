#include <iostream>

#include "../include/criterion.hpp"
#include "../include/helper/helper_functions.hpp"
#include "../include/errors.hpp"

// ----------------------------------------------------------------------------
/*
Please refer to the pdf report for details on the linear algebra computations
*/
// ----------------------------------------------------------------------------

double E_val(const double lambda,
             const double variance,
             kernel_row_vec &vec_row,
             kernel_vec &vec)
{
    double test = lambda + variance - (vec_row * vec);
    return std::pow(test, -1);
}

void G_val(const double e_value, kernel_row_vec &vec_row,
           kernel_row_vec &result)
{
    result = -e_value * vec_row;
}

void H_val(const double lambda, const double variance, kernel_vec &vec,
           kernel_array &A_value, kernel_vec &result)
{
    const double pref = -(1.0 / (variance + lambda));
    result = pref * (A_value.householderQr().solve(vec));
}

double a_val(const double e_value, kernel_vec &ys_without_target, const double y_1,
             kernel_row_vec &g_value)
{
    return e_value * y_1 + g_value * ys_without_target;
}

int F_val(kernel_array &A_value, kernel_array &result)
{
    Eigen::FullPivLU<kernel_array> lu_decomp(A_value);

    if (lu_decomp.isInvertible())
    {
        result = lu_decomp.inverse();
        return ERR_NONE;
    }
    else
    {
        return ERR_MATRIX_NOT_INVERTIBLE;
    }
}

int inverse_matrix_cholsky(kernel_array &delta, kernel_array &result)
{
    if (isPsd(delta))
    {
        Eigen::LLT<kernel_array> lltOfM(delta);
        kernel_array L = lltOfM.matrixL();

        kernel_array L_inverse = L.inverse();
        result = L_inverse.transpose() * L_inverse;
        return ERR_NONE;
    }
    else
    {
        return ERR_MATRIX_NOT_PSD;
    }
}

void M_val(kernel_array &F_value,
           kernel_array &delta_inverse_tilde,
           kernel_array &M_value)
{
    M_value = F_value - delta_inverse_tilde;
}

void b_val(kernel_vec &h_value, const double y_1, kernel_vec &ys_without_target,
           kernel_array &m_value,
           kernel_row_vec &result)
{
    result = h_value.transpose() * y_1 + ys_without_target.transpose() * m_value;
}

void c_val(kernel_vec &h_value, kernel_row_vec &g_value, kernel_row_vec &result)
{
    result = g_value + h_value.transpose();
}

int normalization_criterion(const double lambda, kernel_vec &a_vec, kernel_array &c_plus_coeff, kernel_array &inverse_delta,
                            const double num_obs, double &result)
{
    const double first = a_vec.transpose() * c_plus_coeff * a_vec;
    const double second = kernel_vec::Ones(num_obs).transpose() * (inverse_delta.cwiseProduct(c_plus_coeff)) * kernel_vec::Ones(num_obs);

    if (std::sqrt(first + second) < 0)
    {
        return ERR_NEGATIVE_SQRT;
    }
    result = lambda * std::sqrt(first + second);
    return ERR_NONE;
}