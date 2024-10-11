#include <iostream>
#include <exception>
#include <tuple>

#include "../../include/kernel/kernel.hpp"
#include "../../include/kernel/kernel_functions.hpp"
#include "../../include/errors.hpp"
#include "../../include/helper/helper_functions.hpp"
#include "../../include/kernel/kernel_types.hpp"

void kernel_column_vector_and_conv_vector(kernel_vec &ker_vec, kernel_vec &conv_vec, const int index, kernel_array &kernel,
                                          kernel_array &conv_kernel,
                                          const int num_obs)
{
    kernel_vec tmp_1 = kernel.col(index);
    ker_vec << tmp_1.head(index),
        tmp_1.tail(num_obs - index - 1);

    kernel_vec tmp_2 = conv_kernel.col(index);
    conv_vec << tmp_2.head(index),
        tmp_2.tail(num_obs - index - 1);
}

void create_kernel_and_coeff_C_Plus(kernel_array &k, Eigen::Ref<compatible_storage_order_matrix> &dataset_x,
                                    Eigen::Ref<kernel_vec> &time_vec,
                                    KernelParams &kernel_space, KernelParams &kernel_time,
                                    kernel_array &coeff_c_plus, const int d, const double t0, const int num_obs,
                                    const double lambda)
{

    if (dynamic_cast<const RBFKernel *>(&kernel_space) != nullptr && dynamic_cast<const RBFKernel *>(&kernel_time) != nullptr)
    {
        // SPACE AND TIME EXP
        const RBFKernel *rbf_space = dynamic_cast<const RBFKernel *>(&kernel_space);
        const RBFKernel *rbf_time = dynamic_cast<const RBFKernel *>(&kernel_time);

        double l_s, junk;
        std::tie(l_s, junk) = rbf_space->params();

        double l_t, junk_;
        std::tie(l_t, junk_) = rbf_time->params();

        kernel_array temp(num_obs, num_obs);
        kernel_array temp_2(num_obs, num_obs);
        kernel_array kernel_time(num_obs, num_obs);
        kernel_array kernel_conv_time(num_obs, num_obs);

        // -------------------------------
        // kernel time
        temp = time_vec.replicate(1, num_obs) - time_vec.transpose().replicate(num_obs, 1);
        kernel_time = temp.unaryExpr([&l_t](double t)
                                     { return square_exp_kernel_time(t, l_t); });
        // -------------------------------

        // -------------------------------
        // kernel conv time
        temp_2 = time_vec.replicate(1, num_obs) + time_vec.transpose().replicate(num_obs, 1);
        square_exp_conv_time_kernel(t0, temp, temp_2, l_t, kernel_conv_time);
        // -------------------------------

        // -------------------------------
        // kernel space
        kernel_row_vec square_norms = dataset_x.rowwise().squaredNorm();
        temp_2 = square_norms.replicate(num_obs, 1) + square_norms.transpose().replicate(1, num_obs);
        temp = temp_2 - (2.0 * dataset_x * dataset_x.transpose());

        temp_2 = temp.unaryExpr([&l_s](double norm_squared)
                                { return square_exp_kernel_space(norm_squared, l_s); });
        // -------------------------------

        k = lambda * temp_2.cwiseProduct(kernel_time);

        // -------------------------------
        // kernel conv space
        temp_2 = temp.unaryExpr([&l_s, &d](double norm_squared)
                                { return square_exp_conv_space_kernel(norm_squared, d, l_s); });
        // -------------------------------

        coeff_c_plus = temp_2.cwiseProduct(kernel_conv_time);
    }
    else if (dynamic_cast<const MaternKernel *>(&kernel_space) != nullptr && dynamic_cast<const RBFKernel *>(&kernel_time) != nullptr)
    {
        // SPACE MATERN AND TIME EXP
        const MaternKernel *matern_space = dynamic_cast<const MaternKernel *>(&kernel_space);
        const RBFKernel *rbf_time = dynamic_cast<const RBFKernel *>(&kernel_time);

        double l_s, nu_s;
        std::tie(l_s, nu_s) = matern_space->params();

        double l_t, junk_;
        std::tie(l_t, junk_) = rbf_time->params();

        kernel_array temp(num_obs, num_obs);
        kernel_array temp_2(num_obs, num_obs);
        kernel_array kernel_time(num_obs, num_obs);
        kernel_array kernel_conv_time(num_obs, num_obs);

        // -------------------------------
        // kernel time
        temp = time_vec.replicate(1, num_obs) - time_vec.transpose().replicate(num_obs, 1);
        kernel_time = temp.unaryExpr([&l_t](double t)
                                     { return square_exp_kernel_time(t, l_t); });
        // -------------------------------

        // -------------------------------
        // kernel conv time
        temp_2 = time_vec.replicate(1, num_obs) + time_vec.transpose().replicate(num_obs, 1);
        square_exp_conv_time_kernel(t0, temp, temp_2, l_t, kernel_conv_time);
        // -------------------------------

        // -------------------------------
        // kernel space
        kernel_row_vec square_norms = dataset_x.rowwise().squaredNorm();
        temp_2 = square_norms.replicate(num_obs, 1) + square_norms.transpose().replicate(1, num_obs);
        temp = temp_2 - (2.0 * dataset_x * dataset_x.transpose());

        temp_2 = temp.unaryExpr([&l_s, &nu_s](double norm_squared)
                                { return matern_kernel_space(norm_squared, nu_s, l_s); });
        // -------------------------------

        k = lambda * temp_2.cwiseProduct(kernel_time);

        // -------------------------------
        // kernel conv space
        temp_2 = temp.unaryExpr([&l_s, &nu_s, &d](double norm_squared)
                                { return matern_conv_space_kernel(norm_squared, d, nu_s, l_s); });
        // -------------------------------

        coeff_c_plus = temp_2.cwiseProduct(kernel_conv_time);
    }
    else if (dynamic_cast<const RBFKernel *>(&kernel_space) != nullptr && dynamic_cast<const MaternKernel *>(&kernel_time) != nullptr)
    {
        // SPACE EXP AND TIME MATERN
        const RBFKernel *rbf_space = dynamic_cast<const RBFKernel *>(&kernel_space);
        const MaternKernel *matern_time = dynamic_cast<const MaternKernel *>(&kernel_time);

        double l_s, junk;
        std::tie(l_s, junk) = rbf_space->params();

        double l_t, nu_t;
        std::tie(l_t, nu_t) = matern_time->params();

        kernel_array temp(num_obs, num_obs);
        kernel_array temp_2(num_obs, num_obs);
        kernel_array kernel_time(num_obs, num_obs);
        kernel_array kernel_conv_time(num_obs, num_obs);
        kernel_array results_derivative(num_obs, num_obs);

        // t0 - ti
        kernel_array param_polynome = (t0 - time_vec.array()).matrix().replicate(1, num_obs);

        // -------------------------------
        // kernel time
        temp = time_vec.replicate(1, num_obs) - time_vec.transpose().replicate(num_obs, 1);
        kernel_time = temp.unaryExpr([&l_t, &nu_t](double t)
                                     { return matern_kernel_time(abs(t), nu_t, l_t); });
        // -------------------------------

        // -------------------------------
        // kernel conv time
        temp_2 = time_vec.replicate(1, num_obs) + time_vec.transpose().replicate(num_obs, 1);
        matern_conv_time_kernel(t0, nu_t, l_t, num_obs, temp, temp_2, param_polynome,
                                results_derivative, kernel_conv_time);
        // -------------------------------

        // -------------------------------
        // kernel space
        kernel_row_vec square_norms = dataset_x.rowwise().squaredNorm();
        temp_2 = square_norms.replicate(num_obs, 1) + square_norms.transpose().replicate(1, num_obs);
        temp = temp_2 - (2.0 * dataset_x * dataset_x.transpose());

        temp_2 = temp.unaryExpr([&l_s](double norm_squared)
                                { return square_exp_kernel_space(norm_squared, l_s); });
        // -------------------------------

        k = lambda * temp_2.cwiseProduct(kernel_time);

        // -------------------------------
        // kernel conv space
        temp_2 = temp.unaryExpr([&l_s, &d](double norm_squared)
                                { return square_exp_conv_space_kernel(norm_squared, d, l_s); });
        // -------------------------------

        coeff_c_plus = temp_2.cwiseProduct(kernel_conv_time);
    }
    else if (dynamic_cast<const MaternKernel *>(&kernel_space) != nullptr && dynamic_cast<const MaternKernel *>(&kernel_time) != nullptr)
    {
        // SPACE AND TIME MATERN
        const MaternKernel *matern_space = dynamic_cast<const MaternKernel *>(&kernel_space);
        const MaternKernel *matern_time = dynamic_cast<const MaternKernel *>(&kernel_time);

        double l_s, nu_s;
        std::tie(l_s, nu_s) = matern_space->params();

        double l_t, nu_t;
        std::tie(l_t, nu_t) = matern_time->params();

        kernel_array temp(num_obs, num_obs);
        kernel_array temp_2(num_obs, num_obs);
        kernel_array kernel_time(num_obs, num_obs);
        kernel_array kernel_conv_time = kernel_array::Zero(num_obs, num_obs);
        kernel_array results_derivative = kernel_array::Zero(num_obs, num_obs);

        // t0 - ti
        kernel_array param_polynome = (t0 - time_vec.array()).matrix().replicate(1, num_obs);

        // -------------------------------
        // kernel time
        temp = time_vec.replicate(1, num_obs) - time_vec.transpose().replicate(num_obs, 1);
        kernel_time = temp.unaryExpr([&l_t, &nu_t](double t)
                                     { return matern_kernel_time(abs(t), nu_t, l_t); });
        // -------------------------------

        // -------------------------------
        // kernel conv time
        temp_2 = time_vec.replicate(1, num_obs) + time_vec.transpose().replicate(num_obs, 1);
        matern_conv_time_kernel(t0, nu_t, l_t, num_obs, temp, temp_2, param_polynome,
                                results_derivative, kernel_conv_time);
        // -------------------------------

        // -------------------------------
        // kernel space
        kernel_row_vec square_norms = dataset_x.rowwise().squaredNorm();
        temp_2 = square_norms.replicate(num_obs, 1) + square_norms.transpose().replicate(1, num_obs);
        temp = temp_2 - (2.0 * dataset_x * dataset_x.transpose());

        temp_2 = temp.unaryExpr([&l_s, &nu_s](double norm_squared)
                                { return matern_kernel_space(norm_squared, nu_s, l_s); });
        // -------------------------------

        k = lambda * temp_2.cwiseProduct(kernel_time);

        // -------------------------------
        // kernel conv space
        temp_2 = temp.unaryExpr([&l_s, &nu_s, &d](double norm_squared)
                                { return matern_conv_space_kernel(norm_squared, d, nu_s, l_s); });
        // -------------------------------
        coeff_c_plus = temp_2.cwiseProduct(kernel_conv_time);
    }
    else if (dynamic_cast<const ARDKernel *>(&kernel_space) != nullptr && dynamic_cast<const RBFKernel *>(&kernel_time) != nullptr)
    {
        // SPACE ARD AND TIME EXP

        ARDKernel *ard_space = dynamic_cast<ARDKernel *>(&kernel_space);
        const RBFKernel *rbf_time = dynamic_cast<const RBFKernel *>(&kernel_time);

        kernel_vec lengthscales_ard = ard_space->getLengthScales();

        double l_t, junk_;
        std::tie(l_t, junk_) = rbf_time->params();

        kernel_vec ard_inverse = lengthscales_ard.unaryExpr([](double val)
                                                            { return 1.0 / (val * val); });

        kernel_array sigma_inverse_2(d, d);
        sigma_inverse_2 = ard_inverse.asDiagonal();
        double determinant = lengthscales_ard.prod();

        kernel_array temp(num_obs, num_obs);
        kernel_array temp_2(num_obs, num_obs);
        kernel_array kernel_time(num_obs, num_obs);
        kernel_array kernel_conv_time(num_obs, num_obs);

        // -------------------------------
        // kernel time
        temp = time_vec.replicate(1, num_obs) - time_vec.transpose().replicate(num_obs, 1);
        kernel_time = temp.unaryExpr([&l_t](double t)
                                     { return square_exp_kernel_time(t, l_t); });
        // -------------------------------

        // -------------------------------
        // kernel conv time
        temp_2 = time_vec.replicate(1, num_obs) + time_vec.transpose().replicate(num_obs, 1);
        square_exp_conv_time_kernel(t0, temp, temp_2, l_t, kernel_conv_time);
        // -------------------------------

        kernel_vec xi(d), xj(d), xi_minus_xj(d);
        for (int i = 0; i < num_obs; ++i)
        {
            for (int j = 0; j < num_obs; ++j)
            {

                xi = dataset_x.row(i);
                xj = dataset_x.row(j);
                xi_minus_xj = xi - xj;

                k(i, j) = lambda * ard_space_kernel(xi_minus_xj, d, sigma_inverse_2) * kernel_time(i, j);
                coeff_c_plus(i, j) = kernel_conv_time(i, j) * ard_space_conv_kernel(xi_minus_xj, d, sigma_inverse_2, determinant);
            }
        }
    }
    else if (dynamic_cast<const ARDKernel *>(&kernel_space) != nullptr && dynamic_cast<const MaternKernel *>(&kernel_time) != nullptr)
    {
        // SPACE ARD AND TIME MATERN

        const ARDKernel *ard_space = dynamic_cast<const ARDKernel *>(&kernel_space);
        const MaternKernel *matern_time = dynamic_cast<const MaternKernel *>(&kernel_time);

        kernel_vec lengthscales_ard = ard_space->lengthscales;

        double l_t, nu_t;
        std::tie(l_t, nu_t) = matern_time->params();

        kernel_vec ard_inverse = lengthscales_ard.unaryExpr([](double val)
                                                            { return 1.0 / (val * val); });

        kernel_array sigma_inverse_2(d, d);
        sigma_inverse_2 = ard_inverse.asDiagonal();
        double determinant = lengthscales_ard.prod();

        kernel_array temp(num_obs, num_obs);
        kernel_array temp_2(num_obs, num_obs);
        kernel_array kernel_time(num_obs, num_obs);
        kernel_array kernel_conv_time = kernel_array::Zero(num_obs, num_obs);
        kernel_array results_derivative = kernel_array::Zero(num_obs, num_obs);

        // t0 - ti
        kernel_array param_polynome = (t0 - time_vec.array()).matrix().replicate(1, num_obs);

        // -------------------------------
        // kernel time
        temp = time_vec.replicate(1, num_obs) - time_vec.transpose().replicate(num_obs, 1);
        kernel_time = temp.unaryExpr([&l_t, &nu_t](double t)
                                     { return matern_kernel_time(abs(t), nu_t, l_t); });
        // -------------------------------

        // -------------------------------
        // kernel conv time
        temp_2 = time_vec.replicate(1, num_obs) + time_vec.transpose().replicate(num_obs, 1);
        matern_conv_time_kernel(t0, nu_t, l_t, num_obs, temp, temp_2, param_polynome,
                                results_derivative, kernel_conv_time);
        // -------------------------------

        kernel_vec xi(d), xj(d), xi_minus_xj(d);
        for (int i = 0; i < num_obs; ++i)
        {
            for (int j = 0; j < num_obs; ++j)
            {

                xi = dataset_x.row(i);
                xj = dataset_x.row(j);
                xi_minus_xj = xi - xj;

                k(i, j) = lambda * ard_space_kernel(xi_minus_xj, d, sigma_inverse_2) * kernel_time(i, j);
                coeff_c_plus(i, j) = kernel_conv_time(i, j) * ard_space_conv_kernel(xi_minus_xj, d, sigma_inverse_2, determinant);
            }
        }
    }
}

void create_kernel_tilde_and_C_plus(kernel_array &matrix_kernel, int index, kernel_array &result_kernel,
                                    kernel_array &c_plus, kernel_array &result_c_plus,
                                    const int num_obs)
{
    // Concatenate slices of the matrix before and after the row and column to remove
    result_kernel << matrix_kernel.block(0, 0, index, index),                                // Top-left block
        matrix_kernel.block(0, index + 1, index, num_obs - index - 1),                       // Top-right block
        matrix_kernel.block(index + 1, 0, num_obs - index - 1, index),                       // Bottom-left block
        matrix_kernel.block(index + 1, index + 1, num_obs - index - 1, num_obs - index - 1); // Bottom-right block

    result_c_plus << c_plus.block(0, 0, index, index),
        c_plus.block(0, index + 1, index, num_obs - index - 1),
        c_plus.block(index + 1, 0, num_obs - index - 1, index),
        c_plus.block(index + 1, index + 1, num_obs - index - 1, num_obs - index - 1);
}

void create_y_vector_tilde(Eigen::Ref<kernel_vec> &y_vec, int index, kernel_vec &result,
                           const int num_obs)
{
    result << y_vec.head(index),
        y_vec.tail(num_obs - index - 1);
}

// ----------------------------------------------------------------------------
/*
Please refer to the pdf report for details on the linear algebra computations
*/
// ----------------------------------------------------------------------------

int E_sub_product(kernel_array &kernel_tilde,
                  kernel_vec &vec, double &result)
{
    if (isPsd(kernel_tilde))
    {
        kernel_vec x = kernel_tilde.ldlt().solve(vec);
        result = vec.transpose() * x;
        return ERR_NONE;
    }
    return ERR_MATRIX_NOT_PSD;
}

int G_sub_product(kernel_array &kernel_tilde,
                  kernel_vec &vec,
                  kernel_row_vec &result)
{
    if (isPsd(kernel_tilde))
    {
        result = kernel_tilde.ldlt().solve(vec).transpose();
        return ERR_NONE;
    }
    return ERR_MATRIX_NOT_PSD;
}

int normalization_criterion_sub_product(kernel_array &matrix, Eigen::Ref<kernel_vec> &vec, kernel_vec &result)
{
    if (isPsd(matrix))
    {
        result = matrix.ldlt().solve(vec);
        return ERR_NONE;
    }
    return ERR_MATRIX_NOT_PSD;
}