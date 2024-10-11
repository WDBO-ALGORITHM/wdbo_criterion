#include <iostream>
#include <exception>

#include "../include/bayesian_types.hpp"
#include "../include/helper/helper_functions.hpp"
#include "../include/kernel/kernel.hpp"
#include "../include/kernel/kernel_functions.hpp"
#include "../include/criterion.hpp"
#include "../include/bayesian.hpp"
#include "../include/kernel/kernel_types.hpp"

/*
Entry point. Computes the criterion for each point in the dataset provided.
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
                                 const int normalize_criterion)
{
    if (variance < 1e-05)
    {
        variance = 1e-05;
    }

    if (verbose == 1)
    {
        std::cout << dataset_x << std::endl;
        std::cout << "------------------------------------" << std::endl;
        std::cout << "number of points:" << std::endl;
        std::cout << num_obs << std::endl;
        std::cout << "dimension of points:" << std::endl;
        std::cout << dim << std::endl;
        std::cout << "lambda:" << std::endl;
        std::cout << lambda << std::endl;
        std::cout
            << "variance (casted to 1e-05 is provided less):" << std::endl;
        std::cout << variance << std::endl;
        std::cout << "space kernel" << std::endl;
        kernel_space.print();
        std::cout << "time kernel" << std::endl;
        kernel_time.print();
        std::cout << "t0:" << std::endl;
        std::cout << t0 << std::endl;
        std::cout << "normalized (0 -> no, 1 -> yes):" << std::endl;
        std::cout << normalization_criterion << std::endl;
    }

    if (num_obs <= 1 || dim <= 0 || lambda == 0)
    {
        throw std::invalid_argument("Use verbose param to show which param in {num_obs, dim, lambda} is not set correctly");
    }

    if (dynamic_cast<const RBFKernel *>(&kernel_space) != nullptr)
    {
        const RBFKernel *rbf = dynamic_cast<const RBFKernel *>(&kernel_space);
        if (rbf->lengthscale <= 0)
        {
            throw std::invalid_argument("lengthscale of rbf space kernel is not valid");
        }
    }

    if (dynamic_cast<const RBFKernel *>(&kernel_time) != nullptr)
    {
        const RBFKernel *rbf = dynamic_cast<const RBFKernel *>(&kernel_time);
        if (rbf->lengthscale <= 0)
        {
            throw std::invalid_argument("lengthscale of rbf time kernel is not valid");
        }
    }

    // TODO: ADD TEST ARD KERNEL

    if (dynamic_cast<const MaternKernel *>(&kernel_space) != nullptr)
    {
        const MaternKernel *matern = dynamic_cast<const MaternKernel *>(&kernel_space);
        if (matern->lengthscale <= 0 || !isPositiveHalfInteger(matern->nu))
        {
            throw std::invalid_argument("nu_space or nu_time is invalide. It should by a positive half integer");
        }
    }

    if (dynamic_cast<const MaternKernel *>(&kernel_time) != nullptr)
    {
        const MaternKernel *matern = dynamic_cast<const MaternKernel *>(&kernel_time);
        if (matern->lengthscale <= 0 || !isPositiveHalfInteger(matern->nu))
        {
            throw std::invalid_argument("nu_space or nu_time is invalide. It should by a positive half integer");
        }
    }

    if (dataset_x.rows() != num_obs || dataset_x.cols() != dim)
    {
        throw std::invalid_argument("dimensions of points' matrix is invalid");
    }

    if (y_vector.size() != num_obs || time_vec.size() != num_obs)
    {
        throw std::invalid_argument("dimensions of vector of timings of function values is invalid");
    }

    kernel_array kernel_D(num_obs, num_obs);
    kernel_array coeff_c_plus(num_obs, num_obs);
    create_kernel_and_coeff_C_Plus(kernel_D, dataset_x, time_vec, kernel_space, kernel_time,
                                   coeff_c_plus, dim, t0, num_obs, lambda);

    kernel_D = kernel_D + (variance * kernel_array::Identity(num_obs, num_obs));

    kernel_vec results_criterion = wasserstein_critetion_point(kernel_D, lambda, variance, y_vector, coeff_c_plus,
                                                               num_obs, normalize_criterion);
    return results_criterion;
}


kernel_vec wasserstein_critetion_point(kernel_array &kernel_D, const double lambda,
                                       const double variance, Eigen::Ref<kernel_vec> &y_vec, kernel_array &coeff_c_plus,
                                       const int num_obs, const int normalize_criterion)
{
    kernel_vec results = kernel_vec::Zero(num_obs);
    const int num_obs_minus_1 = num_obs - 1;
    int index = 0;
    int error_code = 0;
    const double constante_1 = 1.0 / (lambda + variance);

    // --------------

    kernel_array kernel_D_tilde(num_obs_minus_1, num_obs_minus_1);
    kernel_vec k_vector(num_obs_minus_1);
    kernel_array coeff_c_plus_tilde(num_obs_minus_1, num_obs_minus_1);

    // --------------

    kernel_row_vec g_sub_prod(num_obs_minus_1);
    kernel_row_vec g_value(num_obs_minus_1);

    // --------------

    kernel_array A_value(num_obs_minus_1, num_obs_minus_1);
    kernel_vec h_value(num_obs_minus_1);
    kernel_array F_value(num_obs_minus_1, num_obs_minus_1);
    kernel_array inverse_tilde(num_obs_minus_1, num_obs_minus_1);
    kernel_array M_value(num_obs_minus_1, num_obs_minus_1);

    // --------------

    kernel_vec y_vec_tilde(num_obs_minus_1);

    // --------------

    kernel_row_vec b_value(num_obs_minus_1);
    kernel_row_vec c_value(num_obs_minus_1);

    // --------------

    kernel_vec conv_vec(num_obs_minus_1);
    kernel_array sub_prod_hadamard(num_obs_minus_1, num_obs_minus_1);
    kernel_array had(num_obs_minus_1, num_obs_minus_1);

    // -------------
    // normalization
    // -------------
    kernel_vec a_vec(num_obs);
    kernel_array inverse_delta(num_obs, num_obs);

    error_code = normalization_criterion_sub_product(kernel_D, y_vec, a_vec);
    if (error_code != 0)
    {
        throw std::runtime_error("error normalization factor not PD");
    }

    error_code = inverse_matrix_cholsky(kernel_D, inverse_delta);
    if (error_code != 0)
    {
        throw std::runtime_error("error matrix delta not PD");
    }

    double normalization_crit = 0.0;
    error_code = normalization_criterion(lambda, a_vec, coeff_c_plus, inverse_delta, num_obs, normalization_crit);
    if (error_code != 0)
    {
        throw std::runtime_error("error normalization factor square root negative");
    }

    while (index < num_obs)
    {

        // calcul du kernel_tilde
        create_kernel_tilde_and_C_plus(kernel_D, index, kernel_D_tilde, coeff_c_plus, coeff_c_plus_tilde, num_obs);

        // store vector k
        kernel_column_vector_and_conv_vector(k_vector, conv_vec, index, kernel_D, coeff_c_plus, num_obs);

        // calcul de g_sub_prod = k^{T} * Delta^{-1}
        error_code = G_sub_product(kernel_D_tilde, k_vector, g_sub_prod);
        if (error_code != 0)
        {
            throw std::runtime_error("error G sub product");
        }

        // calcul de E (avec g_sub_prod et k)
        double e_value = E_val(lambda, variance, g_sub_prod, k_vector);

        // calcul de G (avec E et g_sub_prod)
        G_val(e_value, g_sub_prod, g_value);

        // calcul de w = Delta - ... * k * k^{T}
        A_value = kernel_D_tilde - (constante_1 * k_vector * k_vector.transpose());

        // calcul de H (avec w et k)
        H_val(lambda, variance, k_vector, A_value, h_value);

        error_code = F_val(A_value, F_value);
        if (error_code != 0)
        {
            throw std::runtime_error("error F val");
        }
        error_code = inverse_matrix_cholsky(kernel_D_tilde, inverse_tilde);
        if (error_code != 0)
        {
            throw std::runtime_error("error matrix delta is not PD");
        }
        M_val(F_value, inverse_tilde, M_value);

        // calcul de a (avec E et G)
        create_y_vector_tilde(y_vec, index, y_vec_tilde, num_obs);
        double a_value = a_val(e_value, y_vec_tilde, y_vec(index), g_value);

        // calcul de b (avec H et M)
        b_val(h_value, y_vec(index), y_vec_tilde, M_value, b_value);

        // calcul de c (avec G et H)
        c_val(h_value, g_value, c_value);

        sub_prod_hadamard = b_value.transpose() * b_value + M_value;
        had = sub_prod_hadamard.cwiseProduct(coeff_c_plus_tilde);

        double last_term = std::pow(lambda, 2) * kernel_vec::Ones(num_obs_minus_1).transpose() *
                           had * kernel_vec::Ones(num_obs_minus_1);

        double first_term = (std::pow(lambda, 2) * (std::pow(a_value, 2) + e_value) *
                             coeff_c_plus(index, index)) +
                            (std::pow(lambda, 2) * (2 * a_value * b_value + c_value) * conv_vec);

        if (first_term + last_term >= 0)
        {
            if (normalize_criterion == 1)
            {
                if (normalization_crit != 0)
                {
                    results(index) = std::sqrt((first_term + last_term)) / normalization_crit;
                }
                else
                {
                    results(index) = -2;
                }
            }
            else
            {
                results(index) = std::sqrt((first_term + last_term));
            }
        }
        else
        {
            results(index) = -1;
        }

        ++index;
    }
    return results;
}