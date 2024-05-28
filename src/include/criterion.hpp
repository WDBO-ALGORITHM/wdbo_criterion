#ifndef CRITERION_H
#define CRITERION_H

#include <cmath>

#include "bayesian_types.hpp"
#include "kernel/kernel.hpp"

double E_val(const double lambda,
             const double variance,
             kernel_row_vec &vec_row,
             kernel_vec &vec);

void G_val(const double e_value, kernel_row_vec &vec_row,
           kernel_row_vec &result);

void H_val(const double lambda, const double variance, kernel_vec &vec,
           kernel_array &A_value, kernel_vec &result);

double a_val(const double e_value, kernel_vec &ys_without_target, const double y_1,
             kernel_row_vec &g_value);

int F_val(kernel_array &A_value, kernel_array &result);

int inverse_matrix_cholsky(kernel_array &delta, kernel_array &result);

void M_val(kernel_array &F_value,
           kernel_array &delta_inverse_tilde,
           kernel_array &M_value);

void b_val(kernel_vec &h_value, const double y_1, kernel_vec &ys_without_target,
           kernel_array &m_value,
           kernel_row_vec &result);

void c_val(kernel_vec &h_value, kernel_row_vec &g_value, kernel_row_vec &result);

int normalization_criterion(const double lambda, kernel_vec &a_vec, kernel_array &c_plus_coeff, kernel_array &inverse_delta,
                            const double num_obs, double &result);

#endif