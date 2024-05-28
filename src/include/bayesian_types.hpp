#ifndef BAYESIAN_TYPES_H
#define BAYESIAN_TYPES_H

#include <Eigen/Dense>

using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::RowMajor;
using Eigen::RowVector;
using Eigen::Vector;

using kernel_array = Matrix<double, Dynamic, Dynamic>;

using kernel_vec = Vector<double, Dynamic>;

using kernel_row_vec = RowVector<double, Dynamic>;

using compatible_storage_order_matrix = Matrix<double, Dynamic, Dynamic, RowMajor>;

#endif