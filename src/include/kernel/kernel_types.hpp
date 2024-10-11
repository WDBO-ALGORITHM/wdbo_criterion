#ifndef KERNEL_TYPES_H
#define KERNEL_TYPES_H

#include <iostream>
#include <tuple>
#include "../bayesian_types.hpp"
#include <Eigen/Dense>

/**
 * @brief Kernel abstract class
 * 
 */
class KernelParams
{
public:
    virtual ~KernelParams() = default;
    virtual std::tuple<double, double> params() const = 0;
    virtual void print() const = 0;
};

/**
 * @brief RBF Kernel
 * 
 */
class RBFKernel : public KernelParams
{
public:
    double lengthscale;
    RBFKernel(double lengthscale) : lengthscale(lengthscale) {}

    std::tuple<double, double> params() const override
    {
        return std::make_tuple(lengthscale, 1.0);
    }

    void print() const override
    {
        std::cout << "lengthscale " << lengthscale << std::endl;
    }
};

/**
 * @brief ARD Kernel. Vector of spatial lengthscales.
 * 
 */
class ARDKernel : public KernelParams
{
public:
    kernel_vec lengthscales;
    ARDKernel(kernel_vec lengthscales) : lengthscales(lengthscales) {}

    std::tuple<double, double> params() const override
    {
        return std::make_tuple(1.0, 1.0);
    }

    
    const kernel_vec &getLengthScales()
    {
        return lengthscales;
    }
    
    void print() const override
    {
        std::cout << "lengthscales " << std::endl;
    }
};

/**
 * @brief Matern Kernel
 * 
 */
class MaternKernel : public KernelParams
{
public:
    double lengthscale;
    double nu;

    MaternKernel(double lengthscale, double nu) : lengthscale(lengthscale), nu(nu) {};

    std::tuple<double, double> params() const override
    {
        return std::make_tuple(lengthscale, nu);
    }

    void print() const override
    {
        std::cout << "lengthscale " << lengthscale << " / nu " << nu << std::endl;
    }
};

#endif