#ifndef KERNEL_TYPES_H
#define KERNEL_TYPES_H

#include <iostream>
#include <tuple>

class KernelParams
{
public:
    virtual ~KernelParams() = default;
    virtual std::tuple<double, double> params() const = 0;
    virtual void print() const = 0;
};

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

class MaternKernel : public KernelParams
{
public:
    double lengthscale;
    double nu;

    MaternKernel(double lengthscale, double nu) : lengthscale(lengthscale), nu(nu){};

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