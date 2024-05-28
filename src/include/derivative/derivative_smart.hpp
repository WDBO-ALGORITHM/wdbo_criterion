#ifndef DERIVATIVE_SMART_H
#define DERIVATIVE_SMART_H

#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <string>
#include <memory>

#include "../bayesian_types.hpp"

using std::vector;

class ExpMonome;
class Constant;

class Derivative
{
public:
    virtual ~Derivative() {}
    virtual std::shared_ptr<Derivative> derivative(const int obs) = 0;
    virtual kernel_array evaluate(kernel_array x) = 0;
};

class Constant : public Derivative
{
public:
    kernel_array c;

    ~Constant()
    {
    }

    Constant(kernel_array vec)
    {
        c = vec;
    }

    kernel_array evaluate(kernel_array x)
    {
        return c;
    }

    std::shared_ptr<Derivative> derivative(const int obs)
    {
        kernel_array zer = kernel_array::Zero(obs, obs);
        return std::shared_ptr<Derivative>(new Constant(zer));
    }
};

// Class for sum of two functions
class Sum : public Derivative
{
public:
    std::shared_ptr<Derivative> f1;
    std::shared_ptr<Derivative> f2;

    ~Sum()
    {
    }

    Sum(std::shared_ptr<Derivative> f1, std::shared_ptr<Derivative> f2) : f1(f1), f2(f2) {}

    std::shared_ptr<Derivative> derivative(const int obs)
    {
        std::shared_ptr<Derivative> grad_f1 = f1.get()->derivative(obs);
        std::shared_ptr<Derivative> grad_f2 = f2.get()->derivative(obs);

        // Checking if the derivatives of both functions are constants and zero
        bool gf1_null = typeid(*grad_f1) == typeid(Constant) &&
                        std::all_of(dynamic_cast<Constant *>(grad_f1.get())->c.reshaped().begin(),
                                    dynamic_cast<Constant *>(grad_f1.get())->c.reshaped().end(), [](double i)
                                    { return i == 0.0; });

        bool gf2_null = typeid(*grad_f2) == typeid(Constant) &&
                        std::all_of(dynamic_cast<Constant *>(grad_f2.get())->c.reshaped().begin(),
                                    dynamic_cast<Constant *>(grad_f2.get())->c.reshaped().end(), [](double i)
                                    { return i == 0.0; });

        if (gf1_null && gf2_null)
        {
            kernel_array zer = kernel_array::Zero(obs, obs);
            return std::shared_ptr<Derivative>(new Constant(zer));
        }
        else if (gf1_null)
        {
            return grad_f2;
        }
        else if (gf2_null)
        {
            return grad_f1;
        }
        else
        {
            return std::shared_ptr<Derivative>(new Sum(grad_f1, grad_f2));
        }
    }

    kernel_array evaluate(kernel_array x)
    {
        return f1.get()->evaluate(x) + f2.get()->evaluate(x);
    }
};

class Product : public Derivative
{
public:
    std::shared_ptr<Derivative> f1;
    std::shared_ptr<Derivative> f2;

    ~Product()
    {
    }

    Product(std::shared_ptr<Derivative> f1, std::shared_ptr<Derivative> f2) : f1(f1), f2(f2) {}

    std::shared_ptr<Derivative> derivative(const int obs)
    {
        std::shared_ptr<Derivative> grad_f1 = f1.get()->derivative(obs);
        std::shared_ptr<Derivative> grad_f2 = f2.get()->derivative(obs);

        bool gf1_null = typeid(*grad_f1) == typeid(Constant) &&
                        std::all_of(dynamic_cast<Constant *>(grad_f1.get())->c.reshaped().begin(),
                                    dynamic_cast<Constant *>(grad_f1.get())->c.reshaped().end(), [](double i)
                                    { return i == 0.0; });
        bool gf2_null = typeid(*grad_f2) == typeid(Constant) &&
                        std::all_of(dynamic_cast<Constant *>(grad_f2.get())->c.reshaped().begin(),
                                    dynamic_cast<Constant *>(grad_f2.get())->c.reshaped().end(), [](double i)
                                    { return i == 0.0; });
        bool gf1_1 = typeid(*grad_f1) == typeid(Constant) &&
                     std::all_of(dynamic_cast<Constant *>(grad_f1.get())->c.reshaped().begin(),
                                 dynamic_cast<Constant *>(grad_f1.get())->c.reshaped().end(), [](double i)
                                 { return i == 1.0; });
        bool gf2_1 = typeid(*grad_f2) == typeid(Constant) &&
                     std::all_of(dynamic_cast<Constant *>(grad_f2.get())->c.reshaped().begin(),
                                 dynamic_cast<Constant *>(grad_f2.get())->c.reshaped().end(), [](double i)
                                 { return i == 1.0; });

        if (gf1_null && gf2_null)
        {
            kernel_array zer = kernel_array::Zero(obs, obs);
            return std::shared_ptr<Derivative>(new Constant(zer));
        }
        else if (gf1_null)
        {
            return std::shared_ptr<Derivative>(new Product(f1, grad_f2));
        }
        else if (gf2_null)
        {
            return std::shared_ptr<Derivative>(new Product(f2, grad_f1));
        }
        else if (gf1_1 && gf2_1)
        {
            return std::shared_ptr<Derivative>(new Sum(f1, f2));
        }
        else if (gf1_1)
        {
            return std::shared_ptr<Derivative>(new Sum(f2, std::shared_ptr<Derivative>(new Product(f1, grad_f2))));
        }
        else if (gf2_1)
        {
            return std::shared_ptr<Derivative>(new Sum(f1, std::shared_ptr<Derivative>(new Product(f2, grad_f1))));
        }
        else
        {
            return std::shared_ptr<Derivative>(new Sum(std::shared_ptr<Derivative>(new Product(grad_f1, f2)),
                                                       std::shared_ptr<Derivative>(new Product(grad_f2, f1))));
        }
    }

    kernel_array evaluate(kernel_array x)
    {
        return (f1.get()->evaluate(x).array() * f2.get()->evaluate(x).array()).matrix();
    }
};

class ExpMonome : public Derivative
{
public:
    kernel_array vec;
    int power;

    ~ExpMonome()
    {
    }

    ExpMonome(kernel_array v, int p) : vec(v), power(p) {}

    std::shared_ptr<Derivative> derivative(const int obs)
    {
        if (power == 0)
        {
            kernel_array zer = kernel_array::Zero(obs, obs);
            return std::move(std::shared_ptr<Derivative>(new Constant(zer)));
        }
        else if (power == 1)
        {
            kernel_array one = kernel_array::Ones(obs, obs);
            return std::move(std::shared_ptr<Derivative>(new Constant(one)));
        }
        else
        {
            kernel_array one = kernel_array::Ones(obs, obs);
            kernel_array res = one * power;
            return std::move(std::shared_ptr<Derivative>(new Product(
                std::move(std::shared_ptr<Derivative>(new Constant(res))),
                std::move(std::shared_ptr<Derivative>(new ExpMonome(vec, power - 1))))));
        }
    }

    kernel_array evaluate(kernel_array x)
    {
        kernel_array res = vec + x;
        return res.array().pow(power).matrix();
    }
};

#endif