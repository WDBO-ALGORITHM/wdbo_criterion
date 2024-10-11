// pybind11 main library C++
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

// libraries of the project
#include "../include/bayesian.hpp"
#include "../include/kernel/kernel_types.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(wdbo_criterion, m)
{
    m.doc() = "C++ implementation of a criterion used to remove stale data in Dynamic Bayesian Optimization.";

    py::class_<KernelParams>(m, "KernelParams")
        .def("params", &KernelParams::params)
        .def("print", &KernelParams::print);

    py::class_<RBFKernel, KernelParams>(m, "RBFKernel")
        .def(py::init<double>())
        .def("params", &RBFKernel::params)
        .def("print", &RBFKernel::print)
        .def_readwrite("lengthscale", &RBFKernel::lengthscale);

    py::class_<ARDKernel, KernelParams>(m, "ARDKernel")
        .def(py::init<double>())
        .def("params", &ARDKernel::params)
        .def("print", &ARDKernel::print)
        .def_readwrite("lengthscales", &ARDKernel::lengthscales);

    py::class_<MaternKernel, KernelParams>(m, "MaternKernel")
        .def(py::init<double, double>())
        .def("params", &MaternKernel::params)
        .def("print", &MaternKernel::print)
        .def_readwrite("lengthscale", &MaternKernel::lengthscale)
        .def_readwrite("nu", &MaternKernel::nu);

    m.def("wasserstein_criterion", &wasserstein_criterion, py::return_value_policy::reference_internal);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
