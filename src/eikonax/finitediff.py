"""Finite difference approximation of parametric derivatives.

!!! warning

    Finite difference approximations are typically computationally expensive and inaccurate. They
    should only be used for comparison in small test cases.

Functions:
    finite_diff_1_forward: Forward finite difference approximation of a first order derivative
    finite_diff_1_backward: Backward finite difference approximation of a first order derivative
    finite_diff_1_central: Central finite difference approximation of a first order derivative
    finite_diff_2: Implement second order finite differences
    compute_fd_jacobian: Compute the Jacobian of the Eikonal equation w.r.t. to parameter with
        finite differences
"""

from collections.abc import Callable
from functools import partial
from numbers import Real

import jax
import numpy as np
import numpy.typing as npt
from jaxtyping import Real as jtReal

from eikonax import solver, tensorfield


# ==================================================================================================
def finite_diff_1_forward(
    func: Callable[[jtReal[npt.NDArray, "M"]], jtReal[npt.NDArray, "M"]],
    eval_point: jtReal[npt.NDArray, "M"],
    step_width: Real,
    index: int,
) -> jtReal[npt.NDArray, "N"]:
    r"""Forward finite difference approximation of a first order derivative.

    This method expects vector-valued functions $f: \mathbb{R}^M \to \mathbb{R}^N$, and approximates
    the first derivative of $f$ at a given point $x \in \mathbb{R}^M$ with respect to the $i$-th
    component of $x$ as

    $$
        \frac{\partial f(x)}{\partial x_i} \approx \frac{f(x + h e_i) - f(x)}{h}
    $$

    with a step width $h>0$.

    Args:
        func (Callable): Callable to use for FD computation
        eval_point (npt.NDArray): Parameter value at which to approximate the derivative
        step_width (Real): Step width of the finite difference
        index (int): Vector component for which to compute partial derivative

    Returns:
        npt.NDArray: Partial derivative approximation
    """
    unperturbed_eval = func(eval_point)
    eval_point[index] += step_width
    fwd_perturbed_eval = func(eval_point)
    eval_point[index] -= step_width
    finite_diff = (fwd_perturbed_eval - unperturbed_eval) / step_width
    return finite_diff


# --------------------------------------------------------------------------------------------------
def finite_diff_1_backward(
    func: Callable[[jtReal[npt.NDArray, "M"]], jtReal[npt.NDArray, "M"]],
    eval_point: jtReal[npt.NDArray, "M"],
    step_width: float,
    index: int,
) -> jtReal[npt.NDArray, "N"]:
    r"""Backward finite difference approximation of a first order derivative.

    This method expects vector-valued functions $f: \mathbb{R}^M \to \mathbb{R}^N$, and approximates
    the first derivative of $f$ at a given point $x \in \mathbb{R}^M$ with respect to the $i$-th
    component of $x$ as

    $$
        \frac{\partial f(x)}{\partial x_i} \approx \frac{f(x)- f(x - h e_i)}{h}
    $$

    with a step width $h>0$.

    Args:
        func (Callable): Callable to use for FD computation
        eval_point (npt.NDArray): Parameter value at which to approximate the derivative
        step_width (Real): Step width of the finite difference
        index (int): Vector component for which to compute partial derivative

    Returns:
        npt.NDArray: Partial derivative approximation
    """
    unperturbed_eval = func(eval_point)
    eval_point[index] -= step_width
    bwd_perturbed_eval = func(eval_point)
    eval_point[index] += step_width
    finite_diff = (unperturbed_eval - bwd_perturbed_eval) / step_width
    return finite_diff


# --------------------------------------------------------------------------------------------------
def finite_diff_1_central(
    func: Callable[[jtReal[npt.NDArray, "M"]], jtReal[npt.NDArray, "M"]],
    eval_point: jtReal[npt.NDArray, "M"],
    step_width: float,
    index: int,
) -> jtReal[npt.NDArray, "N"]:
    r"""Central finite difference approximation of a first order derivative.

    This method expects vector-valued functions $f: \mathbb{R}^M \to \mathbb{R}^N$, and approximates
    the first derivative of $f$ at a given point $x \in \mathbb{R}^M$ with respect to the $i$-th
    component of $x$ as

    $$
        \frac{\partial f(x)}{\partial x_i} \approx \frac{f(x + h e_i)- f(x - h e_i)}{2h}
    $$

    with a step width $h>0$.

    Args:
        func (Callable): Callable to use for FD computation
        eval_point (npt.NDArray): Parameter value at which to approximate the derivative
        step_width (Real): Step width of the finite difference
        index (int): Vector component for which to compute partial derivative

    Returns:
        npt.NDArray: Partial derivative approximation
    """
    eval_point[index] += step_width
    fwd_perturbed_eval = func(eval_point)
    eval_point[index] -= 2 * step_width
    bwd_perturbed_eval = func(eval_point)
    eval_point[index] += step_width
    finite_diff = (fwd_perturbed_eval - bwd_perturbed_eval) / (2 * step_width)
    return finite_diff


# --------------------------------------------------------------------------------------------------
def finite_diff_2(
    func: Callable[[jtReal[npt.NDArray, "M"]], jtReal[npt.NDArray, "M"]],
    eval_point: jtReal[npt.NDArray, "M"],
    step_width: float,
    index_1: int,
    index_2: int,
) -> None:
    """Implement second order finite differences.

    !!! failure "Not implemented yet"
    """
    raise NotImplementedError


# ==================================================================================================
def run_eikonax_with_tensorfield(
    parameter_vector: jtReal[npt.NDArray, "M"],
    eikonax_solver: solver.Solver,
    tensor_field: tensorfield.TensorField,
) -> jtReal[npt.NDArray, "N"]:
    """Wrapper function for Eikonax runs.

    Args:
        parameter_vector (npt.NDArray): Parameter vector at which to compute eikonal solution
        eikonax_solver (solver.Solver): Initialized solver object
        tensor_field (tensorfield.TensorField): Initialized tensor field object

    Returns:
        npt.NDArray: Solution of the Eikonal equation
    """
    parameter_field = tensor_field.assemble_field(parameter_vector)
    solution = eikonax_solver.run(parameter_field)
    solution_values = np.array(solution.values)
    return solution_values


# --------------------------------------------------------------------------------------------------
def compute_fd_jacobian(
    eikonax_solver: solver.Solver,
    tensor_field: tensorfield.TensorField,
    stencil: Callable,
    eval_point: jtReal[npt.NDArray | jax.Array, "M"],
    step_width: float,
) -> jtReal[npt.NDArray, "N M"]:
    r"""Finite Difference Jacobian.

    Compute the Jacobian of the discrete Eikonal equation solution $\mathbf{u}\in\mathbb{R}^N$
    w.r.t. to a parameter vector $\mathbf{m}\in\mathbb{R}^M$ with finite differences.

    !!! warning
        This method should only be used for small problems.

    Args:
        eikonax_solver (solver.Solver): Initialized solver object
        tensor_field (tensorfield.TensorField): Initialized tensor field object
        stencil (Callable): Finite difference stencil to use for computation
        eval_point (npt.NDArray): Parameter vector at which to approximate derivative
        step_width (float): Step with in FD stencil

    Returns:
        npt.NDArray: (Dense) Jacobian matrix
    """
    eval_func = partial(
        run_eikonax_with_tensorfield, eikonax_solver=eikonax_solver, tensor_field=tensor_field
    )
    eval_point = np.array(eval_point)
    jacobian = []
    for i, _ in enumerate(eval_point):
        jacobian_column = stencil(eval_func, eval_point, step_width, i)
        jacobian.append(jacobian_column)
    jacobian = np.vstack(jacobian)
    return jacobian.T


# --------------------------------------------------------------------------------------------------
def compute_fd_hessian(
    func: Callable,
    stencil: Callable,
    eval_point: jtReal[npt.NDArray | jax.Array, "M"],
    step_width: float,
) -> None:
    """Implement finite difference Hessian computation.

    !!! failure "Not implemented yet"
    """
    raise NotImplementedError
