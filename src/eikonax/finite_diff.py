from collections.abc import Callable

import jax
import numpy as np
import numpy.typing as npt
from jaxtyping import Real as jtReal


# ==================================================================================================
def finite_diff_1_forward(
    func: Callable, eval_point: jtReal[npt.NDArray, "M"], step_width: float, index: int
):
    fwd_eval_point = eval_point.copy(eval_point)
    fwd_eval_point[index] += step_width
    unperturbed_eval = func(eval_point)
    fwd_perturbed_eval = func(eval_point)
    finite_diff = (fwd_perturbed_eval - unperturbed_eval) / step_width
    return finite_diff


# --------------------------------------------------------------------------------------------------
def finite_diff_1_backward(
    func: Callable, eval_point: jtReal[npt.NDArray, "M"], step_width: float, index: int
):
    bwd_eval_point = eval_point.copy(eval_point)
    bwd_eval_point[index] -= step_width
    unperturbed_eval = func(eval_point)
    bwd_perturbed_eval = func(bwd_eval_point)
    finite_diff = (unperturbed_eval - bwd_perturbed_eval) / step_width
    return finite_diff


# --------------------------------------------------------------------------------------------------
def finite_diff_1_central(
    func: Callable, eval_point: jtReal[npt.NDArray, "M"], step_width: float, index: int
):
    fwd_eval_point = eval_point.copy(eval_point)
    fwd_eval_point[index] += step_width
    bwd_eval_point = eval_point.copy(eval_point)
    bwd_eval_point[index] -= step_width
    fwd_perturbed_eval = func(fwd_eval_point)
    bwd_perturbed_eval = func(bwd_eval_point)
    finite_diff = (fwd_perturbed_eval - bwd_perturbed_eval) / (2 * step_width)
    return finite_diff


# --------------------------------------------------------------------------------------------------
def finite_diff_2(
    func: Callable,
    eval_point: jtReal[npt.NDArray, "M"],
    step_width: float,
    index_1: int,
    index_2: int,
):
    if index_1 == index_2:
        fwd_eval_point = eval_point.copy(eval_point)
        fwd_eval_point[index_1] += step_width
        bwd_eval_point = eval_point.copy(eval_point)
        bwd_eval_point[index_1] -= step_width
        fwd_eval = func(fwd_eval_point)
        bwd_eval = func(bwd_eval_point)
        unperturbed_eval = func(eval_point)
        finite_diff = (fwd_eval - 2 * unperturbed_eval + bwd_eval) / (step_width**2)
    else:
        fwd_fwd_eval_point = eval_point.copy(eval_point)
        fwd_fwd_eval_point[index_1] += step_width
        fwd_fwd_eval_point[index_2] += step_width
        bwd_bwd_eval_point = eval_point.copy(eval_point)
        bwd_bwd_eval_point[index_1] -= step_width
        bwd_bwd_eval_point[index_2] -= step_width
        fwd_bwd_eval_point = eval_point.copy(eval_point)
        fwd_bwd_eval_point[index_1] += step_width
        fwd_bwd_eval_point[index_2] -= step_width
        bwd_fwd_eval_point = eval_point.copy(eval_point)
        bwd_fwd_eval_point[index_1] -= step_width
        bwd_fwd_eval_point[index_2] += step_width
        fwd_fwd_eval = func(fwd_fwd_eval_point)
        fwd_bwd_eval = func(fwd_bwd_eval_point)
        bwd_fwd_eval = func(bwd_fwd_eval_point)
        bwd_bwd_eval = func(bwd_bwd_eval_point)
        finite_diff = (fwd_fwd_eval - fwd_bwd_eval - bwd_fwd_eval + bwd_bwd_eval) / (
            4 * step_width**2
        )
    return finite_diff


# ==================================================================================================
def compute_fd_jacobian(
    func: Callable,
    stencil: Callable,
    eval_point: jtReal[npt.NDArray | jax.Array, "M"],
    step_width: float,
):
    eval_point = np.array(eval_point)
    jacobian = []
    for i, _ in enumerate(eval_point):
        jacobian_column = stencil(func, eval_point, step_width, i)
        jacobian.append(jacobian_column)
    jacobian = np.vstack(jacobian)
    return jacobian


# --------------------------------------------------------------------------------------------------
def compute_fd_hessian(
    func: Callable,
    stencil: Callable,
    eval_point: jtReal[npt.NDArray | jax.Array, "M"],
    step_width: float,
):
    eval_point = np.array(eval_point)
    hessian = []
    for i, _ in enumerate(eval_point):
        subhessian = []
        for j, _ in enumerate(eval_point):
            hessian_entry = stencil(func, eval_point, step_width, i, j)
            subhessian.append(hessian_entry)
        subhessian = np.vstack(subhessian)
        hessian.append(subhessian)
    hessian = np.vstack(hessian)
    return hessian
