r"""Linear algebra utilities and sparse data structures for Eikonax.

This module provides specialized sparse data structures and utilities tailored to the Eikonax
solver's derivative computation workflow. Since the derivative operators are inherently sparse due
to the local nature of the Godunov update scheme, these structures efficiently store only the
non-zero entries along with their connectivity information.

The module defines three main sparse container classes:

1. [`EikonaxSparseMatrix`][eikonax.linalg.EikonaxSparseMatrix]: A COO-like (coordinate format)
   sparse matrix representation compatible with JAX arrays. Stores row indices, column indices,
   and values separately along with the matrix shape.

2. [`DerivatorSparseTensor`][eikonax.linalg.DerivatorSparseTensor]: Specialized tensor for storing
   partial derivatives $\mathbf{G}_M$ computed by the
   [`PartialDerivator`][eikonax.derivator.PartialDerivator]. Pairs derivative values with simplex
   adjacency information for each vertex.

3. [`TensorfieldSparseTensor`][eikonax.linalg.TensorfieldSparseTensor]: Container for the Jacobian
   $\frac{d\mathbf{M}}{d\mathbf{m}}$ computed by the
   [`TensorField`][eikonax.tensorfield.TensorField]. Stores derivative values together with
   parameter indices that indicate which global parameters each simplex depends on.

The key operation in this module is the tensor contraction
[`contract_derivative_tensors`][eikonax.linalg.contract_derivative_tensors], which efficiently
combines the derivatives from the Eikonax solver and the tensor field to compute the total
parametric derivative $\mathbf{G}_m = \mathbf{G}_M \frac{d\mathbf{M}}{d\mathbf{m}}$ required
for gradient computation.

Classes:
    EikonaxSparseMatrix: COO-style sparse matrix for JAX-compatible derivative storage
    DerivatorSparseTensor: Sparse tensor for partial derivatives w.r.t. metric tensors
    TensorfieldSparseTensor: Sparse tensor for tensor field Jacobian

Functions:
    convert_to_scipy_sparse: Convert EikonaxSparseMatrix to SciPy sparse array
    contract_derivative_tensors: Contract DerivatorSparseTensor with TensorfieldSparseTensor
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jaxtyping import Int as jtInt
from jaxtyping import Real as jtReal


# ==================================================================================================
class EikonaxSparseMatrix(eqx.Module):
    """COO-style sparse matrix representation compatible with JAX.

    This class stores a sparse matrix in coordinate (COO) format using JAX arrays. Unlike standard
    SciPy sparse matrices, this representation is JAX-compatible and can be used in JIT-compiled
    functions and automatic differentiation. The matrix is defined by three parallel arrays storing
    row indices, column indices, and corresponding values, along with the matrix shape.

    Attributes:
        row_inds (jax.Array): Row indices of non-zero entries with shape (num_entries,).
        col_inds (jax.Array): Column indices of non-zero entries with shape (num_entries,).
        values (jax.Array): Values of non-zero entries with shape (num_entries,).
        shape (tuple[int, int]): Shape of the matrix (num_rows, num_cols).
    """

    row_inds: jtInt[jax.Array, "num_entries"]
    col_inds: jtInt[jax.Array, "num_entries"]
    values: jtReal[jax.Array, "num_entries"]
    shape: tuple[int, int]


class DerivatorSparseTensor(eqx.Module):
    r"""Sparse tensor for partial derivatives from the PartialDerivator.

    This container stores the partial derivatives $\mathbf{G}_M$ of the Eikonax update operator
    with respect to the metric tensor field. For each vertex, it stores derivative contributions
    from all adjacent simplices in a dense local format, along with indices identifying which
    simplices are adjacent. This structure efficiently represents the sparse global tensor while
    maintaining JAX compatibility.

    The derivative values capture how changes in the metric tensor of each adjacent simplex affect
    the update value at each vertex. The adjacency data maps these local contributions to the
    global simplex numbering.

    Attributes:
        derivative_values (jax.Array): Partial derivative tensors with shape
            (num_vertices, max_num_neighbors, dim, dim). For each vertex and adjacent simplex,
            stores the (dim x dim) derivative of the update w.r.t. that simplex's metric tensor.
        adjacent_simplex_data (jax.Array): Global simplex indices with shape
            (num_vertices, max_num_neighbors). Maps local neighbor index to global simplex index.
            Entries of -1 indicate padding for vertices with fewer than max_num_neighbors adjacent
            simplices.
    """

    derivative_values: jtReal[jax.Array, "num_vertices max_num_neighbors dim dim"]
    adjacent_simplex_data: jtInt[jax.Array, "num_vertices max_num_neighbors"]


class TensorfieldSparseTensor(eqx.Module):
    r"""Sparse tensor for the tensor field Jacobian.

    This container stores the Jacobian $\frac{d\mathbf{M}}{d\mathbf{m}}$ computed by the
    [`TensorField`][eikonax.tensorfield.TensorField] component. For each simplex, it stores
    how the metric tensor depends on the global parameters, along with indices indicating which
    global parameters affect each simplex.

    Since most tensor field parameterizations are local (each simplex depends on only a small
    subset of global parameters), this sparse representation is much more memory-efficient than
    storing the full dense Jacobian tensor.

    Attributes:
        derivative_values (jax.Array): Jacobian values with shape
            (num_simplices, dim, dim, num_parameters_mapped). For each simplex, stores how the
            (dim x dim) metric tensor changes with respect to the relevant parameters.
        parameter_inds (jax.Array): Global parameter indices with shape
            (num_simplices, num_parameters_mapped). Maps local parameter index to global parameter
            index for each simplex.
        num_parameters_global (int): Total number of global parameters in the full parameter
            vector. Used to determine output matrix dimensions in contractions.
    """

    derivative_values: jtReal[jax.Array, "num_simplices dim dim num_parameters_mapped"]
    parameter_inds: jtInt[jax.Array, "num_simplices num_parameters_mapped"]
    num_parameters_global: int


# ==================================================================================================
def convert_to_scipy_sparse(eikonax_sparse_matrix: EikonaxSparseMatrix) -> sp.coo_array:
    """Convert EikonaxSparseMatrix to SciPy sparse COO array.

    This function transforms the JAX-compatible
    [`EikonaxSparseMatrix`][eikonax.linalg.EikonaxSparseMatrix] format into a standard SciPy
    `coo_array`. The conversion involves:

    1. Extracting row indices, column indices, and values as NumPy arrays
    2. Filtering out zero entries to reduce memory footprint
    3. Constructing a SciPy COO array and summing duplicate entries

    This conversion is necessary when interfacing with SciPy's sparse linear algebra routines,
    such as in the [`DerivativeSolver`][eikonax.derivator.DerivativeSolver] which requires CSC
    format for the triangular solve.

    Args:
        eikonax_sparse_matrix (EikonaxSparseMatrix): Sparse matrix in JAX-compatible format.

    Returns:
        scipy.sparse.coo_array: SciPy sparse array in COO format with zeros removed and
            duplicate entries summed.
    """
    row_inds = np.array(eikonax_sparse_matrix.row_inds, dtype=np.int32)
    colinds = np.array(eikonax_sparse_matrix.col_inds, dtype=np.int32)
    values = np.array(eikonax_sparse_matrix.values, dtype=np.float32)

    nonzero_mask = np.nonzero(values)
    row_inds = row_inds[nonzero_mask]
    colinds = colinds[nonzero_mask]
    values = values[nonzero_mask]

    coo_matrix = sp.coo_array((values, (row_inds, colinds)), shape=eikonax_sparse_matrix.shape)
    coo_matrix.sum_duplicates()
    return coo_matrix


# --------------------------------------------------------------------------------------------------
@eqx.filter_jit
def contract_derivative_tensors(
    derivative_sparse_tensor: DerivatorSparseTensor,
    tensorfield_sparse_tensor: TensorfieldSparseTensor,
) -> EikonaxSparseMatrix:
    r"""Contract derivative tensors to compute total parametric derivative.

    This function performs the key operation in Eikonax's gradient computation pipeline: combining
    the partial derivatives from the solver ($\mathbf{G}_M$) with the tensor field Jacobian
    ($\frac{d\mathbf{M}}{d\mathbf{m}}$) to obtain the total parametric derivative
    $\mathbf{G}_m = \mathbf{G}_M \frac{d\mathbf{M}}{d\mathbf{m}}$.

    The contraction is performed efficiently by exploiting the sparse structure of both inputs:

    1. For each vertex, iterate over adjacent simplices
    2. Extract the (dim x dim) derivative matrix for that vertex-simplex pair from
       `derivative_sparse_tensor`
    3. Contract (via Einstein summation) with the corresponding (dim x dim x num_params) Jacobian
       tensor from `tensorfield_sparse_tensor`
    4. Map the result to the appropriate global parameter indices
    5. Assemble all contributions into a sparse matrix of shape (num_vertices, num_parameters)

    The operation is vectorized over all vertices using JAX's `vmap` and JIT-compiled for
    performance.

    Args:
        derivative_sparse_tensor (DerivatorSparseTensor): Partial derivatives $\mathbf{G}_M$
            from the [`PartialDerivator`][eikonax.derivator.PartialDerivator].
        tensorfield_sparse_tensor (TensorfieldSparseTensor): Tensor field Jacobian
            $\frac{d\mathbf{M}}{d\mathbf{m}}$ from the
            [`TensorField`][eikonax.tensorfield.TensorField].

    Returns:
        EikonaxSparseMatrix: Total parametric derivative $\mathbf{G}_m$ with shape
            (num_vertices, num_parameters_global). This can be transposed and multiplied with
            the adjoint vector to compute parametric gradients.
    """
    global_contraction_function = jax.vmap(_contract_vertex_tensors, in_axes=(0, 0, None, None))
    values, col_inds = global_contraction_function(
        derivative_sparse_tensor.derivative_values,
        derivative_sparse_tensor.adjacent_simplex_data,
        tensorfield_sparse_tensor.derivative_values,
        tensorfield_sparse_tensor.parameter_inds,
    )
    num_vertices = derivative_sparse_tensor.derivative_values.shape[0]
    num_parameters_per_simplex = tensorfield_sparse_tensor.parameter_inds.shape[1]
    max_num_adjacent_simplices = derivative_sparse_tensor.adjacent_simplex_data.shape[1]
    row_inds = jnp.repeat(
        jnp.arange(num_vertices), max_num_adjacent_simplices * num_parameters_per_simplex
    )
    eikonax_sparse_matrix = EikonaxSparseMatrix(
        row_inds=row_inds,
        col_inds=col_inds.flatten(),
        values=values.flatten(),
        shape=(num_vertices, tensorfield_sparse_tensor.num_parameters_global),
    )
    return eikonax_sparse_matrix


# --------------------------------------------------------------------------------------------------
def _contract_vertex_tensors(
    derivator_tensor: jtReal[jax.Array, "max_num_neighbors dim dim"],
    adjacent_simplex_data: jtInt[jax.Array, "max_num_neighbors"],
    tensorfield_data: jtReal[jax.Array, "num_simplices dim dim num_parameters_mapped"],
    parameter_inds: jtInt[jax.Array, "num_simplices num_parameters_mapped"],
) -> tuple[
    jtReal[jax.Array, "max_num_neighbors num_parameters_mapped"],
    jtInt[jax.Array, "max_num_neighbors num_parameters_mapped"],
]:
    """Contract derivatives for a single vertex with tensor field Jacobian.

    This helper function performs the tensor contraction for a single vertex. For each adjacent
    simplex, it:

    1. Extracts the (dim x dim) derivative matrix from the derivator tensor
    2. Extracts the (dim x dim x num_params) Jacobian tensor for that simplex
    3. Contracts them via Einstein summation: result[k] = sum_ij derivator[i,j] * jacobian[i,j,k]
    4. Maps the result to the global parameter indices for that simplex

    Invalid simplices (indicated by simplex_ind == -1) are handled by filtering: their
    contributions are set to zero and their column indices set to -1.

    This function is called in a vectorized manner by
    [`contract_derivative_tensors`][eikonax.linalg.contract_derivative_tensors] via `jax.vmap`.

    Args:
        derivator_tensor (jax.Array): Partial derivatives for one vertex with shape
            (max_num_neighbors, dim, dim).
        adjacent_simplex_data (jax.Array): Global simplex indices for one vertex with shape
            (max_num_neighbors,). Entries of -1 indicate padding.
        tensorfield_data (jax.Array): Global tensor field Jacobian with shape
            (num_simplices, dim, dim, num_parameters_mapped).
        parameter_inds (jax.Array): Global parameter indices for all simplices with shape
            (num_simplices, num_parameters_mapped).

    Returns:
        tuple[jax.Array, jax.Array]: Contracted derivative values with shape
            (max_num_neighbors, num_parameters_mapped) and corresponding column indices with
            the same shape. Entries corresponding to invalid simplices are zeroed/set to -1.
    """
    partial_derivative_values = jnp.zeros(
        (adjacent_simplex_data.shape[0], parameter_inds.shape[1]), dtype=jnp.float32
    )
    partial_derivative_cols = jnp.zeros(
        (adjacent_simplex_data.shape[0], parameter_inds.shape[1]), dtype=jnp.int32
    )
    for i, simplex_ind in enumerate(adjacent_simplex_data):
        derivator_matrix = derivator_tensor[i]
        tensorfield_tensor = tensorfield_data[simplex_ind]
        values = jnp.einsum("ij, ijk -> k", derivator_matrix, tensorfield_tensor).flatten()
        partial_derivative_values = partial_derivative_values.at[i, :].set(values)
        partial_derivative_cols = partial_derivative_cols.at[i, :].set(parameter_inds[simplex_ind])

    filtered_values = jnp.where(
        adjacent_simplex_data[:, None] != -1, partial_derivative_values, 0.0
    )
    filtered_cols = jnp.where(adjacent_simplex_data[:, None] != -1, partial_derivative_cols, -1)

    return filtered_values, filtered_cols
