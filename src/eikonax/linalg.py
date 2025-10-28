import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jaxtyping import Int as jtInt
from jaxtyping import Real as jtReal


# ==================================================================================================
class EikonaxSparseMatrix(eqx.Module):
    row_inds: jtInt[jax.Array, "num_entries"]
    col_inds: jtInt[jax.Array, "num_entries"]
    values: jtReal[jax.Array, "num_entries"]
    shape: tuple[int, int]


class DerivatorSparseTensor(eqx.Module):
    derivative_values: jtReal[jax.Array, "num_vertices max_num_neighbors dim dim"]
    adjacent_simplex_data: jtInt[jax.Array, "num_vertices max_num_neighbors"]


class TensorfieldSparseTensor(eqx.Module):
    derivative_values: jtReal[jax.Array, "num_simplices dim dim num_parameters_mapped"]
    parameter_inds: jtInt[jax.Array, "num_simplices num_parameters_mapped"]
    num_parameters_global: int


# ==================================================================================================
def convert_to_scipy_sparse(eikonax_sparse_matrix: EikonaxSparseMatrix) -> sp.coo_array:
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


def _contract_vertex_tensors(
    derivator_tensor: jtReal[jax.Array, "max_num_neighbors dim dim"],
    adjacent_simplex_data: jtInt[jax.Array, "max_num_neighbors"],
    tensorfield_data: jtReal[jax.Array, "num_simplices dim dim num_parameters_mapped"],
    parameter_inds: jtInt[jax.Array, "num_simplices num_parameters_mapped"],
):
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
