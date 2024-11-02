import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp

from . import corefunctions


@chex.dataclass
class PartialDerivatorData:
    softmin_order: int
    softminmax_order: int
    softminmax_cutoff: int


# ==================================================================================================
class PartialDerivator(eqx.Module):
    _vertices: jnp.ndarray
    _adjacency_data: jnp.ndarray
    _initial_sites: corefunctions.InitialSites
    _softmin_order: int
    _softminmax_order: int
    _softminmax_cutoff: int

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        mesh_data: corefunctions.MeshData,
        derivator_data: PartialDerivatorData,
        initial_sites: corefunctions.InitialSites,
    ):
        self._vertices = mesh_data.vertices
        self._adjacency_data = mesh_data.adjacency_data
        self._initial_sites = initial_sites
        self._softmin_order = derivator_data.softmin_order
        self._softminmax_order = derivator_data.softminmax_order
        self._softminmax_cutoff = derivator_data.softminmax_cutoff

    # ----------------------------------------------------------------------------------------------
    def compute_partial_derivatives(
        self,
        solution_vector: jnp.ndarray,
        tensor_field: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        partial_derivative_solution, partial_derivative_parameter = (
            self._compute_global_partial_derivatives(
                solution_vector,
                tensor_field,
            )
        )
        sparse_partial_solution = self._compress_partial_derivative_solution(
            partial_derivative_solution
        )
        sparse_partial_parameter = self._compress_partial_derivative_parameter(
            partial_derivative_parameter
        )

        return sparse_partial_solution, sparse_partial_parameter

    # ----------------------------------------------------------------------------------------------
    def _compress_partial_derivative_solution(
        self,
        partial_derivative_solution: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        current_inds = self._adjacency_data[:, 0, 0]
        adjacent_inds = self._adjacency_data[:, :, 1:3]

        nonzero_mask = jnp.nonzero(partial_derivative_solution)
        rows_compressed = current_inds[nonzero_mask[0]]
        columns_compressed = adjacent_inds[nonzero_mask]
        values_compressed = partial_derivative_solution[nonzero_mask]

        initial_site_mask = jnp.where(rows_compressed != self._initial_sites.inds)
        rows_compressed = rows_compressed[initial_site_mask]
        columns_compressed = columns_compressed[initial_site_mask]
        values_compressed = values_compressed[initial_site_mask]

        rows_compressed = jnp.concatenate((self._initial_sites.inds, rows_compressed))
        columns_compressed = jnp.concatenate((self._initial_sites.inds, columns_compressed))
        values_compressed = jnp.concatenate(
            (jnp.zeros(self._initial_sites.inds.shape), values_compressed)
        )

        return rows_compressed, columns_compressed, values_compressed

    # ----------------------------------------------------------------------------------------------
    def _compress_partial_derivative_parameter(
        self, partial_derivative_parameter: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        vertex_inds = self._adjacency_data[:, 0, 0]
        simplex_inds = self._adjacency_data[:, :, 3]
        tensor_dim = partial_derivative_parameter.shape[2]

        values_reduced = jnp.sum(jnp.abs(partial_derivative_parameter), axis=(2, 3))
        nonzero_mask = jnp.nonzero(values_reduced)
        rows_compressed = vertex_inds[nonzero_mask[0]]
        simplices_compressed = simplex_inds[nonzero_mask]
        values_compressed = partial_derivative_parameter[nonzero_mask]

        initial_site_mask = jnp.where(rows_compressed != self._initial_sites.inds)
        rows_compressed = rows_compressed[initial_site_mask]
        columns_compressed = simplices_compressed[initial_site_mask]
        values_compressed = values_compressed[initial_site_mask]

        rows_compressed = jnp.concatenate((self._initial_sites.inds, rows_compressed))
        columns_compressed = jnp.concatenate((self._initial_sites.inds, columns_compressed))
        values_compressed = jnp.concatenate(
            (jnp.zeros((self._initial_sites.inds.size, tensor_dim, tensor_dim)), values_compressed)
        )

        return rows_compressed, simplices_compressed, values_compressed

    # ----------------------------------------------------------------------------------------------
    @eqx.filter_jit
    def _compute_global_partial_derivatives(
        self,
        solution_vector: jnp.ndarray,
        tensor_field: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        assert solution_vector.shape[0] == self._vertices.shape[0], (
            f"Solution vector needs to have shape {self._vertices.shape[0]}, "
            f"but has shape {solution_vector.shape}"
        )
        assert self._adjacency_data.shape[0] == self._vertices.shape[0], (
            f"Adjacent simplex indix array needs to have shape {self._vertices.shape[0]}, "
            f"but has shape {self._adjacency_data.shape[0]}"
        )
        global_partial_derivative_function = jax.vmap(
            self._compute_vertex_partial_derivatives,
            in_axes=(None, None, 0),
        )
        partial_derivative_solution, partial_derivative_parameter = (
            global_partial_derivative_function(solution_vector, tensor_field, self._adjacency_data)
        )

        max_num_adjacent_simplices = self._adjacency_data.shape[1]
        tensor_dim = tensor_field.shape[1]
        assert partial_derivative_solution.shape == (
            solution_vector.shape[0],
            max_num_adjacent_simplices,
            2,
        )
        assert partial_derivative_parameter.shape == (
            solution_vector.shape[0],
            max_num_adjacent_simplices,
            tensor_dim,
            tensor_dim,
        )
        return partial_derivative_solution, partial_derivative_parameter

    # ----------------------------------------------------------------------------------------------
    def _compute_vertex_partial_derivatives(
        self,
        solution_vector: jnp.ndarray,
        tensor_field: jnp.ndarray,
        adjacency_data: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        max_num_adjacent_simplices = adjacency_data.shape[0]
        tensor_dim = tensor_field.shape[1]
        assert adjacency_data.shape == (max_num_adjacent_simplices, 4), (
            f"node-level adjacency data needs to have shape ({max_num_adjacent_simplices}, 4), "
            f"but has shape {adjacency_data.shape}"
        )

        vertex_update_candidates = corefunctions.compute_vertex_update_candidates(
            solution_vector,
            tensor_field,
            adjacency_data,
            self._vertices,
            self._softminmax_order,
            self._softminmax_cutoff,
        )
        grad_update_solution_candidates, grad_update_parameter_candidates = (
            self._compute_vertex_partial_derivative_candidates(
                solution_vector, tensor_field, adjacency_data
            )
        )
        min_value, grad_update_solution_candidates, grad_update_parameter_candidates = (
            self._filter_candidates(
                vertex_update_candidates,
                grad_update_solution_candidates,
                grad_update_parameter_candidates,
            )
        )
        softmin_grad = corefunctions.grad_softmin(
            vertex_update_candidates.flatten(), min_value, self._softmin_order
        ).reshape(vertex_update_candidates.shape)

        grad_update_solution = jnp.zeros((max_num_adjacent_simplices, 2))
        grad_update_parameter = jnp.zeros((max_num_adjacent_simplices, tensor_dim, tensor_dim))
        for i in range(max_num_adjacent_simplices):
            grad_update_solution = grad_update_solution.at[i, :].set(
                jnp.tensordot(softmin_grad[i, :], grad_update_solution_candidates[i, ...], axes=1)
            )
            grad_update_parameter = grad_update_parameter.at[i, ...].set(
                jnp.tensordot(softmin_grad[i, :], grad_update_parameter_candidates[i, ...], axes=1)
            )

        return grad_update_solution, grad_update_parameter

    # ----------------------------------------------------------------------------------------------
    def _compute_vertex_partial_derivative_candidates(
        self,
        solution_vector: jnp.ndarray,
        tensor_field: jnp.ndarray,
        adjacency_data: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        max_num_adjacent_simplices = adjacency_data.shape[0]
        tensor_dim = tensor_field.shape[1]
        grad_update_solution_candidates = jnp.zeros((max_num_adjacent_simplices, 4, 2))
        grad_update_parameter_candidates = jnp.zeros(
            (max_num_adjacent_simplices, 4, tensor_dim, tensor_dim)
        )

        for i, indices in enumerate(adjacency_data):
            partial_solution, partial_parameter = (
                self._compute_partial_derivative_candidates_from_adjacent_simplex(
                    solution_vector, tensor_field, indices
                )
            )
            grad_update_solution_candidates = grad_update_solution_candidates.at[i, ...].set(
                partial_solution
            )
            grad_update_parameter_candidates = grad_update_parameter_candidates.at[i, ...].set(
                partial_parameter
            )

        return grad_update_solution_candidates, grad_update_parameter_candidates

    # ----------------------------------------------------------------------------------------------
    def _compute_partial_derivative_candidates_from_adjacent_simplex(
        self,
        solution_vector: jnp.ndarray,
        tensor_field: jnp.ndarray,
        adjacency_data: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        assert (
            len(adjacency_data) == 4
        ), f"Indices need to have length 4, but have length {len(adjacency_data)}"
        i, j, k, s = adjacency_data
        tensor_dim = tensor_field.shape[1]
        solution_values = jnp.array((solution_vector[j], solution_vector[k]))
        edges = corefunctions.compute_edges(i, j, k, self._vertices)
        parameter_tensor = tensor_field[s]
        lambda_array = corefunctions.compute_optimal_update_parameters(
            solution_values, parameter_tensor, edges, self._softminmax_order, self._softminmax_cutoff
        )
        lambda_partial_solution, lambda_partial_parameter = self._compute_lambda_grad(
            solution_values, parameter_tensor, edges
        )
        lambda_partial_solution = jnp.concatenate((jnp.zeros((2, 2)), lambda_partial_solution))
        lambda_partial_parameter = jnp.concatenate(
            (jnp.zeros((2, tensor_dim, tensor_dim)), lambda_partial_parameter)
        )
        grad_update_solution = jnp.zeros((4, 2))
        grad_update_parameter = jnp.zeros((4, tensor_dim, tensor_dim))

        for i in range(4):
            update_partial_lambda = corefunctions.grad_update_lambda(
                solution_values, parameter_tensor, lambda_array[i], edges
            )
            update_partial_solution = corefunctions.grad_update_solution(
                solution_values, parameter_tensor, lambda_array[i], edges
            )
            update_partial_parameter = corefunctions.grad_update_parameter(
                solution_values, parameter_tensor, lambda_array[i], edges
            )
            grad_update_solution = grad_update_solution.at[i, :].set(
                update_partial_lambda * lambda_partial_solution[i, :] + update_partial_solution
            )
            grad_update_parameter = grad_update_parameter.at[i, ...].set(
                update_partial_lambda * lambda_partial_parameter[i, ...] + update_partial_parameter
            )
        return grad_update_solution, grad_update_parameter

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _filter_candidates(
        vertex_update_candidates: jnp.ndarray,
        grad_update_solution_candidates: jnp.ndarray,
        grad_update_parameter_candidates: jnp.ndarray,
    ):
        min_value = jnp.min(vertex_update_candidates)
        min_candidate_mask = jnp.where(vertex_update_candidates == min_value, 1, 0)
        vertex_update_candidates = jnp.where(
            min_candidate_mask == 1, vertex_update_candidates, jnp.inf
        )
        grad_update_solution_candidates = jnp.where(
            min_candidate_mask[..., None] == 1, grad_update_solution_candidates, 0
        )
        grad_update_parameter_candidates = jnp.where(
            min_candidate_mask[..., None, None] == 1, grad_update_parameter_candidates, 0
        )

        return min_value, grad_update_solution_candidates, grad_update_parameter_candidates

    # ----------------------------------------------------------------------------------------------
    def _compute_lambda_grad(
        self,
        solution_values: jnp.ndarray,
        parameter_tensor: jnp.ndarray,
        edges: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        lambda_partial_solution = corefunctions.jac_lambda_solution(
            solution_values,
            parameter_tensor,
            edges,
            self._softminmax_order,
            self._softminmax_cutoff,
        )
        lambda_partial_parameter = corefunctions.jac_lambda_parameter(
            solution_values,
            parameter_tensor,
            edges,
            self._softminmax_order,
            self._softminmax_cutoff,
        )

        return lambda_partial_solution, lambda_partial_parameter


# ==================================================================================================
class DerivativeSolver:
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        solution: jnp.ndarray,
        sparse_partial_update_solution: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> None:
        num_points = solution.size
        self._sparse_permutation_matrix = self._assemble_permutation_matrix(solution)
        self._sparse_system_matrix = self._assemble_system_matrix(
            sparse_partial_update_solution, num_points
        )

    # ----------------------------------------------------------------------------------------------
    def solve(self, right_hand_side: np.ndarray | jnp.ndarray) -> np.ndarray:
        permutated_right_hand_side = self._sparse_permutation_matrix @ right_hand_side
        permutated_solution = sp.sparse.linalg.spsolve_triangular(
            self._sparse_system_matrix, permutated_right_hand_side, lower=False, unit_diagonal=True
        )
        solution = self._sparse_permutation_matrix.T @ permutated_solution

        return solution

    # ----------------------------------------------------------------------------------------------
    def _assemble_permutation_matrix(self, solution: jnp.ndarray) -> sp.sparse.csc_matrix:
        num_points = solution.size
        permutation_row_inds = jnp.arange(solution.size)
        permutation_col_inds = jnp.argsort(solution)
        permutation_values = jnp.ones(solution.size)
        sparse_permutation_matrix = sp.sparse.csc_matrix(
            (permutation_values, (permutation_row_inds, permutation_col_inds)),
            shape=(num_points, num_points),
        )

        return sparse_permutation_matrix

    # ----------------------------------------------------------------------------------------------
    def _assemble_system_matrix(
        self,
        sparse_partial_update_solution: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        num_points: int,
    ) -> sp.sparse.csc_matrix:
        rows_compressed, columns_compressed, values_compressed = sparse_partial_update_solution
        sparse_partial_matrix = sp.sparse.csc_matrix(
            (values_compressed, (rows_compressed, columns_compressed)),
            shape=(num_points, num_points),
        )
        sparse_identity_matrix = sp.sparse.identity(num_points, format="csc")
        sparse_system_matrix = sparse_identity_matrix - sparse_partial_matrix
        sparse_system_matrix = (
            self._sparse_permutation_matrix
            @ sparse_system_matrix
            @ self._sparse_permutation_matrix.T
        )

        return sparse_system_matrix.T
