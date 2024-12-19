"""_summary_."""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
from jaxtyping import Array, Float, Int

from . import corefunctions


@chex.dataclass
class PartialDerivatorData:
    """Settings for initialization of partial derivator.

    Attributes:
        softmin_order: Order of the softmin function applied to update candidates with
            identical, minimal arrival times.
        softminmax_order: Order of the the soft minmax function for differentiable transformation
            of the update parameters
        softminmax_cutoff: Cut-off in for minmax transformation, beyond which zero sensitivity
            is assumed.
    """

    softmin_order: int
    softminmax_order: int
    softminmax_cutoff: int


# ==================================================================================================
class PartialDerivator(eqx.Module):
    """Component for computing partial derivatives of the Godunov Update operator.

    Methods:
        compute_partial_derivatives: Compute the partial derivatives of the Godunov update operator
            with respect to the solution vector and the parameter tensor field, given a state for
            both variables
    """

    # Equinox modules are data classes, so we need to specify attributes on class level
    _vertices: Float[Array, "num_vertices dim"]
    _adjacency_data: Int[Array, "num_vertices max_num_adjacent_simplices 4"]
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
    ) -> None:
        """Constructor for the partial derivator object.

        Args:
            mesh_data (corefunctions.MeshData): Mesh data object also utilized for the Eikonax
                solver, contains adjacency data for every vertex.
            derivator_data (PartialDerivatorData): Settings for initialization of the derivator.
            initial_sites (corefunctions.InitialSites): Locations and values at source points
        """
        self._vertices = mesh_data.vertices
        self._adjacency_data = mesh_data.adjacency_data
        self._initial_sites = initial_sites
        self._softmin_order = derivator_data.softmin_order
        self._softminmax_order = derivator_data.softminmax_order
        self._softminmax_cutoff = derivator_data.softminmax_cutoff

    # ----------------------------------------------------------------------------------------------
    def compute_partial_derivatives(
        self,
        solution_vector: Float[Array | np.ndarray, "num_vertices"],
        tensor_field: Float[Array | np.ndarray, "num_simplices dim dim"],
    ) -> tuple[
        tuple[
            Int[Array, "num_sol_values"],
            Int[Array, "num_sol_values"],
            Float[Array, "num_sol_values"],
        ],
        tuple[
            Int[Array, "num_param_values"],
            Int[Array, "num_param_values"],
            Float[Array, "num_param_values dim dim"],
        ],
    ]:
        """Compute the partial derivatives of the Godunov update operator.

        This method provided the main interface for computing the partial derivatives of the global
        Eikonax update operator with respect to the solution vector and the parameter tensor field.
        The updates are computed locally for each vertex, such that the resulting data structures
        are sparse. Subsequently, further zero entries are removed to reduce the memory footprint.
        The derivatives computed in this component can be utilized to compute the total parametric
        derivative via a fix point equation, given that the provided solution vector is that
        fix point.
        The computation of partial derivatives is possible with a single pass over the mesh, since
        the solution of the Eikonax equation, and therefore causality within the Godunov update
        scheme, is known.

        Args:
            solution_vector (jnp.ndarray): Current solution
            tensor_field (jnp.ndarray): Parameter field

        Returns:
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Partial derivatives with respect to
                the solution vector and the parameter tensor field. Both quantities are given as
                arrays over all local contributions, making them sparse in the global context.
        """
        solution_vector = jnp.array(solution_vector, dtype=jnp.float32)
        tensor_field = jnp.array(tensor_field, dtype=jnp.float32)
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
        partial_derivative_solution: Float[Array, "num_vertices max_num_adjacent_simplices 2"],
    ) -> tuple[
        Int[Array, "num_sol_values"],
        Int[Array, "num_sol_values"],
        Float[Array, "num_sol_values"],
    ]:
        """Compress the partial derivative data with respect to the solution vector.

        Compression consists of two steps:
        1. Remove zero entries in the sensitivity vector
        2. Set the sensitivity vector to zero at the initial sites, but keep them for later
           computations.

        Args:
            partial_derivative_solution (jnp.ndarray): Raw data from partial derivative computation,
                with shape (N, num_adjacent_simplices, 2), N depends on the number of identical
                update paths for the vertices in the mesh.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Compressed data, represented as rows,
            columns and values for initialization in sparse matrix. Shape depends on number
                of non-zero entries
        """
        current_inds = self._adjacency_data[:, 0, 0]
        adjacent_inds = self._adjacency_data[:, :, 1:3]

        nonzero_mask = jnp.nonzero(partial_derivative_solution)
        rows_compressed = current_inds[nonzero_mask[0]]
        columns_compressed = adjacent_inds[nonzero_mask]
        values_compressed = partial_derivative_solution[nonzero_mask]

        initial_site_mask = jnp.where(rows_compressed == self._initial_sites.inds)
        values_compressed = values_compressed.at[initial_site_mask].set(
            jnp.zeros(self._initial_sites.inds.shape)
        )
        return rows_compressed, columns_compressed, values_compressed

    # ----------------------------------------------------------------------------------------------
    def _compress_partial_derivative_parameter(
        self,
        partial_derivative_parameter: Float[
            Array, "num_vertices max_num_adjacent_simplices dim dim"
        ],
    ) -> tuple[
        Int[Array, "num_param_values"],
        Int[Array, "num_param_values"],
        Float[Array, "num_param_values dim dim"],
    ]:
        """Compress the partial derivative data with respect to the parameter field.

        Compression consists of two steps:
        1. Remove tensor components from the sensitivity data, if all entries are zero
        2. Set the sensitivity vector to zero at the initial sites, but keep them for later
           computations.

        Args:
            partial_derivative_parameter (jnp.ndarray): Raw data from partial derivative computation,
                with shape (N, num_adjacent_simplices, dim, dim), N depends on the number of
                identical update paths for the vertices in the mesh.

        Returns:
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Compressed data, represented as rows,
                columns and values to be further processes for sparse matrix assembly.
                Shape depends on number of non-zero entries
        """
        vertex_inds = self._adjacency_data[:, 0, 0]
        simplex_inds = self._adjacency_data[:, :, 3]
        tensor_dim = partial_derivative_parameter.shape[2]

        values_reduced = jnp.sum(jnp.abs(partial_derivative_parameter), axis=(2, 3))
        nonzero_mask = jnp.nonzero(values_reduced)
        rows_compressed = vertex_inds[nonzero_mask[0]]
        simplices_compressed = simplex_inds[nonzero_mask]
        values_compressed = partial_derivative_parameter[nonzero_mask]

        initial_site_mask = jnp.where(rows_compressed == self._initial_sites.inds)
        values_compressed = values_compressed.at[initial_site_mask].set(
            jnp.zeros((initial_site_mask[0].size, tensor_dim, tensor_dim))
        )

        return rows_compressed, simplices_compressed, values_compressed

    # ----------------------------------------------------------------------------------------------
    @eqx.filter_jit
    def _compute_global_partial_derivatives(
        self,
        solution_vector: Float[Array, "num_vertices"],
        tensor_field: Float[Array, "num_simplices dim dim"],
    ) -> tuple[
        Float[Array, "num_vertices max_num_adjacent_simplices 2"],
        Float[Array, "num_vertices max_num_adjacent_simplices dim dim"],
    ]:
        """Compute partial derivatives of the global update operator.

        The method is a jitted and vectorized call to the `_compute_vertex_partial_derivative`
        method.

        Args:
            solution_vector (jnp.ndarray): Global solution vector
            tensor_field (jnp.ndarray): Global parameter tensor field

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: Raw data for partial derivatives, with shape
                (N, num_adjacent_simplices, dim, dim), N depends on the number of
                identical update paths for the vertices in the mesh.
        """
        global_partial_derivative_function = jax.vmap(
            self._compute_vertex_partial_derivatives,
            in_axes=(None, None, 0),
        )
        partial_derivative_solution, partial_derivative_parameter = (
            global_partial_derivative_function(solution_vector, tensor_field, self._adjacency_data)
        )
        return partial_derivative_solution, partial_derivative_parameter

    # ----------------------------------------------------------------------------------------------
    def _compute_vertex_partial_derivatives(
        self,
        solution_vector: Float[Array, "num_vertices"],
        tensor_field: Float[Array, "num_simplices dim dim"],
        adjacency_data: Int[Array, "max_num_adjacent_simplices 4"],
    ) -> tuple[
        Float[Array, "max_num_adjacent_simplices 2"],
        Float[Array, "max_num_adjacent_simplices dim dim"],
    ]:
        """Compute partial derivatives for the update of a single vertex.

        The method computes candidates for all respective subterms through calls to further methods.
        These candidates are filtered for feyasibility by means of JAX filters.
        The sofmin function (and its gradient) is applied to the directions of all optimal
        updates to ensure differentiability, other contributions are discarded.
        Lasty, the evaluated contributions are combined according to the form of the
        "total differential" for the parrtial derivatives.

        Args:
            solution_vector (jnp.ndarray): Global solution vector
            tensor_field (jnp.ndarray): Global parameter tensor field
            adjacency_data (jnp.ndarray): Adjacency data for the vertex under consideration

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: Partial derivatives for the given vertex
        """
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
            True,
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
        solution_vector: Float[Array, "num_vertices"],
        tensor_field: Float[Array, "num_simplices dim dim"],
        adjacency_data: Int[Array, "max_num_adjacent_simplices 4"],
    ) -> tuple[
        Float[Array, "max_num_adjacent_simplices 4 2"],
        Float[Array, "max_num_adjacent_simplices 4 dim dim"],
    ]:
        """Compute partial derivatives corresponding to potential update candidates for a vertex.

        Update candidates and corresponding derivatives are computed for all adjacent simplices,
        and for all possible update parameters per simplex.

        Args:
            solution_vector (jnp.ndarray): Global solution vector
            tensor_field (jnp.ndarray): Global parameter field
            adjacency_data (jnp.ndarray): Adjacency data for the given vertex

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: Candidates for partial derivatives
        """
        max_num_adjacent_simplices = adjacency_data.shape[0]
        tensor_dim = tensor_field.shape[1]
        grad_update_solution_candidates = jnp.zeros(
            (max_num_adjacent_simplices, 4, 2), dtype=jnp.float32
        )
        grad_update_parameter_candidates = jnp.zeros(
            (max_num_adjacent_simplices, 4, tensor_dim, tensor_dim), dtype=jnp.float32
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
        solution_vector: Float[Array, "num_vertices"],
        tensor_field: Float[Array, "num_simplices dim dim"],
        adjacency_data: Int[Array, "4"],
    ) -> tuple[Float[Array, "4 2"], Float[Array, "4 dim dim"]]:
        """Compute partial derivatives for all update candidates within an adjacent simplex.

        The update candidates are evaluated according to the different candidates for the
        optimization parameters lambda. Contributions are combined to the form of the involved
        total differentials.

        Args:
            solution_vector (jnp.ndarray): Global solution vector
            tensor_field (jnp.ndarray): Flobal parameter field
            adjacency_data (jnp.ndarray): Adjacency data for the given vertex and simplex

        Returns:
            tuple[jnp.ndarray, jnp.ndarray]: Derivative candidate from the given simplex
        """
        i, j, k, s = adjacency_data
        tensor_dim = tensor_field.shape[1]
        solution_values = jnp.array((solution_vector[j], solution_vector[k]))
        edges = corefunctions.compute_edges(i, j, k, self._vertices)
        parameter_tensor = tensor_field[s]
        lambda_array = corefunctions.compute_optimal_update_parameters_soft(
            solution_values,
            parameter_tensor,
            edges,
            self._softminmax_order,
            self._softminmax_cutoff,
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
        vertex_update_candidates: Float[Array, "max_num_adjacent_simplices 4"],
        grad_update_solution_candidates: Float[Array, "max_num_adjacent_simplices 4 2"],
        grad_update_parameter_candidates: Float[Array, "max_num_adjacent_simplices 4 dim dim"],
    ) -> tuple[
        Float[Array, ""],
        Float[Array, "max_num_adjacent_simplices 4 2"],
        Float[Array, "max_num_adjacent_simplices 4 dim dim"],
    ]:
        """Mask irrelevant derivative candidates so that they are discarded later.

        Values are masked by setting them to zero or infinity, depending on the routine in which
        they are utilized later. Partial derivatives are only relevant if the corresponding update
        corresponds to an optimal path.

        Args:
            vertex_update_candidates (jnp.ndarray): Update candidates for a given vertex
            grad_update_solution_candidates (jnp.ndarray): Partial derivative candidates w.r.t. the
                solution vector
            grad_update_parameter_candidates (jnp.ndarray): Partial derivative candidates w.r.t. the
                parameter field

        Returns:
            tuple[float, jnp.ndarray, jnp.ndarray]: Optimal update values, masked partial
                derivatives
        """
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
        solution_values: Float[Array, "2"],
        parameter_tensor: Float[Array, "dim dim"],
        edges: tuple[Float[Array, "dim"], Float[Array, "dim"], Float[Array, "dim"]],
    ) -> tuple[Float[Array, "4 2"], Float[Array, "4 dim dim"]]:
        """Compute the partial derivatives of update parameters for a single vertex.

        This method evaluates the partial derivatives of the update parameters with respect to the
        current solution vector and the given parameter field, for a single triangle.

        Args:
            solution_values (jnp.ndarray): Current solution values at the opposite vertices of the
                considered triangle
            parameter_tensor (jnp.ndarray): Parameter tensor for the given triangle
            edges (tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]): Edges of the considered triangle

        Returns:
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Jacobians of the update parameters w.r.t.
                the solution vector and the parameter tensor
        """
        lambda_partial_solution = corefunctions.jac_lambda_soft_solution(
            solution_values,
            parameter_tensor,
            edges,
            self._softminmax_order,
            self._softminmax_cutoff,
        )
        lambda_partial_parameter = corefunctions.jac_lambda_soft_parameter(
            solution_values,
            parameter_tensor,
            edges,
            self._softminmax_order,
            self._softminmax_cutoff,
        )

        return lambda_partial_solution, lambda_partial_parameter


# ==================================================================================================
class DerivativeSolver:
    """Main component for obtaining gradients from partial derivatives.

    The Eikonax derivator computes partial derivatives of the global update operator with respect
    to the solution vector and the parameter tensor field. At the fixed point of the iterative
    update scheme, meaning the correct solution according to the discrete upwind scheme, these
    parrial derivatives can be used to obtain the total Jacobian of the global update operator
    with respect to the parameter tensor field. In practice, we are typically concerned with the
    parametric gradient of a cost functional that comprises the solution vector. The partial
    differential equation, which connects solution vector and parameter field, acts as a constraint
    in this context. The partial derivator computes the the so-called adjoint variable in this
    context, by solving a linear system of equation. The system matrix is obtained from the
    partial derivative of the global update operator with respect to the solution vector.
    Importantly, we can order the indices of the solution vector according to the size of the
    respective solution values. Because the update operator obeys upwind causality, the system
    matrix becomes triangular under such a permutation, and we can solve the linear system through
    simple back-substitution. In the context of an optimization problem, the right-hand-side is
    given as the partial derivative of the cost functional with respect to the solution vector.

    Methods:
        solve: Solve the linear system for the adjoint variable
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        solution: Float[Array | np.ndarray, "num_vertices"],
        sparse_partial_update_solution: tuple[
            Int[Array, "num_sol_values"],
            Int[Array, "num_sol_values"],
            Float[Array, "num_sol_values"],
        ],
    ) -> None:
        """Constructor for the derivative solver.

        Initializes the causality-inspired permutation matrix, and afterwards the permuted system
        matrix, which is triangular.

        Args:
            solution (jnp.ndarray): Obtained solution of the Eikonal equation
            sparse_partial_update_solution (tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]): Sparse
                representation of the partial derivative G_u, containing row inds, column inds and
                values. These structures might contain redundances, which are automatically removed
                through summation in the sparse matrix assembly later.
        """
        num_points = solution.size
        solution = np.array(solution, dtype=np.float32)
        sparse_partial_update_solution = (
            np.array(sparse_partial_update_solution[0], dtype=np.int32),
            np.array(sparse_partial_update_solution[1], dtype=np.int32),
            np.array(sparse_partial_update_solution[2], dtype=np.float32),
        )
        self._sparse_permutation_matrix = self._assemble_permutation_matrix(solution)
        self._sparse_system_matrix = self._assemble_system_matrix(
            sparse_partial_update_solution, num_points
        )

    # ----------------------------------------------------------------------------------------------
    def solve(
        self, right_hand_side: Float[Array | np.ndarray, "num_vertices"]
    ) -> Float[np.ndarray, "num_parameters"]:
        """Solve the linear system for the parametric gradient.

        The right-hand-siide needs to be given as the partial derivative of the prescribed cost
        functional w.r.t. the solution vector. This right-hand-side is permutated according to the
        causality of the solution. Subsequently, the linear system can be solved through (sparse)
        back-substitution. The solution is then permutated back to the original ordering.

        Args:
            right_hand_side (np.ndarray | jnp.ndarray): RHS for the linear system solve

        Returns:
            np.ndarray: Solution of the linear system solve, corresponding to the adjoint in an
                optimization context.
        """
        right_hand_side = np.array(right_hand_side, dtype=np.float32)
        permutated_right_hand_side = self._sparse_permutation_matrix @ right_hand_side
        permutated_solution = sp.sparse.linalg.spsolve_triangular(
            self._sparse_system_matrix, permutated_right_hand_side, lower=False, unit_diagonal=True
        )
        solution = self._sparse_permutation_matrix.T @ permutated_solution

        return solution

    # ----------------------------------------------------------------------------------------------
    def _assemble_permutation_matrix(
        self, solution: Float[np.ndarray, "num_vertices"]
    ) -> sp.sparse.csc_matrix:
        """Construct permutation matrix for index ordering.

        From a given solution vector, we know from the properties of the upwind update scheme
        that causaility is given through the size of the respective solution values. This means
        that nodes with higher solution values are only influenced by nodes with lower solution
        values. With respect to linear system solves involving the global update operator,
        this means that we can obtain triangular matrices through an according permutation.

        Args:
            solution (jnp.ndarray): Obtained solution of the eikonal equation

        Returns:
            sp.sparse.csc_matrix: Sparse permutation matrix
        """
        num_points = solution.size
        permutation_row_inds = np.arange(solution.size)
        permutation_col_inds = np.argsort(solution)
        permutation_values = np.ones(solution.size)
        sparse_permutation_matrix = sp.sparse.csc_matrix(
            (permutation_values, (permutation_row_inds, permutation_col_inds)),
            shape=(num_points, num_points),
        )

        return sparse_permutation_matrix

    # ----------------------------------------------------------------------------------------------
    def _assemble_system_matrix(
        self,
        sparse_partial_update_solution: tuple[
            Int[np.ndarray, "num_sol_values"],
            Int[np.ndarray, "num_sol_values"],
            Float[np.ndarray, "num_sol_values"],
        ],
        num_points: int,
    ) -> sp.sparse.csc_matrix:
        """Assemble system matrix for gradient solver.

        The parametric gradient of the global update operator is obtained from a linear system
        solve, where the system matrix is given as (I-G_u)^T, where G_u is the partial derivative of
        the global update operator with respect to the solution vector. According to the
        causality of the solution, we can permutate the system matrix to a triangular form.

        Args:
            sparse_partial_update_solution (tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
                Sparse representation of the partial derivative G_u, containing row inds, column
                inds and values. These structures might contain redundances, which are
                automatically removed through summation in the sparse matrix assembly.
            num_points (int): Number of mesh points

        Returns:
            sp.sparse.csc_matrix: Sparse representation of the permuted system matrix
        """
        rows_compressed, columns_compressed, values_compressed = sparse_partial_update_solution
        sparse_partial_matrix = sp.sparse.csc_matrix(
            (values_compressed, (rows_compressed, columns_compressed)),
            shape=(num_points, num_points),
        )
        sparse_identity_matrix = sp.sparse.identity(num_points, format="csc")
        sparse_system_matrix = sparse_identity_matrix - sparse_partial_matrix
        sparse_system_matrix = (
            self._sparse_permutation_matrix
            @ sparse_system_matrix.T
            @ self._sparse_permutation_matrix.T
        )

        return sparse_system_matrix
