import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from . import utilities as utils


# ==================================================================================================
@chex.dataclass
class MeshData:
    vertices: jnp.ndarray
    adjacent_vertex_inds: jnp.ndarray


@chex.dataclass
class SolverData:
    tolerance: float
    max_num_iterations: int
    max_value: float
    softmin_order: int
    drelu_order: int
    drelu_cutoff: float


@chex.dataclass
class InitialSites:
    inds: jnp.ndarray
    values: jnp.ndarray


# ==================================================================================================
class CoreFunctions:

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def get_adjacent_edges(
        i: int, j: int, k: int, vertices: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        e_ji = vertices[i] - vertices[j]
        e_ki = vertices[i] - vertices[k]
        e_jk = vertices[k] - vertices[j]
        return e_ji, e_ki, e_jk

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def compute_optimal_update_parameters(
        solution_values: jnp.ndarray,
        edges: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        M: jnp.ndarray,
        order: int,
        cutoff: float,
    ) -> jnp.ndarray:
        lambda_1, lambda_2 = CoreFunctions._compute_optimal_update_parameters(
            solution_values, edges, M
        )
        lambda_1_clipped = utils.compute_soft_drelu(lambda_1, order)
        lambda_2_clipped = utils.compute_soft_drelu(lambda_2, order)
        lower_bounds = -cutoff
        upper_bound = 1 + cutoff

        lambda_1_clipped = jnp.where(
            (lambda_1 < lower_bounds) | (lambda_1 > upper_bound), -1, lambda_1_clipped
        )
        lambda_2_clipped = jnp.where(
            (lambda_2 < -lower_bounds) | (lambda_2 > upper_bound) | (lambda_2 == lambda_1),
            -1,
            lambda_2_clipped,
        )
        lambda_array = jnp.array((lambda_1_clipped, lambda_2_clipped))
        return lambda_array

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def compute_fixed_update(
        lambda_value: float,
        solution_values: jnp.ndarray,
        edges: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        M: jnp.ndarray,
    ) -> float:
        u_j, u_k = solution_values
        e_ji, _, e_jk = edges
        diff_vector = e_ji - lambda_value * e_jk
        transport_term = jnp.sqrt(jnp.dot(diff_vector, M @ diff_vector))
        update = lambda_value * u_k + (1 - lambda_value) * u_j + transport_term
        return update

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _compute_optimal_update_parameters(
        solution_values: jnp.ndarray,
        edges: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        M: jnp.ndarray,
    ) -> tuple[float, float]:
        u_j, u_k = solution_values
        _, e_ki, e_jk = edges
        delta_u = u_k - u_j
        a_1 = jnp.dot(e_jk, M @ e_jk)
        a_2 = jnp.dot(e_jk, M @ e_ki)
        a_3 = jnp.dot(e_ki, M @ e_ki)

        nominator = a_1 * a_3 - a_2**2
        denominator = a_1 - delta_u**2
        sqrt_term = jnp.where(denominator > 0, nominator / denominator, jnp.inf)
        c = delta_u * sqrt_term

        lambda_1 = (-a_2 + c) / a_1
        lambda_2 = (-a_2 - c) / a_1
        return lambda_1, lambda_2


# ==================================================================================================
class Solver:

    # ----------------------------------------------------------------------------------------------
    def __init__(self, mesh_data: MeshData, solver_data: SolverData):
        self._vertices = mesh_data.vertices
        self._adjacent_vertex_inds = mesh_data.adjacent_vertex_inds

        self._tolerance = solver_data.tolerance
        self._max_num_iterations = solver_data.max_num_iterations
        self._max_value = solver_data.max_value
        self._softmin_order = solver_data.softmin_order
        self._drelu_order = solver_data.drelu_order
        self._drelu_cutoff = solver_data.drelu_cutoff

    # ----------------------------------------------------------------------------------------------
    def run(self, initial_sites: InitialSites, tensor_field: jnp.ndarray):
        solution_vector = jnp.ones(self._vertices.shape[0]) * self._max_value
        solution_vector = solution_vector.at[initial_sites.inds].set(initial_sites.values)
        solution_vector = Solver._run(
            solution_vector,
            tensor_field,
            self._max_num_iterations,
            self._adjacent_vertex_inds,
            self._vertices,
            self._drelu_order,
            self._drelu_cutoff,
            self._softmin_order,
        )
        return solution_vector

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    @eqx.filter_jit
    def _run(
        solution_vector,
        tensor_field,
        num_iterations,
        adjacent_simplices_inds,
        vertices,
        drelu_order,
        drelu_cutoff,
        softmin_order,
    ):
        compute_global_update = jax.vmap(
            Solver._compute_vertex_update, in_axes=(None, 0, None, None, None, None, None)
        )

        def loop_body_for(_, carry_args):
            solution_vector, *parameters = carry_args
            solution_vector = compute_global_update(solution_vector, *parameters)
            return solution_vector, *parameters

        solution_vector, *_ = jax.lax.fori_loop(
            0,
            num_iterations,
            loop_body_for,
            (
                solution_vector,
                adjacent_simplices_inds,
                vertices,
                tensor_field,
                drelu_order,
                drelu_cutoff,
                softmin_order,
            ),
        )
        return solution_vector

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _compute_vertex_update(
        old_solution_vector,
        adjacent_simplices_inds,
        vertices,
        tensor_field,
        drelu_order,
        drelu_cutoff,
        softmin_order,
    ):
        max_num_adjacent_simplices = adjacent_simplices_inds.shape[0]
        vertex_update_candidates = jnp.zeros((max_num_adjacent_simplices, 4))

        for i, indices in enumerate(adjacent_simplices_inds):
            simplex_update_candidates = Solver._compute_update_from_adjacent_simplex(
                indices, old_solution_vector, vertices, tensor_field, drelu_order, drelu_cutoff
            )
            vertex_update_candidates = vertex_update_candidates.at[i, :].set(
                simplex_update_candidates
            )

        active_inds = jnp.expand_dims(adjacent_simplices_inds[:, 3], axis=-1)
        self_update = jnp.expand_dims(old_solution_vector[adjacent_simplices_inds[0, 0]], axis=-1)
        vertex_update_candidates = jnp.where(active_inds != -1, vertex_update_candidates, jnp.inf)
        vertex_update_candidates = jnp.concatenate(
            (self_update, vertex_update_candidates.flatten())
        )

        min_index = jnp.argmin(vertex_update_candidates)
        min_value = vertex_update_candidates[min_index]
        vertex_update_candidates = jnp.where(
            vertex_update_candidates == min_value, vertex_update_candidates, jnp.inf
        )
        vertex_update = utils.compute_softmin(vertex_update_candidates, softmin_order, min_index)
        return vertex_update

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _compute_update_from_adjacent_simplex(
        indices, old_solution_vector, vertices, tensor_field, drelu_order, drelu_cutoff
    ):
        i, j, k, s = indices
        solution_values = jnp.array((old_solution_vector[j], old_solution_vector[k]))
        edges = CoreFunctions.get_adjacent_edges(i, j, k, vertices)
        M = tensor_field[s]
        lambda_array = CoreFunctions.compute_optimal_update_parameters(
            solution_values, edges, M, drelu_order, drelu_cutoff
        )
        lambda_array = jnp.concatenate((jnp.array((0, 1)), lambda_array))
        update_candidates = jnp.zeros(4)

        for i, lambda_candidate in enumerate(lambda_array):
            update = CoreFunctions.compute_fixed_update(lambda_candidate, solution_values, edges, M)
            update_candidates = update_candidates.at[i].set(update)
        update_candidates = jnp.where(lambda_array != -1, update_candidates, jnp.inf)

        return update_candidates


# ==================================================================================================
class Derivator:
    pass
