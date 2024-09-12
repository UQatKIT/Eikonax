import time

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from . import utilities as utils
from . import logging as logging


# ==================================================================================================
@chex.dataclass
class MeshData:
    vertices: jnp.ndarray
    adjacent_vertex_inds: jnp.ndarray


@chex.dataclass
class SolverData:
    loop_type: str
    max_value: float
    softmin_order: int
    drelu_order: int
    drelu_cutoff: float
    max_num_iterations: int
    tolerance: float | None = None
    log_interval: int | None = None
    logger: logging.Logger | None = None


@chex.dataclass
class InitialSites:
    inds: jnp.ndarray
    values: jnp.ndarray


@chex.dataclass
class Solution:
    values: jnp.ndarray
    num_iterations: int
    tolerance: float | jnp.ndarray | None = None


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
        assert e_ji.shape == e_ki.shape == e_jk.shape, "All edges need to have the same shape"
        assert len(e_ji.shape) == 1, "Edges need to be 1D arrays"
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
        assert solution_values.shape == (
            2,
        ), f"Solution values need to have shape (2,), but have {solution_values.shape}"
        assert (len(edge.shape) == 1 for edge in edges), "Edges need to be 1D arrays"
        assert len(M.shape) == 2, f"M needs to be a 2D array, bu has shape {M.shape}"
        assert M.shape[0] == M.shape[1] == edges[0].shape[0], (
            f"M needs to be a square matrix, with dimensions matching edges, "
            f"but M has shape {M.shape} and first edge has shape {edges[0].shape}"
        )

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
        assert lambda_array.shape == (
            2,
        ), f"Lambda array needs to have shape (2,), but has shape {lambda_array.shape}"
        return lambda_array

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def compute_fixed_update(
        lambda_value: float,
        solution_values: jnp.ndarray,
        edges: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        M: jnp.ndarray,
    ) -> float:
        assert solution_values.shape == (
            2,
        ), f"Solution values need to have shape (2,), but have {solution_values.shape}"
        assert M.shape[0] == M.shape[1] == edges[0].shape[0], (
            f"M needs to be a square matrix, with dimensions matching edges, "
            f"but M has shape {M.shape} and first edge has shape {edges[0].shape}"
        )
        u_j, u_k = solution_values
        e_ji, _, e_jk = edges
        diff_vector = e_ji - lambda_value * e_jk
        transport_term = jnp.sqrt(jnp.dot(diff_vector, M @ diff_vector))
        update = lambda_value * u_k + (1 - lambda_value) * u_j + transport_term
        assert update.shape == (), f"Update needs to be a scalar, but has shape {update.shape}"
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
    @staticmethod
    def run(
        tensor_field: jnp.ndarray,
        initial_sites: InitialSites,
        mesh_data: MeshData,
        solver_data: SolverData,
        logger: logging.Logger | None = None,
    ) -> Solution:
        if solver_data.max_value <= 0:
            raise ValueError(f"Max value needs to be positive, but is {solver_data.max_value}")
        if solver_data.max_num_iterations <= 0:
            raise ValueError(
                f"Max number of iterations needs to be positive, "
                f"but is {solver_data.max_num_iterations}"
            )
        if solver_data.drelu_cutoff <= 0:
            raise ValueError(f"DReLU cutoff needs to be positive, but is {solver_data.drelu_cutoff}")
        if solver_data.drelu_order <= 0:
            raise ValueError(f"DReLU order needs to be positive, but is {solver_data.drelu_order}")
        if solver_data.softmin_order <= 0:
            raise ValueError(
                f"Softmin order needs to be positive, but is {solver_data.softmin_order}"
            )
        
        if solver_data.tolerance is not None and solver_data.tolerance <= 0:
            raise ValueError(f"Tolerance needs to be positive, but is {solver_data.tolerance}")
        if solver_data.log_interval is not None and solver_data.log_interval <= 0:
            raise ValueError(
                f"Log interval needs to be positive, but is {solver_data.log_interval}"
            )

        initial_guess = jnp.ones(mesh_data.vertices.shape[0]) * solver_data.max_value
        initial_guess = initial_guess.at[initial_sites.inds].set(initial_sites.values)

        shared_args = {
            "initial_guess": initial_guess,
            "tensor_field": tensor_field,
            "adjacent_simplices_inds": mesh_data.adjacent_vertex_inds,
            "vertices": mesh_data.vertices,
            "drelu_order": solver_data.drelu_order,
            "drelu_cutoff": solver_data.drelu_cutoff,
            "softmin_order": solver_data.softmin_order,
        }

        match solver_data.loop_type:
            # Jitted for loop, does not take tolerance threshold
            case "jitted_for":
                run_function = Solver._run_jitted_for_loop
                special_args = {"num_iterations": solver_data.max_num_iterations}
            # Jitted while loop, takes tolerance threshold
            case "jitted_while":
                run_function = Solver._run_jitted_while_loop
                special_args = {
                    "tolerance_threshold": solver_data.tolerance,
                    "num_iterations": solver_data.max_num_iterations,
                }
            # Non-jitted while loop, additionally performs online logging
            case "nonjitted_while":
                run_function = Solver._run_nonjitted_while_loop
                special_args = {
                    "tolerance_threshold": solver_data.tolerance,
                    "num_iterations": solver_data.max_num_iterations,
                    "log_interval": solver_data.log_interval,
                    "logger": logger,
                }
            case _:
                raise ValueError(f"Invalid loop type: {solver_data.loop_type}")

        solution_vector, num_iterations, tolerance = run_function(**shared_args, **special_args)
        assert solution_vector.shape[0] == mesh_data.vertices.shape[0], (
            f"Solution vector needs to have shape {mesh_data.vertices.shape[0]}, "
            f"but has shape {solution_vector.shape}"
        )
        solution = Solution(
            values=solution_vector, num_iterations=num_iterations, tolerance=tolerance
        )
        return solution

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    @eqx.filter_jit
    def _run_jitted_for_loop(
        initial_guess: jnp.ndarray,
        tensor_field: jnp.ndarray,
        adjacent_simplices_inds: jnp.ndarray,
        vertices: jnp.ndarray,
        drelu_order: int,
        drelu_cutoff: float,
        softmin_order: int,
        num_iterations: int,
    ) -> tuple[jnp.ndarray, int, float]:
        def loop_body_for(_, carry_args):
            new_solution_vector, tolerance, old_solution_vector, *parameters = carry_args
            old_solution_vector = new_solution_vector
            new_solution_vector = Solver._compute_global_update(old_solution_vector, *parameters)
            tolerance = jnp.max(jnp.abs(new_solution_vector - old_solution_vector))
            return new_solution_vector, tolerance, old_solution_vector, *parameters

        initial_tolerance = 0
        initial_old_solution = jnp.zeros(initial_guess.shape)
        solution_vector, tolerance, *_ = jax.lax.fori_loop(
            0,
            num_iterations,
            loop_body_for,
            (
                initial_guess,
                initial_tolerance,
                initial_old_solution,
                adjacent_simplices_inds,
                vertices,
                tensor_field,
                drelu_order,
                drelu_cutoff,
                softmin_order,
            ),
        )
        return solution_vector, num_iterations, tolerance

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    @eqx.filter_jit
    def _run_jitted_while_loop(
        initial_guess: jnp.ndarray,
        tensor_field: jnp.ndarray,
        adjacent_simplices_inds: jnp.ndarray,
        vertices: jnp.ndarray,
        drelu_order: int,
        drelu_cutoff: float,
        softmin_order: int,
        num_iterations: int,
        tolerance_threshold: float,
    ) -> tuple[jnp.ndarray, int, float]:
        if tolerance_threshold is None:
            raise ValueError("Tolerance threshold must be provided for while loop")

        def loop_body_while(carry_args):
            new_solution_vector, iteration_counter, tolerance, old_solution_vector, *parameters = (
                carry_args
            )
            old_solution_vector = new_solution_vector
            new_solution_vector = Solver._compute_global_update(old_solution_vector, *parameters)
            tolerance = jnp.max(jnp.abs(new_solution_vector - old_solution_vector))
            iteration_counter += 1
            return (
                new_solution_vector,
                iteration_counter,
                tolerance,
                old_solution_vector,
                *parameters,
            )

        def cond_while(carry_args):
            new_solution_vector, iteration_counter, tolerance, old_solution_vector, *parameters = (
                carry_args
            )
            tolerance = jnp.max(jnp.abs(new_solution_vector - old_solution_vector))
            return (tolerance > tolerance_threshold) & (iteration_counter < num_iterations)

        initial_old_solution = jnp.zeros(initial_guess.shape)
        initial_tolerance = 0
        iteration_counter = 0
        solution_vector, num_iterations, tolerance, *_ = jax.lax.while_loop(
            cond_while,
            loop_body_while,
            (
                initial_guess,
                iteration_counter,
                initial_tolerance,
                initial_old_solution,
                adjacent_simplices_inds,
                vertices,
                tensor_field,
                drelu_order,
                drelu_cutoff,
                softmin_order,
            ),
        )

        return solution_vector, num_iterations, tolerance

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _run_nonjitted_while_loop(
        initial_guess: jnp.ndarray,
        tensor_field: jnp.ndarray,
        adjacent_simplices_inds: jnp.ndarray,
        vertices: jnp.ndarray,
        drelu_order: int,
        drelu_cutoff: float,
        softmin_order: int,
        num_iterations: int,
        tolerance_threshold: float,
        log_interval: int,
        logger: logging.Logger,
    ) -> tuple[jnp.ndarray, int, jnp.ndarray]:
        if tolerance_threshold is None:
            raise ValueError("Tolerance threshold must be provided for while loop")
        if log_interval is None:
            raise ValueError("Log interval must be provided for while loop")
        if logger is None:
            raise ValueError("Logger must be provided for while loop")

        iteration_counter = 0
        old_solution_vector = initial_guess
        tolerance = jnp.inf
        tolerance_vector = []
        start_time = time.time()

        log_values = {
            "time": logging.LogValue(f"{'Time[s]:':<15}", "<15.3e"),
            "iters": logging.LogValue(f"{'#Iterations:':<15}", "<15.3e"),
            "tol": logging.LogValue(f"{'Tolerance:':<15}", "<15.3e"),
        }
        logger.header(log_values)

        while (tolerance > tolerance_threshold) and (iteration_counter < num_iterations):
            new_solution_vector = Solver._compute_global_update(
                old_solution_vector,
                adjacent_simplices_inds,
                vertices,
                tensor_field,
                drelu_order,
                drelu_cutoff,
                softmin_order,
            )
            tolerance = jnp.max(jnp.abs(new_solution_vector - old_solution_vector))
            tolerance_vector.append(tolerance)
            old_solution_vector = new_solution_vector
            iteration_counter += 1

            if (iteration_counter % log_interval == 0) or (iteration_counter == num_iterations):
                current_time = time.time() - start_time
                log_values["time"].value = current_time
                log_values["iters"].value = iteration_counter
                log_values["tol"].value = tolerance
                logger.log(log_values)

        tolerance_vector = jnp.array(tolerance_vector)
        return new_solution_vector, iteration_counter, tolerance_vector

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    @eqx.filter_jit
    def _compute_global_update(
        solution_vector: jnp.ndarray,
        adjacent_simplices_inds: jnp.ndarray,
        vertices: jnp.ndarray,
        tensor_field: jnp.ndarray,
        drelu_order: int,
        drelu_cutoff: float,
        softmin_order: int,
    ) -> jnp.ndarray:
        assert solution_vector.shape[0] == vertices.shape[0], (
            f"Solution vector needs to have shape {vertices.shape[0]}, "
            f"but has shape {solution_vector.shape}"
        )
        assert adjacent_simplices_inds.shape[0] == vertices.shape[0], (
            f"Adjacent simplex indix array needs to have shape {vertices.shape[0]}, "
            f"but has shape {adjacent_simplices_inds.shape[0]}"
        )
        global_update_function = jax.vmap(
            Solver._compute_vertex_update, in_axes=(None, 0, None, None, None, None, None)
        )
        global_update = global_update_function(
            solution_vector,
            adjacent_simplices_inds,
            vertices,
            tensor_field,
            drelu_order,
            drelu_cutoff,
            softmin_order,
        )

        assert global_update.shape == solution_vector.shape, (
            f"New solution has shape {global_update.shape}, "
            f"but should have shape {solution_vector.shape}"
        )
        return global_update

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    @eqx.filter_jit
    def _compute_vertex_update(
        old_solution_vector: jnp.ndarray,
        adjacent_simplices_inds: jnp.ndarray,
        vertices: jnp.ndarray,
        tensor_field: jnp.ndarray,
        drelu_order: int,
        drelu_cutoff: int,
        softmin_order: int,
    ) -> float:
        max_num_adjacent_simplices = adjacent_simplices_inds.shape[0]
        assert adjacent_simplices_inds.shape == (max_num_adjacent_simplices, 4), (
            f"node-level adjacency data needs to have shape ({max_num_adjacent_simplices}, 4), "
            f"but has shape {adjacent_simplices_inds.shape}"
        )
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

        min_value = jnp.min(vertex_update_candidates)
        vertex_update_candidates = jnp.where(
            vertex_update_candidates == min_value, vertex_update_candidates, jnp.inf
        )
        vertex_update = utils.compute_softmin(vertex_update_candidates, softmin_order)
        assert (
            vertex_update.shape == ()
        ), f"Vertex update has to be scalar, but has shape {vertex_update.shape}"
        return vertex_update

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _compute_update_from_adjacent_simplex(
        indices: jnp.ndarray,
        old_solution_vector: jnp.ndarray,
        vertices: jnp.ndarray,
        tensor_field: jnp.ndarray,
        drelu_order: int,
        drelu_cutoff: float,
    ) -> jnp.ndarray:
        assert len(indices) == 4, f"Indices need to have length 4, but have length {len(indices)}"
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
        assert update_candidates.shape == (4,), (
            f"Update candidates have shape {update_candidates.shape}, "
            "but need to have shape (4,)"
        )
        return update_candidates


# ==================================================================================================
class Derivator:
    pass
