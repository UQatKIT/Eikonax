import time

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from . import corefunctions, logging, utilities


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
class DerivatorData:
    softmin_order: int
    drelu_order: int
    drelu_cutoff: int


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
            raise ValueError(
                f"DReLU cutoff needs to be positive, but is {solver_data.drelu_cutoff}"
            )
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
            "adjacent_vertex_inds": mesh_data.adjacent_vertex_inds,
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
        adjacent_vertex_inds: jnp.ndarray,
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
                adjacent_vertex_inds,
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
        adjacent_vertex_inds: jnp.ndarray,
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
                adjacent_vertex_inds,
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
        adjacent_vertex_inds: jnp.ndarray,
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
                adjacent_vertex_inds,
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
        adjacent_vertex_inds: jnp.ndarray,
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
        assert adjacent_vertex_inds.shape[0] == vertices.shape[0], (
            f"Adjacent simplex indix array needs to have shape {vertices.shape[0]}, "
            f"but has shape {adjacent_vertex_inds.shape[0]}"
        )
        global_update_function = jax.vmap(
            Solver._compute_vertex_update, in_axes=(None, 0, None, None, None, None, None)
        )
        global_update = global_update_function(
            solution_vector,
            adjacent_vertex_inds,
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
    def _compute_vertex_update(
        old_solution_vector: jnp.ndarray,
        adjacent_vertex_inds: jnp.ndarray,
        vertices: jnp.ndarray,
        tensor_field: jnp.ndarray,
        drelu_order: int,
        drelu_cutoff: int,
        softmin_order: int,
    ) -> float:
        vertex_update_candidates = corefunctions.compute_vertex_update_candidates(
            old_solution_vector,
            adjacent_vertex_inds,
            vertices,
            tensor_field,
            drelu_order,
            drelu_cutoff,
        )
        self_update = jnp.expand_dims(old_solution_vector[adjacent_vertex_inds[0, 0]], axis=-1)
        vertex_update_candidates = jnp.concatenate(
            (self_update, vertex_update_candidates.flatten())
        )

        min_value = jnp.min(vertex_update_candidates)
        vertex_update_candidates = jnp.where(
            vertex_update_candidates == min_value, vertex_update_candidates, jnp.inf
        )
        vertex_update = utilities.compute_softmin(
            vertex_update_candidates, min_value, softmin_order
        )
        assert (
            vertex_update.shape == ()
        ), f"Vertex update has to be scalar, but has shape {vertex_update.shape}"
        return vertex_update


# ==================================================================================================
class Derivator:
    @staticmethod
    def compute_partial_derivatives(
        solution_vector: jnp.ndarray,
        tensor_field: jnp.ndarray,
        mesh_data: MeshData,
        derivator_data: DerivatorData,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        partial_derivative_solution, partial_derivative_parameter = (
            Derivator._compute_global_partial_derivatives(
                solution_vector,
                mesh_data.adjacent_vertex_inds,
                mesh_data.vertices,
                tensor_field,
                derivator_data.drelu_order,
                derivator_data.drelu_cutoff,
                derivator_data.softmin_order,
            )
        )

        return partial_derivative_solution, partial_derivative_parameter

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    @eqx.filter_jit
    def _compute_global_partial_derivatives(
        solution_vector: jnp.ndarray,
        adjacent_vertex_inds: jnp.ndarray,
        vertices: jnp.ndarray,
        tensor_field: jnp.ndarray,
        drelu_order: int,
        drelu_cutoff: int,
        softmin_order: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        assert solution_vector.shape[0] == vertices.shape[0], (
            f"Solution vector needs to have shape {vertices.shape[0]}, "
            f"but has shape {solution_vector.shape}"
        )
        assert adjacent_vertex_inds.shape[0] == vertices.shape[0], (
            f"Adjacent simplex indix array needs to have shape {vertices.shape[0]}, "
            f"but has shape {adjacent_vertex_inds.shape[0]}"
        )
        global_partial_derivative_function = jax.vmap(
            Derivator._compute_vertex_partial_derivatives,
            in_axes=(None, 0, None, None, None, None, None),
        )
        partial_derivative_solution, partial_derivative_parameter = (
            global_partial_derivative_function(
                solution_vector,
                adjacent_vertex_inds,
                vertices,
                tensor_field,
                drelu_order,
                drelu_cutoff,
                softmin_order,
            )
        )

        max_num_adjacent_simplices = adjacent_vertex_inds.shape[1]
        tensor_dim = tensor_field.shape[1]
        # assert partial_derivative_solution.shape == (
        #     solution_vector.shape[0],
        #     max_num_adjacent_simplices,
        #     2,
        # )
        # assert partial_derivative_parameter.shape == (
        #     solution_vector.shape[0],
        #     max_num_adjacent_simplices,
        #     tensor_dim,
        #     tensor_dim,
        # )
        return partial_derivative_solution, partial_derivative_parameter

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _compute_vertex_partial_derivatives(
        solution_vector: jnp.ndarray,
        adjacent_vertex_inds: jnp.ndarray,
        vertices: jnp.ndarray,
        tensor_field: jnp.ndarray,
        drelu_order: int,
        drelu_cutoff: int,
        softmin_order: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        vertex_update_candidates = corefunctions.compute_vertex_update_candidates(
            solution_vector,
            adjacent_vertex_inds,
            vertices,
            tensor_field,
            drelu_order,
            drelu_cutoff,
        )
        max_num_adjacent_simplices = adjacent_vertex_inds.shape[0]
        tensor_dim = tensor_field.shape[1]
        assert adjacent_vertex_inds.shape == (max_num_adjacent_simplices, 4), (
            f"node-level adjacency data needs to have shape ({max_num_adjacent_simplices}, 4), "
            f"but has shape {adjacent_vertex_inds.shape}"
        )
        grad_update_solution_candidates = jnp.zeros((max_num_adjacent_simplices, 4, 2))
        grad_update_parameter_candidates = jnp.zeros(
            (max_num_adjacent_simplices, 4, tensor_dim, tensor_dim)
        )
        grad_update_solution = jnp.zeros((max_num_adjacent_simplices, 2))
        grad_update_parameter = jnp.zeros((max_num_adjacent_simplices, tensor_dim, tensor_dim))

        for i, indices in enumerate(adjacent_vertex_inds):
            partial_solution, partial_parameter = (
                Derivator._compute_partial_derivatives_from_adjacent_simplex(
                    indices, solution_vector, vertices, tensor_field, drelu_order, drelu_cutoff
                )
            )
            grad_update_solution_candidates = grad_update_solution_candidates.at[i, ...].set(
                partial_solution
            )
            grad_update_parameter_candidates = grad_update_parameter_candidates.at[i, ...].set(
                partial_parameter
            )

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
        softmin_grad = utilities.compute_softmin_grad(
            vertex_update_candidates.flatten(), min_value, softmin_order
        ).reshape(vertex_update_candidates.shape)

        for i in range(max_num_adjacent_simplices):
            grad_update_solution = grad_update_solution.at[i, :].set(
                jnp.tensordot(softmin_grad[i, :], grad_update_solution_candidates[i, ...], axes=1)
            )
            grad_update_parameter = grad_update_parameter.at[i, ...].set(
                jnp.tensordot(softmin_grad[i, :], grad_update_parameter_candidates[i, ...], axes=1)
            )

        return grad_update_solution, grad_update_parameter

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _compute_partial_derivatives_from_adjacent_simplex(
        indices: jnp.ndarray,
        solution_vector: jnp.ndarray,
        vertices: jnp.ndarray,
        tensor_field: jnp.ndarray,
        drelu_order: int,
        drelu_cutoff: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        assert len(indices) == 4, f"Indices need to have length 4, but have length {len(indices)}"
        i, j, k, s = indices
        tensor_dim = tensor_field.shape[1]
        solution_values = jnp.array((solution_vector[j], solution_vector[k]))
        edges = corefunctions.get_adjacent_edges(i, j, k, vertices)
        M = tensor_field[s]
        lambda_array = corefunctions.compute_optimal_update_parameters(
            solution_values, M, edges, drelu_order, drelu_cutoff
        )
        lambda_partial_solution, lambda_partial_parameter = Derivator._compute_lambda_grad(
            solution_values, M, edges, drelu_order, drelu_cutoff
        )
        lambda_partial_solution = jnp.concatenate((jnp.zeros((2, 2)), lambda_partial_solution))
        lambda_partial_parameter = jnp.concatenate(
            (jnp.zeros((2, tensor_dim, tensor_dim)), lambda_partial_parameter)
        )
        grad_update_solution = jnp.zeros((4, 2))
        grad_update_parameter = jnp.zeros((4, tensor_dim, tensor_dim))

        for i in range(4):
            update_partial_lambda = corefunctions.grad_update_lambda(
                lambda_array[i], solution_values, M, edges
            )
            update_partial_solution = corefunctions.grad_update_solution(
                lambda_array[i], solution_values, M, edges
            )
            update_partial_parameter = corefunctions.grad_update_parameter(
                lambda_array[i], solution_values, M, edges
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
    def _compute_lambda_grad(
        solution_values: jnp.ndarray,
        M: jnp.ndarray,
        edges: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        drelu_order: int,
        drelu_cutoff: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        lambda_partial_solution = corefunctions.jac_lambda_solution(
            solution_values, M, edges, drelu_order, drelu_cutoff
        )
        lambda_partial_parameter = corefunctions.jac_lambda_parameter(
            solution_values, M, edges, drelu_order, drelu_cutoff
        )

        return lambda_partial_solution, lambda_partial_parameter
