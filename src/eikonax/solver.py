import time

import chex
import equinox as eqx
import jax
import jax.numpy as jnp

from . import corefunctions, logging


# ==================================================================================================
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


@chex.dataclass
class Solution:
    values: jnp.ndarray
    num_iterations: int
    tolerance: float | jnp.ndarray | None = None


# ==================================================================================================
class Solver(eqx.Module):
    _vertices: jnp.ndarray
    _adjacent_vertex_inds: jnp.ndarray
    _loop_type: str
    _max_value: float
    _softmin_order: int
    _drelu_order: int
    _drelu_cutoff: float
    _max_num_iterations: int
    _initial_sites: corefunctions.InitialSites
    _tolerance: float | None
    _log_interval: int | None
    _logger: logging.Logger | None

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        mesh_data: corefunctions.MeshData,
        solver_data: SolverData,
        initial_sites: corefunctions.InitialSites,
        logger: logging.Logger | None = None,
    ):
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

        self._vertices = mesh_data.vertices
        self._adjacent_vertex_inds = mesh_data.adjacent_vertex_inds
        self._loop_type = solver_data.loop_type
        self._max_value = solver_data.max_value
        self._softmin_order = solver_data.softmin_order
        self._drelu_order = solver_data.drelu_order
        self._drelu_cutoff = solver_data.drelu_cutoff
        self._max_num_iterations = solver_data.max_num_iterations
        self._tolerance = solver_data.tolerance
        self._log_interval = solver_data.log_interval
        self._initial_sites = initial_sites
        self._logger = logger

    # ----------------------------------------------------------------------------------------------
    def run(
        self,
        tensor_field: jnp.ndarray,
    ) -> Solution:
        initial_guess = jnp.ones(self._vertices.shape[0]) * self._max_value
        initial_guess = initial_guess.at[self._initial_sites.inds].set(self._initial_sites.values)

        match self._loop_type:
            case "jitted_for":
                run_function = self._run_jitted_for_loop
            case "jitted_while":
                run_function = self._run_jitted_while_loop
            # Non-jitted while loop, additionally performs online logging
            case "nonjitted_while":
                run_function = self._run_nonjitted_while_loop
            case _:
                raise ValueError(f"Invalid loop type: {self._.loop_type}")

        solution_vector, num_iterations, tolerance = run_function(initial_guess, tensor_field)
        assert solution_vector.shape[0] == self._vertices.shape[0], (
            f"Solution vector needs to have shape {self._vertices.shape[0]}, "
            f"but has shape {solution_vector.shape}"
        )
        solution = Solution(
            values=solution_vector, num_iterations=num_iterations, tolerance=tolerance
        )
        return solution

    # ----------------------------------------------------------------------------------------------
    @eqx.filter_jit
    def _run_jitted_for_loop(
        self,
        initial_guess: jnp.ndarray,
        tensor_field: jnp.ndarray,
    ) -> tuple[jnp.ndarray, int, float]:
        def loop_body_for(_: int, carry_args: tuple) -> tuple:
            new_solution_vector, tolerance, old_solution_vector, tensor_field = carry_args
            old_solution_vector = new_solution_vector
            new_solution_vector = self._compute_global_update(old_solution_vector, tensor_field)
            tolerance = jnp.max(jnp.abs(new_solution_vector - old_solution_vector))
            return new_solution_vector, tolerance, old_solution_vector, tensor_field

        initial_tolerance = 0
        initial_old_solution = jnp.zeros(initial_guess.shape)
        solution_vector, tolerance, *_ = jax.lax.fori_loop(
            0,
            self._max_num_iterations,
            loop_body_for,
            (
                initial_guess,
                initial_tolerance,
                initial_old_solution,
                tensor_field,
            ),
        )
        return solution_vector, self._max_num_iterations, tolerance

    # ----------------------------------------------------------------------------------------------
    @eqx.filter_jit
    def _run_jitted_while_loop(
        self,
        initial_guess: jnp.ndarray,
        tensor_field: jnp.ndarray,
    ) -> tuple[jnp.ndarray, int, float]:
        if self._tolerance is None:
            raise ValueError("Tolerance threshold must be provided for while loop")

        def loop_body_while(carry_args: tuple) -> tuple:
            new_solution_vector, iteration_counter, tolerance, old_solution_vector, tensor_field = (
                carry_args
            )
            old_solution_vector = new_solution_vector
            new_solution_vector = self._compute_global_update(old_solution_vector, tensor_field)
            tolerance = jnp.max(jnp.abs(new_solution_vector - old_solution_vector))
            iteration_counter += 1
            return (
                new_solution_vector,
                iteration_counter,
                tolerance,
                old_solution_vector,
                tensor_field,
            )

        def cond_while(carry_args: tuple) -> bool:
            new_solution_vector, iteration_counter, tolerance, old_solution_vector, _ = carry_args
            tolerance = jnp.max(jnp.abs(new_solution_vector - old_solution_vector))
            return (tolerance > self._tolerance) & (
                iteration_counter < self._max_num_iterations
            )

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
                tensor_field,
            ),
        )
        return solution_vector, num_iterations, tolerance

    # ----------------------------------------------------------------------------------------------
    def _run_nonjitted_while_loop(
        self,
        initial_guess: jnp.ndarray,
        tensor_field: jnp.ndarray,
    ) -> tuple[jnp.ndarray, int, jnp.ndarray]:
        if self._tolerance is None:
            raise ValueError("Tolerance threshold must be provided for while loop")
        if self._log_interval is None:
            raise ValueError("Log interval must be provided for while loop")
        if self._logger is None:
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
        self._logger.header(log_values)

        while (tolerance > self._tolerance) and (
            iteration_counter < self._max_num_iterations
        ):
            new_solution_vector = self._compute_global_update(old_solution_vector, tensor_field)
            tolerance = jnp.max(jnp.abs(new_solution_vector - old_solution_vector))
            tolerance_vector.append(tolerance)
            old_solution_vector = new_solution_vector
            iteration_counter += 1

            if (iteration_counter % self._log_interval == 0) or (
                iteration_counter == self._max_num_iterations
            ):
                current_time = time.time() - start_time
                log_values["time"].value = current_time
                log_values["iters"].value = iteration_counter
                log_values["tol"].value = tolerance
                self._logger.log(log_values)

        tolerance_vector = jnp.array(tolerance_vector)
        return new_solution_vector, iteration_counter, tolerance_vector

    # ----------------------------------------------------------------------------------------------
    @eqx.filter_jit
    def _compute_global_update(
        self,
        solution_vector: jnp.ndarray,
        tensor_field: jnp.ndarray,
    ) -> jnp.ndarray:
        assert solution_vector.shape[0] == self._vertices.shape[0], (
            f"Solution vector needs to have shape {self._vertices.shape[0]}, "
            f"but has shape {solution_vector.shape}"
        )
        assert self._adjacent_vertex_inds.shape[0] == self._vertices.shape[0], (
            f"Adjacent simplex indix array needs to have shape {self._vertices.shape[0]}, "
            f"but has shape {self._adjacent_vertex_inds.shape[0]}"
        )
        global_update_function = jax.vmap(self._compute_vertex_update, in_axes=(None, 0, None))
        global_update = global_update_function(
            solution_vector,
            self._adjacent_vertex_inds,
            tensor_field,
        )

        assert global_update.shape == solution_vector.shape, (
            f"New solution has shape {global_update.shape}, "
            f"but should have shape {solution_vector.shape}"
        )
        return global_update

    # ----------------------------------------------------------------------------------------------
    def _compute_vertex_update(
        self,
        old_solution_vector: jnp.ndarray,
        adjacent_vertex_inds: jnp.ndarray,
        tensor_field: jnp.ndarray,
    ) -> float:
        vertex_update_candidates = corefunctions.compute_vertex_update_candidates(
            old_solution_vector,
            adjacent_vertex_inds,
            self._vertices,
            tensor_field,
            self._drelu_order,
            self._drelu_cutoff,
        )
        self_update = jnp.expand_dims(old_solution_vector[adjacent_vertex_inds[0, 0]], axis=-1)
        vertex_update_candidates = jnp.concatenate(
            (self_update, vertex_update_candidates.flatten())
        )

        min_value = jnp.min(vertex_update_candidates)
        vertex_update_candidates = jnp.where(
            vertex_update_candidates == min_value, vertex_update_candidates, jnp.inf
        )
        vertex_update = corefunctions.compute_softmin(
            vertex_update_candidates, min_value, self._softmin_order
        )
        assert (
            vertex_update.shape == ()
        ), f"Vertex update has to be scalar, but has shape {vertex_update.shape}"
        return vertex_update
