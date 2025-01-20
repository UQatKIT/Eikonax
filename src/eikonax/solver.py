"""Eikonax forward solver.

Classes:
    SolverData: Settings for the initialization of the Eikonax Solver.
    Solution: Eikonax solution object, returned by the solver.
    Solver: Eikonax solver class.
"""

import time
from dataclasses import dataclass
from numbers import Real
from typing import Annotated

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy.typing as npt
from beartype.vale import Is, IsEqual
from jaxtyping import Bool as jtBool
from jaxtyping import Float as jtFloat
from jaxtyping import Int as jtInt

from . import corefunctions, logging


# ==================================================================================================
@dataclass
class SolverData:
    """Settings for the initialization of the Eikonax Solver.

    See the [Forward Solver](../usage/solve.md) documentation for more detailed explanations.

    Args:
        loop_type (str): Type of loop for iterations,
            options are 'jitted_for', 'jitted_while', 'nonjitted_while'.
        max_value (Real): Maximum value for the initialization of the solution vector.
        use_soft_update: Flag for using soft minmax approximation for optimization parameters
        softminmax_order (int | None): Order of the soft minmax approximation for optimization
            parameters. Only required if `use_soft_update` is True.
        softminmax_cutoff (Real | None): Cutoff distance from [0,1] for the soft minmax function.
            Only required if `use_soft_update` is True.
        max_num_iterations (int): Maximum number of iterations after which to terminate the solver.
            Required for all loop types
        tolerance (Real): Absolute difference between iterates in supremum norm, after which to
            terminate solver. Required for while loop types
        log_interval (int): Iteration interval after which log info is written. Required for
            non-jitted while loop type.
    """

    loop_type: Annotated[
        str, IsEqual["jitted_for"] | IsEqual["jitted_while"] | IsEqual["nonjitted_while"]
    ]
    max_value: Annotated[Real, Is[lambda x: x > 0]]
    use_soft_update: bool
    softminmax_order: Annotated[int, Is[lambda x: x > 0]] | None
    softminmax_cutoff: Annotated[Real, Is[lambda x: x > 0]] | None
    max_num_iterations: Annotated[int, Is[lambda x: x > 0]]
    tolerance: Annotated[Real, Is[lambda x: x > 0]] | None = None
    log_interval: Annotated[int, Is[lambda x: x > 0]] | None = None


@dataclass
class Solution:
    """Eikonax solution object, returned by the solver.

    See the [Forward Solver](../usage/solve.md) documentation for more detailed explanations.

    Args:
        values (jax.Array): Actual solution vector.
        num_iterations (int): Number of iterations performed in the solve.
        tolerance (float | jax.Array): Tolerance from last two iterates, or entire tolerance history
    """

    values: jtFloat[jax.Array, "num_vertices"]
    num_iterations: int | jtInt[jax.Array, ""]
    tolerance: float | jtFloat[jax.Array, "..."] | None = None


# ==================================================================================================
class Solver(eqx.Module):
    r"""Eikonax solver class.

    The solver class is the main component for computing the solution $u$ of the Eikonal equation
    for given geometry $\Omega$, tensor field $\mathbf{M}$, and initial sites $\Gamma$,

    $$
    \begin{gather*}
    \sqrt{\big(\nabla u(\mathbf{x}),\mathbf{M}(\mathbf{x})\nabla u(\mathbf{x})\big)} = 1,\quad
    \mathbf{x}\in\Omega, \\
    \nabla u(\mathbf{x}) \cdot \mathbf{n}(\mathbf{x}) \geq 0,\quad \mathbf{x}\in\partial\Omega, \\
    u(\mathbf{x}_0) = u_0,\quad \mathbf{x}_0 \in \Gamma.
    \end{gather*}
    $$

    On the discrete level, the solver perform global Jacobi iterations of the form

    $$
        \mathbf{u}^{(j+1)} = \mathbf{G}(\mathbf{u}^{(j)}),
    $$

    where $\mathbf{G}$ is the global update function, derived from Godunov-type upwinding
    principles. The solver can either be run with a fixed number of iterations, or until a
    user-defined tolerance for the difference between two consecutive iterates in supremum norm is
    undercut.

    The Eikonax solver works on the vertex level,
    meaning that it considers updates from all adjacent triangles to a vertex, instead of all
    updates for all vertices per triangle. This allows to establish causality in the final solution,
    which is important for the efficient computation of parametric derivatives.
    The solver class is mainly a wrapper around different loop constructs, which call vectorized
    forms of the methods implemented in the [`corefunctions`][eikonax.corefunctions] module. These
    loop constructs evolve around the loop functionality provided by JAX.

    Methods:
        run: Main interface for Eikonax runs.
    """

    # Equinox modules are data classes, so specify attributes on class level
    _vertices: jax.Array
    _adjacency_data: jax.Array
    _loop_type: str
    _max_value: Real
    _use_soft_update: bool
    _softminmax_order: int | None
    _softminmax_cutoff: Real | None
    _max_num_iterations: int
    _initial_site_inds: jax.Array
    _initial_site_values: jax.Array
    _tolerance: Real | None
    _log_interval: int | None
    _logger: logging.Logger | None

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        mesh_data: corefunctions.MeshData,
        solver_data: SolverData,
        initial_sites: corefunctions.InitialSites,
        logger: logging.Logger | None = None,
    ) -> None:
        """Constructor of the solver class.

        The constructor initializes all data structures that are re-used in many-query scenarios,
        such as the solution of inverse problems.

        Args:
            mesh_data (corefunctions.MeshData): Vertex-based mesh data.
            solver_data (SolverData): Settings for the solver.
            initial_sites (corefunctions.InitialSites): vertices and values for source points
            logger (logging.Logger | None, optional): Logger object, only required for non-jitted
                while loops. Defaults to None.
        """
        self._vertices = mesh_data.vertices
        self._adjacency_data = mesh_data.adjacency_data
        self._loop_type = solver_data.loop_type
        self._max_value = solver_data.max_value
        self._use_soft_update = solver_data.use_soft_update
        self._softminmax_order = solver_data.softminmax_order
        self._softminmax_cutoff = solver_data.softminmax_cutoff
        self._max_num_iterations = solver_data.max_num_iterations
        self._tolerance = solver_data.tolerance
        self._log_interval = solver_data.log_interval
        self._initial_site_inds = initial_sites.inds
        self._initial_site_values = initial_sites.values
        self._logger = logger

    # ----------------------------------------------------------------------------------------------
    def run(
        self,
        tensor_field: jtFloat[jax.Array | npt.NDArray, "num_simplices dim dim"],
    ) -> Solution:
        """Main interface for conducting solver runs.

        The method initializes the solution vector and dispatches to the run method for the
        selected loop type.

        !!! note
            The derivator expects the metric tensor field as used in the inner product for the
            update stencil of the eikonal equation. This is the **INVERSE** of the conductivity
            tensor, which is the actual tensor field in the eikonal equation. The
            [`Tensorfield`][eikonax.tensorfield.TensorField] component provides the inverse tensor
            field.

        Args:
            tensor_field (jax.Array): Parameter field for which to solve the Eikonal equation.
                Provides an anisotropy tensor for each simplex of the mesh.

        Raises:
            ValueError: Checks that the chosen loop type is valid.

        Returns:
            Solution: Eikonax solution object.
        """
        tensor_field = jnp.array(tensor_field, dtype=jnp.float32)
        initial_guess = jnp.ones(self._vertices.shape[0]) * self._max_value
        initial_guess = initial_guess.at[self._initial_site_inds].set(self._initial_site_values)

        match self._loop_type:
            case "jitted_for":
                run_function = self._run_jitted_for_loop
            case "jitted_while":
                run_function = self._run_jitted_while_loop
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
        initial_guess: jtFloat[jax.Array, "num_vertices"],
        tensor_field: jtFloat[jax.Array, "num_vertices dim dim"],
    ) -> tuple[jtFloat[jax.Array, "num_vertices"], int, float]:
        """Solver run with jitted for loop for iterations.

        The method constructs a JAX-type for loop with fixed number of iterations. For every
        iteration, a new solution vector is computed from the
        [`_compute_global_update`][eikonax.solver.Solver._compute_global_update] method.

        Args:
            initial_guess (jax.Array): Initial solution vector
            tensor_field (jax.Array): Parameter field

        Returns:
            tuple[jax.Array, int, float]: Solution values, number of iterations, tolerance
        """

        # JAX body for for loop, has to carry over all args
        def loop_body_for(_: jtInt[jax.Array, ""], carry_args: tuple) -> tuple:
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
        initial_guess: jtFloat[jax.Array, "num_vertices"],
        tensor_field: jtFloat[jax.Array, "num_vertices dim dim"],
    ) -> tuple[jtFloat[jax.Array, "num_vertices"], int, float]:
        """Solver run with jitted while loop for iterations.

        The iterator is tolerance-based, terminating after a user-defined tolerance for the
        difference between two consecutive iterates in supremum norm is undercut. For every
        iteration, a new solution vector is computed from the
        [`_compute_global_update`][eikonax.solver.Solver._compute_global_update] method.

        Args:
            initial_guess (jax.Array): Initial solution vector
            tensor_field (jax.Array): Parameter field

        Raises:
            ValueError: Checks that tolerance has been provided by the user

        Returns:
            tuple[jax.Array, int, float]: Solution values, number of iterations, tolerance
        """
        if self._tolerance is None:
            raise ValueError("Tolerance threshold must be provided for while loop")

        # JAX body for while loop, has to carry over all args
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

        # JAX termination condition for while loop
        def cond_while(carry_args: tuple) -> jtBool[jax.Array, ""]:
            new_solution_vector, iteration_counter, tolerance, old_solution_vector, _ = carry_args
            tolerance = jnp.max(jnp.abs(new_solution_vector - old_solution_vector))
            return (tolerance > self._tolerance) & (iteration_counter < self._max_num_iterations)

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
        initial_guess: jtFloat[jax.Array, "num_vertices"],
        tensor_field: jtFloat[jax.Array, "num_vertices dim dim"],
    ) -> tuple[jtFloat[jax.Array, "num_vertices"], int, jtFloat[jax.Array, "..."]]:
        """Solver run with standard Python while loop for iterations.

        While being less performant, the Python while loop allows for logging of infos between
        iterations. The iterator is tolerance-based, terminating after a user-defined tolerance for
        the difference between two consecutive iterates in supremum norm is undercut. For every
        iteration, a new solution vector is computed from the
        [`_compute_global_update`][eikonax.solver.Solver._compute_global_update] method.

        Args:
            initial_guess (jax.Array): Initial solution vector
            tensor_field (jax.Array): Parameter field

        Raises:
            ValueError: Checks that tolerance has been provided by the user
            ValueError: Checks that log interval has been provided by the user
            ValueError: Checks that logger object has been provided by the user

        Returns:
            tuple[jax.Array, int, int]: Solution values, number of iterations, tolerance
                vector over all iterations
        """
        if self._tolerance is None:
            raise ValueError("Tolerance threshold must be provided for while loop")
        if self._log_interval is None:
            raise ValueError("Log interval must be provided for non-jitted while loop")
        if self._logger is None:
            raise ValueError("Logger must be provided for non-jitted while loop")

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
        self._logger.header(log_values.values())

        while (tolerance > self._tolerance) and (iteration_counter < self._max_num_iterations):
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
                self._logger.log(log_values.values())

        tolerance_vector = jnp.array(tolerance_vector)
        return new_solution_vector, iteration_counter, tolerance_vector

    # ----------------------------------------------------------------------------------------------
    @eqx.filter_jit
    def _compute_global_update(
        self,
        solution_vector: jtFloat[jax.Array, "num_vertices"],
        tensor_field: jtFloat[jax.Array, "num_vertices dim dim"],
    ) -> jtFloat[jax.Array, "num_vertices"]:
        """Given a current state and tensor field, compute a new solution vector.

        This method is basically a vectorized call to the
        [`_compute_vertex_update`][eikonax.solver.Solver._compute_vertex_update] method, evaluated
        over all vertices of the mesh.

        Args:
            solution_vector (jax.Array): Current state
            tensor_field (jax.Array): Parameter field

        Returns:
            jax.Array: New iterate
        """
        assert solution_vector.shape[0] == self._vertices.shape[0], (
            f"Solution vector needs to have shape {self._vertices.shape[0]}, "
            f"but has shape {solution_vector.shape}"
        )
        global_update_function = jax.vmap(self._compute_vertex_update, in_axes=(None, None, 0))
        global_update = global_update_function(
            solution_vector,
            tensor_field,
            self._adjacency_data,
        )
        assert global_update.shape == solution_vector.shape, (
            f"New solution has shape {global_update.shape}, "
            f"but should have shape {solution_vector.shape}"
        )
        return global_update

    # ----------------------------------------------------------------------------------------------
    def _compute_vertex_update(
        self,
        old_solution_vector: jtFloat[jax.Array, "num_vertices"],
        tensor_field: jtFloat[jax.Array, "num_vertices dim dim"],
        adjacency_data: jtInt[jax.Array, "max_num_adjacent_simplices 4"],
    ) -> jtFloat[jax.Array, ""]:
        """Compute the update value for a single vertex.

        This method links to the main logic of the solver routine, based on functions in the
        [`corefunctions`][eikonax.corefunctions] module.

        Args:
            old_solution_vector (jax.Array): Current state
            tensor_field (jax.Array): Parameter field
            adjacency_data (jax.Array): Info on all adjacent triangles and respective vertices
                for the current vertex

        Returns:
            jax.Array: Optimal update value for the current vertex
        """
        vertex_update_candidates = corefunctions.compute_vertex_update_candidates(
            old_solution_vector,
            tensor_field,
            adjacency_data,
            self._vertices,
            self._use_soft_update,
            self._softminmax_order,
            self._softminmax_cutoff,
        )
        self_update = jnp.expand_dims(old_solution_vector[adjacency_data[0, 0]], axis=-1)
        vertex_update_candidates = jnp.concatenate(
            (self_update, vertex_update_candidates.flatten())
        )
        vertex_update = jnp.min(vertex_update_candidates)
        assert (
            vertex_update.shape == ()
        ), f"Vertex update has to be scalar, but has shape {vertex_update.shape}"
        return vertex_update
