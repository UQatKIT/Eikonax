"""Core functions for Eikonax forward solves and parametric derivatives.

This module contains atomic functions that make up the Eikonax solver routines. They (and their
automatic derivatives computed with JAX) are further used to evaluate parametric derivatives.

Classes:
    MeshData: Data characterizing a computational mesh from a vertex-centered perspective.
    InitialSites: Initial site info.

Functions:
    compute_softmin: Numerically stable computation of the softmin function based on the Boltzmann
        operator.
    compute_softminmax: Smooth double ReLU-type approximation that restricts a variable to the
        interval [0, 1].
    compute_edges: Compute the edges of a triangle from vertex indices and coordinates.
    compute_optimal_update_parameters_soft: Compute position parameter for update of a node within a
        specific triangle.
    compute_optimal_update_parameters_hard: Compute position parameter for update of a node within a
        specific triangle.
    _compute_optimal_update_parameters: Compute the optimal update parameter for the solution of the
        Eikonal equation.
    compute_fixed_update: Compute update for a given vertex, triangle, and update parameter.
    compute_update_candidates_from_adjacent_simplex: Compute all possible update candidates from an
        adjacent triangle.
    compute_vertex_update_candidates: Compute all update candidates for a given vertex.
    grad_softmin: The gradient of the softmin function requires further masking of infeasible
        values.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

import jax
import jax.numpy as jnp
import numpy.typing as npt
from jaxtyping import Float as jtFloat
from jaxtyping import Int as jtInt
from jaxtyping import Real as jtReal


# ==================================================================================================
@dataclass
class MeshData:
    """Data characterizing a computational mesh from a vertex-centered perspective.

    Attributes:
        vertices (jax.Array | npt.NDArray): The coordinates of the vertices in the mesh.
            The dimension of this array is `(num_vertices, dim)`, where num_vertices is the number
            of vertices in the mesh and dim is the dimension of the space in which the mesh is
            embedded.
        adjacency_data (jax.Array | npt.NDArray): Adjacency data for each vertex. This is the list
            of adjacent triangles, together with the two vertices that span the respective triangle
            with the current vertex. The dimension of this array is
            `(num_vertices, max_num_adjacent_simplices, 4)`, where max_num_adjacent_simplices is the
            maximum number of simplices that are adjacent to a vertex in the mesh. All entries have
            this maximum size, as JAX only operates on homogeneous data structures. If a vertex has
            fewer than max_num_adjacent_simplices adjacent simplices, the remaining entries are
            filled with -1.
    """

    vertices: jtFloat[jax.Array | npt.NDArray, "num_vertices dim"]
    adjacency_data: jtInt[jax.Array | npt.NDArray, "num_vertices max_num_adjacent_simplices 4"]

    def __post_init__(self) -> None:
        """Convert to jax arrays."""
        self.vertices = jnp.array(self.vertices, dtype=jnp.float32)
        self.adjacency_data = jnp.array(self.adjacency_data, dtype=jnp.int32)


@dataclass
class InitialSites:
    """Initial site info.

    For a unique solution of the state-constrained Eikonal equation, the solution values need to be
    given a number of initial points (at least one). Multiple initial sites need to be compatible,
    in the sense that the arrival time from another source is not smaller than the initial value
    itself.

    Attributes:
        inds (jax.Array | npt.NDArray): The indices of the nodes where the initial sites are placed.
        values (jax.Array | npt.NDArray): The values of the initial sites.
    """

    inds: jtInt[jax.Array | npt.NDArray, "num_initial_sites"] | Iterable
    values: jtFloat[jax.Array | npt.NDArray, "num_initial_sites"] | Iterable

    def __post_init__(self) -> None:
        """Convert to jax arrays."""
        self.inds = jnp.array(self.inds, dtype=jnp.int32)
        self.values = jnp.array(self.values, dtype=jnp.float32)


# ==================================================================================================
def compute_softmin(
    args: jtReal[jax.Array, "..."], min_arg: jtReal[jax.Array, ""], order: int
) -> jtFloat[jax.Array, ""]:
    """Numerically stable computation of the softmin function based on the Boltzmann operator.

    !!! danger "To be rewritten"
        The softmin function applied to only optimal update values simply amounts to the average
        of this solution candidates, meaning we deal with a sub-gradient here. Will be changed in a
        future version.

    This softmin function is applied to actual minimum values, meaning it does not have a purpose
    on the evaluation level. It renders the minimum computation differentiable, however.
    Importantly, the Boltzmann softmin is value preserving, meaning that the solution of the eikonal
    equation is the same as for a hard minimum. As JAX works on homogeneous arrays only, non-minimal
    values are also passed to this function. They are assumed to be masked as jnp.inf, which are
    handled in a numerically stable way by this routine.

    Args:
        args (jax.Array): Values to compute the soft minimum over
        min_arg (jax.Array): The actual value of the minimum argument, necessary for numerical
            stability
        order (int): Approximation order of the softmin function

    Returns:
        jax.Array: Soft minimum value
    """
    arg_diff = min_arg - args
    nominator = jnp.where(args == jnp.inf, 0, args * jnp.exp(order * arg_diff))
    denominator = jnp.where(args == jnp.inf, 0, jnp.exp(order * arg_diff))
    soft_value = jnp.sum(nominator) / jnp.sum(denominator)
    return soft_value


# --------------------------------------------------------------------------------------------------
def compute_softminmax(value: jtReal[jax.Array, "..."], order: int) -> jtFloat[jax.Array, "..."]:
    r"""Smooth double ReLU-type approximation that restricts a variable to the interval $[0, 1]$.

    !!! bug "Unstable AD"
        While the method itself is stable, its derivative computed with JAX is not for higher orders
        $\kappa$

    Given an input value $x$ and an order parameter $\kappa > 0$, the method performs a
    differentiable transformation according to

    $$
    \begin{gather*}
        f_{\text{lb}}(x) = \frac{1}{\kappa}\log\Big[ 1 + \exp\big( \kappa x \big) \Big], \\
        f_{\text{ub}}(x) = 1 - \frac{1}{\kappa}\log\Big[ 1 + \exp\big( -\kappa(x-1) \big) \Big], \\
        \tilde{\phi}(x) = f_{\text{ub}}(f_{\text{lb}}(x)), \\
        \phi_{\text{lb}} = 1 - \frac{\log\big(1+\exp(\kappa)\big)}{\kappa} < 0, \\
        \phi(x) = \frac{\tilde{\phi}(x) - \phi_{\text{lb}}}{1-\phi_{\text{lb}}}.
    \end{gather*}
    $$

    The method is numerically stable, obeys the value range, and does not introduce any new extrema.

    Args:
        value (jax.Array): variable to restrict to range [0,1]
        order (int): Approximation order of the smooth approximation

    Returns:
        jax.Array: Smoothed/restricted value
    """
    lower_bound = -jnp.log(1 + jnp.exp(-order)) / order
    soft_value = jnp.where(
        value <= 0,
        jnp.log(1 + jnp.exp(order * value)) / order,
        value + jnp.log(1 + jnp.exp(-order * value)) / order,
    )
    soft_value = jnp.where(
        soft_value <= 1,
        soft_value - jnp.log(1 + jnp.exp(order * (soft_value - 1))) / order,
        1 - jnp.log(1 + jnp.exp(-order * (soft_value - 1))) / order,
    )
    soft_value = (soft_value - lower_bound) / (1 - lower_bound)
    return soft_value


# --------------------------------------------------------------------------------------------------
def compute_edges(
    i: jtInt[jax.Array, ""],
    j: jtInt[jax.Array, ""],
    k: jtInt[jax.Array, ""],
    vertices: jtFloat[jax.Array, "num_vertices dim"],
) -> tuple[jtFloat[jax.Array, "dim"], jtFloat[jax.Array, "dim"], jtFloat[jax.Array, "dim"]]:
    """Compute the edges of a triangle from vertex indices and coordinates.

    Args:
        i (jax.Array): First vertex index of a triangle
        j (jax.Array): Second vertex index of a triangle
        k (jax.Array): Third vertex index of a triangle
        vertices (jax.Array): jax.Array of all vertex coordinates

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: Triangle edge vectors
    """
    e_ji = vertices[i] - vertices[j]
    e_ki = vertices[i] - vertices[k]
    e_jk = vertices[k] - vertices[j]
    return e_ji, e_ki, e_jk


# --------------------------------------------------------------------------------------------------
def compute_optimal_update_parameters_soft(
    solution_values: jtFloat[jax.Array, "2"],
    parameter_tensor: jtFloat[jax.Array, "dim dim"],
    edges: tuple[jtFloat[jax.Array, "dim"], jtFloat[jax.Array, "dim"], jtFloat[jax.Array, "dim"]],
    softminmax_order: int,
    softminmax_cutoff: Real,
) -> jtFloat[jax.Array, "4"]:
    """Compute position parameter for update of a node within a specific triangle.

    For a given vertex $i$ and adjacent triangle, we compute the update for the solution of the
    Eikonal as propagating from a point on the connecting edge of the opposite vertices $j$ and $k$.
    We thereby assume the solution value to vary linearly on that edge. The update parameter in
    $[0,1]$ indicates the optimal linear combination of the solution values at $j$ and $k$, in the
    sense that the solution value at $i$ is minimized. As the underlying optimization problem is
    constrained, we compute the solutions of the unconstrained problem, as well as the boundary
    values. The former are constrained to the feasible region [0,1] by the soft minmax function
    implemented in [`compute_softminmax`][eikonax.corefunctions.compute_softminmax].
    We further clip values lying to far outside the feasible region, by masking them with value -1.
    This function is a wrapper, for the unconstrained solution values, it calls the implementation
    function
    [`_compute_optimal_update_parameters`][eikonax.corefunctions._compute_optimal_update_parameters].

    Args:
        solution_values (jax.Array): Current solution values, as per the previous iteration
        parameter_tensor (jax.Array): Parameter tensor for the current triangle
        edges (tuple[jax.Array, jax.Array, jax.Array]): Edge vectors of the triangle under
            consideration
        softminmax_order (int): Order of the soft minmax function to be employed
        softminmax_cutoff (int): Cutoff value beyond parameter values are considered infeasible
            and masked with -1

    Returns:
        jax.Array: All possible candidates for the update parameter, according to the solution
            of the constrained optimization problem
    """
    lambda_1, lambda_2 = _compute_optimal_update_parameters(
        solution_values, parameter_tensor, edges
    )
    lambda_1_clipped = compute_softminmax(lambda_1, softminmax_order)
    lambda_2_clipped = compute_softminmax(lambda_2, softminmax_order)
    lower_bound = -softminmax_cutoff
    upper_bound = 1 + softminmax_cutoff

    lambda_1_clipped = jnp.where(
        (lambda_1 < lower_bound) | (lambda_1 > upper_bound), -1, lambda_1_clipped
    )
    lambda_2_clipped = jnp.where(
        (lambda_2 < lower_bound) | (lambda_2 > upper_bound) | (lambda_2 == lambda_1),
        -1,
        lambda_2_clipped,
    )
    lambda_array = jnp.array((0, 1, lambda_1_clipped, lambda_2_clipped), dtype=jnp.float32)
    return lambda_array


# --------------------------------------------------------------------------------------------------
def compute_optimal_update_parameters_hard(
    solution_values: jtFloat[jax.Array, "2"],
    parameter_tensor: jtFloat[jax.Array, "dim dim"],
    edges: tuple[jtFloat[jax.Array, "dim"], jtFloat[jax.Array, "dim"], jtFloat[jax.Array, "dim"]],
) -> jtFloat[jax.Array, "4"]:
    """Compute position parameter for update of a node within a specific triangle.

    For a given vertex $i$ and adjacent triangle, we compute the update for the solution of the
    Eikonal as propagating from a point on the connecting edge of the opposite vertices $j$ and $k$.
    We thereby assume the solution value to vary linearly on that dge. The update parameter in
    $[0,1]$ indicates the optimal linear combination of the solution values at $j$ and $k$, in the
    sense that the solution value at $i$ is minimized. As the underlying optimization problem is
    constrained, we compute the solutions of the unconstrained problem, as well as the boundary
    values. The former are constrained to the feasible region $[0,1]$ by a simple cutoff.
    This function is a wrapper, for the unconstrained solution values, it calls the implementation
    function
    [`_compute_optimal_update_parameters`][eikonax.corefunctions._compute_optimal_update_parameters].

    Args:
        solution_values (jax.Array): Current solution values, as per the previous iteration
        parameter_tensor (jax.Array): Parameter tensor for the current triangle
        edges (tuple[jax.Array, jax.Array, jax.Array]): Edge vectors of the triangle under
            consideration

    Returns:
        jax.Array: All possible candidates for the update parameter, according to the solution
            of the constrained optimization problem
    """
    lambda_1, lambda_2 = _compute_optimal_update_parameters(
        solution_values, parameter_tensor, edges
    )
    lambda_1_clipped = jnp.where((lambda_1 <= 0) | (lambda_1 >= 1), -1, lambda_1)
    lambda_2_clipped = jnp.where(
        (lambda_2 <= 0) | (lambda_2 >= 1) | (lambda_2 == lambda_1),
        -1,
        lambda_2,
    )
    lambda_array = jnp.array((0, 1, lambda_1_clipped, lambda_2_clipped), dtype=jnp.float32)
    return lambda_array


# --------------------------------------------------------------------------------------------------
def _compute_optimal_update_parameters(
    solution_values: jtFloat[jax.Array, "2"],
    parameter_tensor: jtFloat[jax.Array, "dim dim"],
    edges: tuple[jtFloat[jax.Array, "dim"], jtFloat[jax.Array, "dim"], jtFloat[jax.Array, "dim"]],
) -> tuple[jtFloat[jax.Array, ""], jtFloat[jax.Array, ""]]:
    r"""Compute the optimal update parameter for the solution of the Eikonal equation.

    The function works for the update within a given triangle. The solutions of the unconstrained
    minimization problem are given as the roots of a quadratic polynomial. For the local metric
    tensor $M_s$ for the given simplex, the set of known solution values $\mathbf{U}_{jk}$, and the
    simplex edges $\mathbf{E}_{ijk}$, we have that

    $$
        \lambda_{1/2}(\mathbf{M}_{s},\mathbf{U}_{jk},\mathbf{E}_{ijk}) =
        \frac{(\mathbf{e}_{jk},\mathbf{M}_{s}^{-1}\mathbf{e}_{ki})
        \pm c(\mathbf{M}_{s},\mathbf{U}_{jk},\mathbf{E}_{ijk})}{(\mathbf{e}_{jk},
        \mathbf{M}_{s}^{-1}\mathbf{e}_{jk})},
    $$

    with

    $$
        c(\mathbf{M}_{s},\mathbf{U}_{jk},\mathbf{E}_{ijk}) =
        (u_k - u_j) \sqrt{\frac{(\mathbf{e}_{jk},\mathbf{M}_{s}^{-1}\mathbf{e}_{jk})
        (\mathbf{e}_{ki},\mathbf{M}_{s}^{-1}\mathbf{e}_{ki})
        - (\mathbf{e}_{jk},\mathbf{M}_{s}^{-1}\mathbf{e}_{ki})^2}
        {(\mathbf{e}_{jk},\mathbf{M}_{s}^{-1}\mathbf{e}_{jk}) - (u_k - u_j)^2}}.
    $$

    These roots may or may not lie inside the feasible region $[0,1]$.The function returns both
    solutions, which are then further processed in the calling wrapper.
    """
    u_j, u_k = solution_values
    e_ji, _, e_jk = edges
    delta_u = u_k - u_j
    a_1 = jnp.dot(e_jk, parameter_tensor @ e_jk)
    a_2 = jnp.dot(e_jk, parameter_tensor @ e_ji)
    a_3 = jnp.dot(e_ji, parameter_tensor @ e_ji)

    nominator = a_1 * a_3 - a_2**2
    denominator = a_1 - delta_u**2
    # Treat imaginary roots as inf
    sqrt_term = jnp.where(denominator > 0, jnp.sqrt(nominator / denominator), jnp.inf)
    c = delta_u * sqrt_term

    lambda_1 = (a_2 + c) / a_1
    lambda_2 = (a_2 - c) / a_1
    return lambda_1, lambda_2


# --------------------------------------------------------------------------------------------------
def compute_fixed_update(
    solution_values: jtFloat[jax.Array, "2"],
    parameter_tensor: jtFloat[jax.Array, "dim dim"],
    lambda_value: jtFloat[jax.Array, ""],
    edges: tuple[jtFloat[jax.Array, "dim"], jtFloat[jax.Array, "dim"], jtFloat[jax.Array, "dim"]],
) -> jtFloat[jax.Array, ""]:
    r"""Compute update for a given vertex, triangle, and update parameter.

    The update value is given by the solution at a point  on the edge between the opposite vertices,
    plus the travel time from that point to the vertices under consideration. For a local metric
    tensor $M_s$ for the given simplex, the set of known solution values $\mathbf{U}_{jk}$, the
    simplex edges $\mathbf{E}_{ijk}$, and an optimal update parameter $\lambda$, it reads

    $$
        G_{\text{fix}}(\lambda,\mathbf{M}_s,\mathbf{U}_{jk}, \mathbf{E}_{ijk}) =
        u_j + \lambda (u_k-u_j) + \sqrt{\big((\mathbf{e}_{ji} - \lambda\mathbf{e}_{jk}),
        \mathbf{M}_{s}^{-1}(\mathbf{e}_{ji} - \lambda\mathbf{e}_{jk})\big)}
    $$

    Args:
        solution_values (jax.Array): Current solution values at opposite vertices j and k,
            as per the previous iteration
        parameter_tensor (jax.Array): Conductivity tensor for the current triangle
        lambda_value (jax.Array): Optimal update parameter
        edges (tuple[jax.Array, jax.Array, jax.Array]): Edge vectors of the triangle under
            consideration

    Returns:
        jax.Array: Updated solution value for the vertex under consideration
    """
    u_j, u_k = solution_values
    e_ji, _, e_jk = edges
    diff_vector = e_ji - lambda_value * e_jk
    transport_term = jnp.sqrt(jnp.dot(diff_vector, parameter_tensor @ diff_vector))
    update = lambda_value * u_k + (1 - lambda_value) * u_j + transport_term
    return update


# --------------------------------------------------------------------------------------------------
def compute_update_candidates_from_adjacent_simplex(
    old_solution_vector: jtFloat[jax.Array, "num_vertices"],
    tensor_field: jtFloat[jax.Array, "num_simplices dim dim"],
    adjacency_data: jtInt[jax.Array, "max_num_adjacent_simplices"],
    vertices: jtFloat[jax.Array, "num_vertices dim"],
    use_soft_update: bool,
    softminmax_order: int | None,
    softminmax_cutoff: Real | None,
) -> tuple[jtFloat[jax.Array, "4"], jtFloat[jax.Array, "4"]]:
    """Compute all possible update candidates from an adjacent triangle.

    Given a vertex and an adjacent triangle, this method computes all optimal update parameter
    candidates and the corresponding update values. To obey JAX's homogeneous array requirement,
    update values are also computed for infeasible parameter values, and have to be masked in the
    calling routine. This methods basically collects all results from the
    [`compute_optimal_update_parameters`][eikonax.corefunctions.compute_optimal_update_parameters_soft]
    and [`compute_fixed_update`][eikonax.corefunctions.compute_fixed_update] functions.

    Args:
        old_solution_vector (jax.Array): Given solution vector, as per a previous iteration
        tensor_field (jax.Array): Array of all tensor fields
        adjacency_data (jax.Array): Index of one adjaccent triangle and corresponding vertices
        vertices (jax.Array): Array of all vertex coordinates
        use_soft_update (bool): Flag to indicate whether to use a soft update or a hard update
        softminmax_order (int | None): Order of the soft minmax function for the update parameter,
            see `compute_softminmax`. Only required for `use_soft_update=True`
        softminmax_cutoff (Real | None): Cutoff value for the soft minmax computation, see
            `compute_optimal_update_parameters_soft`. Only required for `use_soft_update=True`

    Returns:
        tuple[jax.Array, jax.Array]: Update values and parameter candidates from the given
            triangle
    """
    i, j, k, s = adjacency_data
    solution_values = jnp.array((old_solution_vector[j], old_solution_vector[k]), dtype=jnp.float32)
    edges = compute_edges(i, j, k, vertices)
    parameter_tensor = tensor_field[s]
    if use_soft_update:
        lambda_array = compute_optimal_update_parameters_soft(
            solution_values, parameter_tensor, edges, softminmax_order, softminmax_cutoff
        )
    else:
        lambda_array = compute_optimal_update_parameters_hard(
            solution_values, parameter_tensor, edges
        )
    update_candidates = jnp.zeros(4)

    for i, lambda_candidate in enumerate(lambda_array):
        update = compute_fixed_update(solution_values, parameter_tensor, lambda_candidate, edges)
        update_candidates = update_candidates.at[i].set(update)
    return update_candidates, lambda_array


# --------------------------------------------------------------------------------------------------
def compute_vertex_update_candidates(
    old_solution_vector: jtFloat[jax.Array, "num_vertices"],
    tensor_field: jtFloat[jax.Array, "num_simplices dim dim"],
    adjacency_data: jtInt[jax.Array, "max_num_adjacent_simplices 4"],
    vertices: jtFloat[jax.Array, "num_vertices dim"],
    use_soft_update: bool,
    softminmax_order: int,
    softminmax_cutoff: Real,
) -> jtFloat[jax.Array, "max_num_adjacent_simplices 4"]:
    """Compute all update candidates for a given vertex.

    This method combines all updates from adjacent triangles to a given vertex, as computed in the
    function
    [`compute_update_candidates_from_adjacent_simplex`][eikonax.corefunctions.compute_update_candidates_from_adjacent_simplex].
    Infeasible candidates are masked with `jnp.inf`.

    Args:
        old_solution_vector (jax.Array): Given solution vector, as per a previous iteration
        tensor_field (jax.Array): jax.Array of all tensor fields
        adjacency_data (jax.Array): Data of all adjacent triangles and corresponding vertices
        vertices (jax.Array): jax.Array of all vertex coordinates
        use_soft_update (bool): Flag to indicate whether to use a soft update or a hard update
        softminmax_order (int | None): Order of the soft minmax function for the update parameter,
            see `compute_softminmax`. Only required for `use_soft_update=True`
        softminmax_cutoff (Real | None): Cutoff value for the soft minmax computation, see
            `compute_optimal_update_parameters_soft`. Only required for `use_soft_update=True`

    Returns:
        jax.Array: All possible update values for the given vertex, infeasible vertices are masked
            with jnp.inf
    """
    max_num_adjacent_simplices = adjacency_data.shape[0]
    vertex_update_candidates = jnp.zeros((max_num_adjacent_simplices, 4), dtype=jnp.float32)
    lambda_arrays = jnp.zeros((max_num_adjacent_simplices, 4), dtype=jnp.float32)

    for i, indices in enumerate(adjacency_data):
        simplex_update_candidates, lambda_array_candidates = (
            compute_update_candidates_from_adjacent_simplex(
                old_solution_vector,
                tensor_field,
                indices,
                vertices,
                use_soft_update,
                softminmax_order,
                softminmax_cutoff,
            )
        )
        vertex_update_candidates = vertex_update_candidates.at[i, :].set(simplex_update_candidates)
        lambda_arrays = lambda_arrays.at[i, :].set(lambda_array_candidates)

    # Mask infeasible updates
    # 1. Not an adjacent triangle, only buffer/fill value in adjacency data
    # 2. Infeasible update parameter, indicated with -1
    active_simplex_inds = adjacency_data[:, 3]
    vertex_update_candidates = jnp.where(
        (active_simplex_inds[..., None] != -1) & (lambda_arrays != -1),
        vertex_update_candidates,
        jnp.inf,
    )
    return vertex_update_candidates


# ==================================================================================================
"""Derivatives of elementary function based on JAX's AD capabilities."""
# Derivative of update value function w.r.t. current solution values, 1x2
grad_update_solution = jax.grad(compute_fixed_update, argnums=0)
# Deritative of update value function w.r.t. parameter tensor, 1xDxD
grad_update_parameter = jax.grad(compute_fixed_update, argnums=1)
# Derivative of update value function w.r.t. update parameter, 1x1
grad_update_lambda = jax.grad(compute_fixed_update, argnums=2)
# Derivative of update parameter function w.r.t. current solution values, 2x2
jac_lambda_soft_solution = jax.jacobian(compute_optimal_update_parameters_soft, argnums=0)
jac_lambda_hard_solution = jax.jacobian(compute_optimal_update_parameters_hard, argnums=0)
# Derivative of update parameter function w.r.t. parameter tensor, 2xDxD
jac_lambda_soft_parameter = jax.jacobian(compute_optimal_update_parameters_soft, argnums=1)
jac_lambda_hard_parameter = jax.jacobian(compute_optimal_update_parameters_hard, argnums=1)


# --------------------------------------------------------------------------------------------------
def grad_softmin(
    args: jtFloat[jax.Array, "num_args"], min_arg: jtFloat[jax.Array, ""], _order: int
) -> jtFloat[jax.Array, "num_args"]:
    """The gradient of the softmin function requires further masking of infeasible values.

    !!! danger "Needs reimplementation"
        See the disclaimer under [compute_softmin][eikonax.corefunctions.compute_softmin].
    """
    num_min_args = jnp.count_nonzero(args == min_arg)
    softmin_grad = jnp.where(args == min_arg, 1 / num_min_args, 0)
    return softmin_grad
