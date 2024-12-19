"""_summary_

Returns:
    _type_: _description_
"""

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Real


# ==================================================================================================
@chex.dataclass
class MeshData:
    """Data characterizing a computational mesh from a vertex-centered perspective.

    Attributes:
        vertices: The coordinates of the vertices in the mesh. The dimension of this array is
            (num_vertices, dim), where num_vertices is the number of vertices in the mesh and dim
            is the dimension of the space in which the mesh is embedded.
        adjacency_data: Adjacency data for each vertex. This is the list of adjacent triangles,
            together with the two vertices that span the respective triangle with the current
            vertex. The dimension of this array is (num_vertices, max_num_adjacent_simplices, 4),
            where max_num_adjacent_simplices is the maximum number of simplices that are adjacent
            to a vertex in the mesh. All entries have this maximum size, as JAX only operates on
            homogeneous data structures. If a vertex has fewer than max_num_adjacent_simplices
            adjacent simplices, the remaining entries are filled with -1.
    """

    vertices: Float[Array, "num_vertices dim"]
    adjacency_data: Float[Array, "num_vertices max_num_adjacent_simplices 4"]


@chex.dataclass
class InitialSites:
    """Initial site info.

    For a unique solution of the state-constrained Eikonal equation, the solution values need to be
    given a number of initial points (at least one). Multiple initial sites need to be compatible,
    in the sense that the arrival time from another source is not smaller than the initial value
    itself.

    Attributes:
        inds: The indices of the nodes where the initial sites are placed..
        values: The values of the initial sites.
    """

    inds: Float[Array, "num_initial_sites"]
    values: Float[Array, "num_initial_sites"]


# ==================================================================================================
def compute_softmin(
    args: Real[Array, "..."], min_arg: Real[Array, ""], order: int
) -> Float[Array, ""]:
    """Numerically stable computation of the softmin function based on the Boltzmann operator.

    This softmin function is applied to actual minimum values, meaning it does not have a purpose
    on the evaluation level. It renders the minimum computation differentiable, however.
    Importantly, the Boltzmann softmin is value preserving, meaning that the solution of the eikonal
    equation is the same as for a hard minimum. As JAX works on homogeneous arrays only, non-minimal
    values are also passed to this function. They are assumed to be masked as jnp.inf, which are
    handled in a numerically stable way by this routine.

    Args:
        args (jnp.ndarray): Values to compute the soft minimum over
        min_arg (Float): The actual value of the minimum argument, necessary for numerical stability
        order (Int): Approximation order of the softmin function

    Returns:
        float: Soft minimum value
    """
    arg_diff = min_arg - args
    nominator = jnp.where(args == jnp.inf, 0, args * jnp.exp(order * arg_diff))
    denominator = jnp.where(args == jnp.inf, 0, jnp.exp(order * arg_diff))
    soft_value = jnp.sum(nominator) / jnp.sum(denominator)
    return soft_value


# --------------------------------------------------------------------------------------------------
def compute_softminmax(value: Real[Array, "..."], order: int) -> Float[Array, "..."]:
    """Smooth double ReLU-type approximation that restricts a variable to the interval [0, 1].

    The method is numerically stable, obeys the value range, and does not introduce any new extrema.

    Args:
        value (Float): variable to restrict to range [0,1]
        order (Int): Approximation order of the smooth approximation

    Returns:
        float: Smoothed/restricted value
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
    i: Int[Array, ""],
    j: Int[Array, ""],
    k: Int[Array, ""],
    vertices: Float[Array, "num_vertices dim"],
) -> tuple[Float[Array, "dim"], Float[Array, "dim"], Float[Array, "dim"]]:
    """Compute the edged of a triangle from vertex indices and coordinates.

    Args:
        i (Int): First vertex index of a triangle
        j (Int): Second vertex index of a triangle
        k (Int): Third vertex index of a triangle
        vertices (jnp.ndarray): Array of all vertex coordinates

    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Triangle edge vectors
    """
    e_ji = vertices[i] - vertices[j]
    e_ki = vertices[i] - vertices[k]
    e_jk = vertices[k] - vertices[j]
    return e_ji, e_ki, e_jk


# --------------------------------------------------------------------------------------------------
def compute_optimal_update_parameters_soft(
    solution_values: Float[Array, "2"],
    parameter_tensor: Float[Array, "dim dim"],
    edges: tuple[Float[Array, "dim"], Float[Array, "dim"], Float[Array, "dim"]],
    softminmax_order: int,
    softminmax_cutoff: int,
) -> Float[Array, "4"]:
    """Compute position parameter for update of a node within a specific triangle.

    For a given vertex i and adjacent triangle, we compute the update for the solution of the
    Eikonal as propagating from a point on the connecting edge of the opposite vertices j and k.
    We thereby assume the solution value to vary linearly on that dge. The update parameter in [0,1]
    indicates the optimal linear combination of the solution values at j and k, in the sense that
    the solution value at i is minimized. As the underlying optimization problem is constrained,
    we compute the solutions of the unconstrained problem, as well as the boundary values. The
    former are constrained to the feasible region [0,1] by a soft minmax function.
    We further clip values lying to far outside the feasible region, by masking them with value -1.
    This function is a wrapper, for the unconstrained solution values, it calls the implementation
    function `_compute_optimal_update_parameters`.

    Args:
        solution_values (jnp.ndarray): Current solution values, as per the previous iteration
        parameter_tensor (jnp.ndarray): Parameter tensor for the current triangle
        edges (tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]): Edge vectors of the triangle under
            consideration
        softminmax_order (Int): Order of the soft minmax function to be employed
        softminmax_cutoff (Float): Cutoff value beyond parameter values are considered infeasible
            and masked with -1

    Returns:
        jnp.ndarray: All possible candidates for the update parameter, according to the solution
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
    solution_values: Float[Array, "2"],
    parameter_tensor: Float[Array, "dim dim"],
    edges: tuple[Float[Array, "dim"], Float[Array, "dim"], Float[Array, "dim"]],
) -> Float[Array, "4"]:
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
    solution_values: Float[Array, "2"],
    parameter_tensor: Float[Array, "dim dim"],
    edges: tuple[Float[Array, "dim"], Float[Array, "dim"], Float[Array, "dim"]],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Compute the optimal update parameter for the solution of the Eikonal equation.

    The function works for the update within a given triangle. The solutions of the unconstrained
    minimization problem are given as the roots of a quadratic polynomial. They may or may not
    lie inside the feasible region [0,1]. The function returns both solutions, which are then
    further processed in the calling wrapper.
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
    solution_values: Float[Array, "2"],
    parameter_tensor: Float[Array, "dim dim"],
    lambda_value: Float[Array, ""],
    edges: tuple[Float[Array, "dim"], Float[Array, "dim"], Float[Array, "dim"]],
) -> Float[Array, ""]:
    """Compute update for a given vertex, triangle, and update parameter.

    The update value is given by the solution at a point  on the edge between the opposite vertices,
    plus the travel time from that point to the vertices under consideration.

    Args:
        solution_values (jnp.ndarray): Current solution values at opposite vertices j and k,
            as per the previous iteration
        parameter_tensor (jnp.ndarray): Conductivity tensor for the current triangle
        lambda_value (Float): Optimal update parameter
        edges (tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]): Edge vectors of the triangle under
            consideration

    Returns:
        float: Updated solution value for the vertex under consideration
    """
    u_j, u_k = solution_values
    e_ji, _, e_jk = edges
    diff_vector = e_ji - lambda_value * e_jk
    transport_term = jnp.sqrt(jnp.dot(diff_vector, parameter_tensor @ diff_vector))
    update = lambda_value * u_k + (1 - lambda_value) * u_j + transport_term
    return update


# --------------------------------------------------------------------------------------------------
def compute_update_candidates_from_adjacent_simplex(
    old_solution_vector: Float[Array, "num_vertices"],
    tensor_field: Float[Array, "num_simplices dim dim"],
    adjacency_data: Int[Array, "max_num_adjacent_simplices"],
    vertices: Float[Array, "num_vertices dim"],
    use_soft_update: bool,
    softminmax_order: int,
    softminmax_cutoff: int | float,
) -> tuple[Float[Array, "4"], Float[Array, "4"]]:
    """Compute all possible update candidates from an adjacent triangle.

    Given a vertex and an adjacent triangle, this method computes all optimal update parameter
    candidates and the corresponding update values. To obey JAX's homogeneous array requirement,
    update values are also computed for infeasible parameter values, and have to be masked in the
    calling routine. This methods basically collects all results from the
    `compute_optimal_update_parameters` and `compute_fixed_update` methods.

    Args:
        old_solution_vector (jnp.ndarray): Given solution vector, as per a previous iteration
        tensor_field (jnp.ndarray): Array of all tensor fields
        adjacency_data (jnp.ndarray): Index of one adjaccent triangle and corresponding vertices
        vertices (jnp.ndarray): Array of all vertex coordinates
        softminmax_order (Int): Order of the soft minmax function for the update parameter, see
            `compute_softminmax`
        softminmax_cutoff (Float): Cutoff value for the soft minmax computation, see
            `compute_optimal_update_parameters`

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: Update values and parameter candidates from the given
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
    old_solution_vector: Float[Array, "num_vertices"],
    tensor_field: Float[Array, "num_simplices dim dim"],
    adjacency_data: Int[Array, "max_num_adjacent_simplices 4"],
    vertices: Float[Array, "num_vertices dim"],
    use_soft_update: bool,
    softminmax_order: int,
    softminmax_cutoff: int | float,
) -> Float[Array, "max_num_adjacent_simplices 4"]:
    """Compute all update candidates for a given vertex.

    This method combines all updates from adjacent triangles to a given vertex, as computed in the
    function `compute_update_candidates_from_adjacent_simplex`. Infeasible candidates are masked
    with jnp.inf.

    Args:
        old_solution_vector (jnp.ndarray): Given solution vector, as per a previous iteration
        tensor_field (jnp.ndarray): Array of all tensor fields
        adjacency_data (jnp.ndarray): Data of all adjacent triangles and corresponding vertices
        vertices (jnp.ndarray): Array of all vertex coordinates
        softminmax_order (Int): Order of the soft minmax function for the update parameter, see
            `compute_softminmax`
        softminmax_cutoff (Float): Cutoff value for the soft minmax computation, see
            `compute_optimal_update_parameters`

    Returns:
        jnp.ndarray: All possible update values for the given vertex, infeasible vertices are masked
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
# Derivative of update parameter function w.r.t. parameter tensor, 2xDxD
jac_lambda_soft_parameter = jax.jacobian(compute_optimal_update_parameters_soft, argnums=1)


# --------------------------------------------------------------------------------------------------
def grad_softmin(
    args: Float[Array, "num_args"], min_arg: Float[Array, ""], _order: int
) -> Float[Array, "num_args"]:
    """The gradient of the softmin function requires further masking of infeasible values.

    NOTE: this is only the gradient of the softmin function for identical, minimal values!
    """
    num_min_args = jnp.count_nonzero(args == min_arg)
    softmin_grad = jnp.where(args == min_arg, 1 / num_min_args, 0)
    return softmin_grad
