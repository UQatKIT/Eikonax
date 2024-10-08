import jax
import jax.numpy as jnp

from . import utilities


# --------------------------------------------------------------------------------------------------
def get_adjacent_edges(
    i: int, j: int, k: int, vertices: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    e_ji = vertices[i] - vertices[j]
    e_ki = vertices[i] - vertices[k]
    e_jk = vertices[k] - vertices[j]
    assert e_ji.shape == e_ki.shape == e_jk.shape, "All edges need to have the same shape"
    assert len(e_ji.shape) == 1, "Edges need to be 1D arrays"
    return e_ji, e_ki, e_jk


# --------------------------------------------------------------------------------------------------
def compute_optimal_update_parameters(
    solution_values: jnp.ndarray,
    M: jnp.ndarray,
    edges: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
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

    lambda_1, lambda_2 = _compute_optimal_update_parameters(solution_values, M, edges)
    lambda_1_clipped = utilities.compute_soft_drelu(lambda_1, order)
    lambda_2_clipped = utilities.compute_soft_drelu(lambda_2, order)
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
    lambda_array = jnp.array((0, 1, lambda_1_clipped, lambda_2_clipped))
    assert lambda_array.shape == (
        4,
    ), f"Lambda array needs to have shape (2,), but has shape {lambda_array.shape}"
    return lambda_array


# --------------------------------------------------------------------------------------------------
def compute_fixed_update(
    lambda_value: float,
    solution_values: jnp.ndarray,
    M: jnp.ndarray,
    edges: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
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


# --------------------------------------------------------------------------------------------------
def _compute_optimal_update_parameters(
    solution_values: jnp.ndarray,
    M: jnp.ndarray,
    edges: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
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


# --------------------------------------------------------------------------------------------------
def compute_update_from_adjacent_simplex(
    indices: jnp.ndarray,
    old_solution_vector: jnp.ndarray,
    vertices: jnp.ndarray,
    tensor_field: jnp.ndarray,
    drelu_order: int,
    drelu_cutoff: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    assert len(indices) == 4, f"Indices need to have length 4, but have length {len(indices)}"
    i, j, k, s = indices
    solution_values = jnp.array((old_solution_vector[j], old_solution_vector[k]))
    edges = get_adjacent_edges(i, j, k, vertices)
    M = tensor_field[s]
    lambda_array = compute_optimal_update_parameters(
        solution_values, M, edges, drelu_order, drelu_cutoff
    )
    update_candidates = jnp.zeros(4)

    for i, lambda_candidate in enumerate(lambda_array):
        update = compute_fixed_update(lambda_candidate, solution_values, M, edges)
        update_candidates = update_candidates.at[i].set(update)
    assert update_candidates.shape == (4,), (
        f"Update candidates have shape {update_candidates.shape}, but need to have shape (4,)"
    )
    return update_candidates, lambda_array

# --------------------------------------------------------------------------------------------------
def compute_vertex_update_candidates(
    old_solution_vector: jnp.ndarray,
    adjacent_vertex_inds: jnp.ndarray,
    vertices: jnp.ndarray,
    tensor_field: jnp.ndarray,
    drelu_order: int,
    drelu_cutoff: int,
) -> jnp.ndarray:
    max_num_adjacent_simplices = adjacent_vertex_inds.shape[0]
    assert adjacent_vertex_inds.shape == (max_num_adjacent_simplices, 4), (
        f"node-level adjacency data needs to have shape ({max_num_adjacent_simplices}, 4), "
        f"but has shape {adjacent_vertex_inds.shape}"
    )
    vertex_update_candidates = jnp.zeros((max_num_adjacent_simplices, 4))
    lambda_arrays = jnp.zeros((max_num_adjacent_simplices, 4))

    for i, indices in enumerate(adjacent_vertex_inds):
        simplex_update_candidates, lambda_array_candidates = (
            compute_update_from_adjacent_simplex(
                indices, old_solution_vector, vertices, tensor_field, drelu_order, drelu_cutoff
            )
        )
        vertex_update_candidates = vertex_update_candidates.at[i, :].set(
            simplex_update_candidates
        )
        lambda_arrays = lambda_arrays.at[i, :].set(lambda_array_candidates)

    active_simplex_inds = adjacent_vertex_inds[:, 3]
    vertex_update_candidates = jnp.where(
        (active_simplex_inds[..., None] != -1) & (lambda_arrays != -1),
        vertex_update_candidates,
        jnp.inf,
    )

    return vertex_update_candidates


# --------------------------------------------------------------------------------------------------
grad_update_lambda = jax.grad(compute_fixed_update, argnums=0)
grad_update_solution = jax.grad(compute_fixed_update, argnums=1)
grad_update_parameter = jax.grad(compute_fixed_update, argnums=2)
jac_lambda_solution = jax.jacobian(compute_optimal_update_parameters, argnums=0)
jac_lambda_parameter = jax.jacobian(compute_optimal_update_parameters, argnums=1)
