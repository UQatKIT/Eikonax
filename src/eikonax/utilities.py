import jax.numpy as jnp
import numpy as np
from scipy.spatial import Delaunay


# ==================================================================================================
def create_test_mesh(mesh_bounds: tuple, num_points: int) -> tuple[np.ndarray, np.ndarray]:
    mesh_points_x = np.linspace(*mesh_bounds, num_points)
    mesh_points_y = np.linspace(*mesh_bounds, num_points)
    mesh_points = np.column_stack(
        (np.repeat(mesh_points_x, num_points), np.tile(mesh_points_y, num_points))
    )
    triangulation = Delaunay(mesh_points)
    nodes = triangulation.points
    simplices = triangulation.simplices
    return nodes, simplices


# --------------------------------------------------------------------------------------------------
def get_adjacent_vertex_data(simplices: np.ndarray, num_vertices: np.ndarray) -> np.ndarray:
    max_num_adjacent_simplices = np.max(np.bincount(simplices.flatten()))
    adjacent_vertex_inds = -1 * np.ones((num_vertices, max_num_adjacent_simplices, 4), dtype=int)
    counter_array = np.zeros(num_vertices, dtype=int)
    node_permutations = ((0, 1, 2), (1, 0, 2), (2, 0, 1))

    for simplex_inds, simplex in enumerate(simplices):
        for permutation in node_permutations:
            center_vertex, adj_vertex_1, adj_vertex_2 = simplex[permutation,]
            adjacent_vertex_inds[center_vertex, counter_array[center_vertex]] = np.array(
                [center_vertex, adj_vertex_1, adj_vertex_2, simplex_inds]
            )
            counter_array[center_vertex] += 1
    return adjacent_vertex_inds


# --------------------------------------------------------------------------------------------------
def compute_softmin(
    args: list | tuple | np.ndarray | jnp.ndarray, order: int, start_index=0
) -> float:
    args = jnp.array(args)
    start_arg = args[start_index]
    args = args.at[start_index].set(jnp.inf)
    softmin = start_arg - 1 / order * jnp.log(1 + jnp.sum(jnp.exp(order * (start_arg - args))))
    return softmin


# --------------------------------------------------------------------------------------------------
def compute_soft_drelu(value: float, order: int) -> float:
    soft_value = jnp.log(1 + jnp.exp(order * value)) / order
    soft_value = 1 - jnp.log(1 + jnp.exp(-order * (soft_value - 1))) / order
    return soft_value
