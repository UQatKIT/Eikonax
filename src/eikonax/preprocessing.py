import jax.numpy as jnp
import numpy as np
from scipy.spatial import Delaunay


# ==================================================================================================
def create_test_mesh(
    mesh_bounds: tuple[float, float], num_points: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(mesh_bounds) != 2:
        raise ValueError(f"Mesh bounds must be a tuple with two elements, not {len(mesh_bounds)}")
    if mesh_bounds[0] >= mesh_bounds[1]:
        raise ValueError(
            f"Lower domain bound ({mesh_bounds[0]}) must be less than upper bound ({mesh_bounds[1]})"
        )
    if num_points < 2:
        raise ValueError(f"Number of mesh points must be at least 2, not {num_points}")
    mesh_points_x = np.linspace(*mesh_bounds, num_points)
    mesh_points_y = np.linspace(*mesh_bounds, num_points)
    mesh_points = np.column_stack(
        (np.repeat(mesh_points_x, num_points), np.tile(mesh_points_y, num_points))
    )
    triangulation = Delaunay(mesh_points)
    nodes = jnp.array(triangulation.points)
    simplices = jnp.array(triangulation.simplices)
    return nodes, simplices


# --------------------------------------------------------------------------------------------------
def get_adjacent_vertex_data(simplices: jnp.ndarray, num_vertices: int) -> jnp.ndarray:
    simplices = np.array(simplices)
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

    adjacent_vertex_inds = jnp.array(adjacent_vertex_inds)
    assert adjacent_vertex_inds.shape == (num_vertices, max_num_adjacent_simplices, 4), (
        f"Shape of adjacent vertex data is {adjacent_vertex_inds.shape} "
        f"but should be {(num_vertices, max_num_adjacent_simplices, 4)}"
    )
    return adjacent_vertex_inds
