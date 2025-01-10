"""Test mesh creation and preparation for Eikonax solver runs.

!!! info
    The creation of test meshes can be done with any other tool. The format of the required
    adjacency data for Eikonax is strict, however.

Functions:
    create_test_mesh: Create a simple test mesh with Scipy's Delauny functionality.
    get_adjacent_vertex_data: Preprocess mesh data for a vertex-centered evaluation.
"""

from collections.abc import Iterable
from typing import Annotated

import numpy as np
import numpy.typing as npt
from beartype.vale import Is
from jaxtyping import Float as jtFloat
from jaxtyping import Int as jtInt
from scipy.spatial import Delaunay


# ==================================================================================================
def create_test_mesh(
    mesh_bounds_x: Annotated[Iterable[float], Is[lambda x: len(x) == 2]],
    mesh_bounds_y: Annotated[Iterable[float], Is[lambda x: len(x) == 2]],
    num_points_x: Annotated[int, Is[lambda x: x >= 2]],
    num_points_y: Annotated[int, Is[lambda x: x >= 2]],
) -> tuple[jtFloat[npt.NDArray, "num_vertices dim"], jtInt[npt.NDArray, "num_simplices 3"]]:
    """Create a simple test mesh with Scipy's Delaunay functionality.

    This methods creates a simple square mesh with Scipy's
    [Delaunay](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html)
    triangulation.

    Args:
        mesh_bounds_x (Iterable[float, float]): Mesh bounds for x-direction
        mesh_bounds_y (Iterable[float, float]): Mesh bounds for y-direction
        num_points_x (int): Number of vertices for x-direction
        num_points_y (int): Number of vertices for y-direction

    Raises:
        ValueError: Checks that mesh bounds have correct dimension
        ValueError: Checks that mesh bounds are provided correctly
        ValueError: Checks that at least two mesh points are chosen

    Returns:
        tuple[npt.NDArray, npt.NDArray]: Array of vertex coordinates and array of simplex indices
    """
    for mesh_bounds in (mesh_bounds_x, mesh_bounds_y):
        if mesh_bounds[0] >= mesh_bounds[1]:
            raise ValueError(
                f"Lower domain bound ({mesh_bounds[0]}) must be less than upper bound"
                f"({mesh_bounds[1]})"
            )
    mesh_points_x = np.linspace(*mesh_bounds, num_points_x)
    mesh_points_y = np.linspace(*mesh_bounds, num_points_y)
    mesh_points = np.column_stack(
        (np.repeat(mesh_points_x, num_points_x), np.tile(mesh_points_y, num_points_y))
    )
    triangulation = Delaunay(mesh_points)
    return triangulation.points, triangulation.simplices


# --------------------------------------------------------------------------------------------------
def get_adjacent_vertex_data(
    simplices: jtInt[npt.NDArray, "num_simplices 3"],
    num_vertices: Annotated[int, Is[lambda x: x > 0]],
) -> jtInt[npt.NDArray, "num_vertices max_num_adjacent_simplices 4"]:
    """Preprocess mesh data for a vertex-centered evaluation.

    Standard mesh tools provide vertex coordinates and the vertex indices for each simplex.
    For the vertex-centered solution of the Eikonal equation, however, we need the adjacent
    simplices/vertices for each vertex. This method performs the necessary transformation.

    Args:
        simplices (npt.NDArray): Vertex indices for all simplices
        num_vertices (int): Number of vertices in  the mesh

    Returns:
        npt.NDArray: Array containing for each vertex the vertex and simplex indices of all
            adjacent simplices. Dimension is `(num_vertices, max_num_adjacent_simplices, 4)`,
            where the 4 entries contain the index of an adjacent simplex and the associated
            vertices. To ensure homogeneous arrays, all vertices have the same (maximum) number
            of adjacent simplices. Non-existing simplices are buffered with the value -1.
    """
    max_num_adjacent_simplices = np.max(np.bincount(simplices.flatten()))
    adjacent_vertex_inds = -1 * np.ones(
        (num_vertices, max_num_adjacent_simplices, 4), dtype=np.int32
    )
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
