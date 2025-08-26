"""Test mesh creation and preparation for Eikonax solver runs.

!!! info
    The creation of test meshes can be done with any other tool. The format of the required
    adjacency data for Eikonax is strict, however.

Classes:
    MeshData: Data characterizing a computational mesh from a vertex-centered perspective.
    InitialSites: Initial site info.

Functions:
    create_test_mesh: Create a simple test mesh with Scipy's Delauny functionality.
    get_adjacent_vertex_data: Preprocess mesh data for a vertex-centered evaluation.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from numbers import Real
from typing import Annotated

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from beartype.vale import Is
from jaxtyping import Float as jtFloat
from jaxtyping import Int as jtInt
from scipy.spatial import Delaunay


# ==================================================================================================
@dataclass
class MeshData:
    """Data characterizing a computational mesh from a vertex-centered perspective.

    Attributes:
        vertices (jax.Array | npt.NDArray): The coordinates of the vertices in the mesh.
            The dimension of this array is `(num_vertices, dim)`, where num_vertices is the number
            of vertices in the mesh and dim is the dimension of the space in which the mesh is
            embedded.
        simplices (jax.Array | npt.NDArray): The vertex indices for each simplex in the mesh.
            The dimension of this array is `(num_simplices, 3)`, where num_simplices is the number
            of simplices in the mesh.
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
    simplices: jtInt[jax.Array | npt.NDArray, "num_simplices 3"]
    adjacency_data: jtInt[jax.Array | npt.NDArray, "num_vertices max_num_adjacent_simplices 4"] = (
        field(init=False)
    )
    num_vertices: int = field(init=False)
    num_simplices: int = field(init=False)

    def __post_init__(self) -> None:
        """Initialize data structures and convert to JAX arrays."""
        self.num_vertices = self.vertices.shape[0]
        self.num_simplices = self.simplices.shape[0]
        self.adjacency_data = get_adjacent_vertex_data(self.simplices, self.num_vertices)
        self.vertices = jnp.array(self.vertices, dtype=jnp.float32)
        self.simplices = jnp.array(self.simplices, dtype=jnp.int32)


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
def create_test_mesh(
    mesh_bounds_x: Annotated[Iterable[Real], Is[lambda x: len(x) == 2]],
    mesh_bounds_y: Annotated[Iterable[Real], Is[lambda x: len(x) == 2]],
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
    simplices: jtInt[jax.Array | npt.NDArray, "num_simplices 3"],
    num_vertices: Annotated[int, Is[lambda x: x > 0]],
) -> jtInt[jax.Array, "num_vertices max_num_adjacent_simplices 4"]:
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
    adjacent_vertex_inds = jnp.array(adjacent_vertex_inds, dtype=jnp.int32)

    return adjacent_vertex_inds
