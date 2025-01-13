# Forward Solver

## Test Mesh Setup

```py
import numpy as np
from scipy.spatial import Delaunay
```

```py
mesh_points_x = np.linspace(0, 1, 100)
mesh_points_y = np.linspace(0, 1, 100)
mesh_points = np.column_stack((np.repeat(mesh_points_x, 100), np.tile(mesh_points_y, 100)))
triangulation = Delaunay(mesh_points)
vertices = triangulation.points
simplices = triangulation.simplices
```

## Tensor Field Setup
```py
simplex_centers = np.mean(vertices[simplices], axis=1)
inv_speed_values = \
    1 / (1 + 10 * np.exp(-50 * np.linalg.norm(simplex_centers - np.array([[0.65, 0.65]]), axis=-1) ** 2))
tensor_field = np.repeat(np.identity(2)[np.newaxis, :, :], simplices.shape[0], axis=0)
tensor_field = np.einsum("i,ijk->ijk", inv_speed_values, tensor_field)
```

## Solver Setup
```py
from eikonax import corefunctions, preprocessing, solver
```

```py
solver_data = solver.SolverData(
    tolerance=1e-8,
    max_num_iterations=1000,
    loop_type="jitted_while",
    max_value=1000,
    use_soft_update=False,
    softminmax_order=20,
    softminmax_cutoff=1.0,
    log_interval=1,
)

adjacency_data = preprocessing.get_adjacent_vertex_data(simplices, vertices.shape[0])
mesh_data = corefunctions.MeshData(vertices=vertices, adjacency_data=adjacency_data)
initial_sites = corefunctions.InitialSites(inds=(0,), values=(0,))
```

## Initialization and Run
```py
eikonax_solver = solver.Solver(mesh_data, solver_data, initial_sites)
solution = eikonax_solver.run(tensor_field)
```