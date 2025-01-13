# Parametric Derivatives

## Test Mesh Setup

```py
import numpy as np
from eikonax import (
    corefunctions,
    derivator,
    logging,
    preprocessing,
    solver,
    tensorfield,
)
```

```py
vertices, simplices = preprocessing.create_test_mesh((0, 1), (0, 1), 100, 100)
adjacency_data = preprocessing.get_adjacent_vertex_data(simplices, vertices.shape[0])
mesh_data = corefunctions.MeshData(vertices=vertices, adjacency_data=adjacency_data)
```

## Tensor Field Setup

```py
tensor_on_simplex = tensorfield.InvLinearScalarSimplexTensor(vertices.shape[1])
tensor_field_mapping = tensorfield.LinearScalarMap()
tensor_field_object = tensorfield.TensorField(simplices.shape[0], tensor_field_mapping, tensor_on_simplex)
```

```py
rng = np.random.default_rng(seed=0)
parameter_vector = rng.uniform(0.5, 1.5, simplices.shape[0])
tensor_field_instance = tensor_field_object.assemble_field(parameter_vector)
```



## Solver Setup and Run

```py
solver_data = solver.SolverData(
    tolerance=1e-8,
    max_num_iterations=1000,
    loop_type="jitted_while",
    max_value=1000,
    use_soft_update=False,
    softminmax_order=10,
    softminmax_cutoff=0.01,
    log_interval=1,
)
initial_sites = corefunctions.InitialSites(inds=(0,), values=(0,))
eikonax_solver = solver.Solver(mesh_data, solver_data, initial_sites, logger)
solution = eikonax_solver.run(parameter_field)
```

## Partial Derivatives

```py
derivator_data = derivator.PartialDerivatorData(
    use_soft_update=False,
    softmin_order=10,
    softminmax_order=10,
    softminmax_cutoff=0.01,
)

eikonax_derivator = derivator.PartialDerivator(mesh_data, derivator_data, initial_sites)
sparse_partial_solution, sparse_partial_tensor = \
    eikonax_derivator.compute_partial_derivatives(solution.values, parameter_field)
sparse_partial_parameter = \
    tensor_field.assemble_jacobian(solution.values.size, sparse_partial_tensor, parameter_vector)
```

## Derivative Solver
```py
loss_grad = np.ones(solution.values.size)
derivative_solver = derivator.DerivativeSolver(solution.values, sparse_partial_solution)
adjoint = derivative_solver.solve(loss_grad)
total_grad = partial_derivative_parameter.T @ adjoint
```