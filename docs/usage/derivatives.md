# Parametric Derivatives

In the [Forward Solver](./solve.md) tutorial, we have shown how to solve the Eikonal equation with
Eikonax for a given mesh and tensor field. In this tutorial, we build on this procedure to obtain
parametric derivatives $\frac{d\mathbf{u}(\mathbf{M})}{d\mathbf{M}}$ of the solution w.r.t.
a given input tensor field. We further assume that a tensor field is defined via some parameter vector
$\mathbf{m}\in\mathbb{R}^M$, s.th. we can define the mapping $\mathbf{M}: \mathbb{R}^M \to \mathbb{R}^{N_S\times d\times d}$,
under the constraint that $\mathbf{M}$ is pointwise s.p.d.

In the following, we consider the scenario of having some loss functional
$l: \mathbb{R}^{N_V} \to \mathbb{R},\ l = l(\mathbf{u})$, depending on the solution $\mathbf{u}(\mathbf{M})$
of the eikonal equation for a specific tensor field $\mathbf{M}(\mathbf{m})$. In various problem settings,
such as minimization of the loss, it is essential to be able to obtain gradients w.r.t. the input parameter
vector,

$$
    g(\mathbf{m}) = \frac{d l(\mathbf{u})}{d\mathbf{u}}\frac{d\mathbf{u}(\mathbf{M})}{d\mathbf{M}}\frac{d\mathbf{M}(\mathbf{m})}{d\mathbf{m}}.
$$

This is the scenario we cover in this tutorial. Eikonax follows a *discretize-then-optimize* approach
to computing the gradient. Moreover, it efficiently computes discrete adjoints by exploitsing the causality in the
forward solution of the eikonal equation. A detailed description of this procedure is given
[here][eikonax.derivator.DerivativeSolver].

## Test Mesh Setup

We start by setting up the same square mesh as for the [Forward Solver](./solve.md) tutorial.
However, we rely on Eikonax' built-in [create_test_mesh][eikonax.preprocessing.create_test_mesh] function,
instead of using `scipy`. We also choose a much smaller mesh with $3\times 3$ vertices, to efficiently
compare derivatives against finite differences later.

```py
from eikonax import corefunctions, preprocessing

vertices, simplices = preprocessing.create_test_mesh((0, 1), (0, 1), 3, 3)
adjacency_data = preprocessing.get_adjacent_vertex_data(simplices, vertices.shape[0])
mesh_data = corefunctions.MeshData(vertices=vertices, adjacency_data=adjacency_data)
```

## Tensor Field Setup

In the [Forward Solver](./solve.md) tutorial, we have constructed a specific tensor field instance as a simple
`numpy` array. To evaluate derivatives, however, we need to properly define a mapping $\mathbf{M}(\mathbf{m})$,
and its derivative. Such a mapping is provided by the [tensorfield][eikonax.tensorfield] module.
The tensor field module comprises interfaces and basic implementations for two separate components.
The [`SimplexTensor`][eikonax.tensorfield.BaseSimplexTensor] describes how, for a given simplex index
$s$ and local parameter vector $\mathbf{m}_s$, the tensor $M_s$ for that simplex is constructed.
The [`VectorToSimplicesMap`][eikonax.tensorfield.BaseVectorToSimplicesMap], in turn, defines the
comtributions to $\mathbf{m}_s$ from the global parameter vector $\mathbf{m}$ for a given simplex s.
The actual [`TensorField`][eikonax.tensorfield.TensorField] object is created from these two components.

!!! info
    The above procedure is quite low-level and requires some effort from the user side. On the other
    hand, it guarantees flexibility with respect to the employed type of tensor field. Through strict
    application of the composition-over-inheritance principle, we can mix different global-to-local
    mappings and tensor assemblies, which are swiftly vectorized and differentiated by JAX.

In our example, we define local tensors with the built-in [InvLinearScalarSimplexTensor][eikonax.tensorfield.InvLinearScalarSimplexTensor]. This assembles the local tensor from a scalar $m_s > 0$
simply as $\mathbf{M}_s = \frac{1}{m_s}\mathbf{I}$. We further employ the [LinearScalarMap][eikonax.tensorfield.LinearScalarMap],
which is basically the map $m_s = \mathbf{m}[s]$. In total, we create our tensor field like this:
```py
from eikonax import tensorfield

tensor_on_simplex = tensorfield.InvLinearScalarSimplexTensor(vertices.shape[1])
tensor_field_mapping = tensorfield.LinearScalarMap()
tensor_field_object = tensorfield.TensorField(simplices.shape[0], tensor_field_mapping, tensor_on_simplex)
```

The `tensor_field_object` is an intelligent mapping for any valid input vector $\mathbf{m}$. For
demonstration purposes, we simply create a random inpur vector and build the tensor field with the
[`assemble_field`][eikonax.tensorfield.TensorField.assemble_field] method,
```py
import numpy as np

rng = np.random.default_rng(seed=0)
parameter_vector = rng.uniform(0.5, 1.5, simplices.shape[0])
tensor_field_instance = tensor_field_object.assemble_field(parameter_vector)
```


## Solver Setup and Run

We now have all components to conduct a forward solver run with Eikonax, analogously to the one
described in the [Forward Solver](./solve.md) tutorial.
```py
from eikonax import solver

solver_data = solver.SolverData(
    tolerance=1e-8,
    max_num_iterations=1000,
    loop_type="jitted_while",
    max_value=1000,
    use_soft_update=False,
    softminmax_order=10,
    softminmax_cutoff=0.01,
)
initial_sites = corefunctions.InitialSites(inds=(0,), values=(0,))
eikonax_solver = solver.Solver(mesh_data, solver_data, tensor_field_instance)
solution = eikonax_solver.run(parameter_field)
```

## Partial Derivatives

Evaluating the gradient $g(\mathbf{m})$ is a two-step procedure in Eikonax.
```py
from eikonax import derivator

derivator_data = derivator.PartialDerivatorData(
    use_soft_update=False,
    softminmax_order=None,
    softminmax_cutoff=None,
)
eikonax_derivator = derivator.PartialDerivator(mesh_data, derivator_data, initial_sites)
```

```py
sparse_partial_solution, sparse_partial_tensor = \
    eikonax_derivator.compute_partial_derivatives(solution.values, parameter_field)
```

```py
sparse_partial_parameter = \
    tensor_field.assemble_jacobian(solution.values.size, sparse_partial_tensor, parameter_vector)
```


## Derivative Solver

```py
derivative_solver = derivator.DerivativeSolver(solution.values, sparse_partial_solution)
```

```py
loss_grad = np.ones(solution.values.size)
adjoint = derivative_solver.solve(loss_grad)
total_grad = partial_derivative_parameter.T @ adjoint
```

## Comparison to Finite Differences