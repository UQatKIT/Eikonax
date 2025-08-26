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
such as minimization of the loss, it is essential to be able to obtain the gradient $\mathbf{g}$ w.r.t. the input parameter
vector,

$$
    \mathbf{g}(\mathbf{m}) = \frac{d l(\mathbf{u})}{d\mathbf{u}}\frac{d\mathbf{u}(\mathbf{M})}{d\mathbf{M}}\frac{d\mathbf{M}(\mathbf{m})}{d\mathbf{m}}.
$$

This is the scenario we cover in this tutorial. Eikonax follows a *discretize-then-optimize* approach
to computing the gradient. Moreover, it efficiently computes discrete adjoints by exploiting the causality in the
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
mesh_data = preprocessing.MeshData(vertices, simplices)
```

## Tensor Field Setup

In the [Forward Solver](./solve.md) tutorial, we have constructed a specific tensor field instance as a simple
`numpy` array. To evaluate derivatives, however, we need to properly define a mapping $\mathbf{M}(\mathbf{m})$,
and its derivative. Such a mapping is provided by the [tensorfield][eikonax.tensorfield] module.
The tensor field module comprises interfaces and basic implementations for two separate components.
The [`SimplexTensor`][eikonax.tensorfield.AbstractSimplexTensor] describes how, for a given simplex index
$s$ and local parameter vector $\mathbf{m}_s$, the tensor $M_s$ for that simplex is constructed.
The [`VectorToSimplicesMap`][eikonax.tensorfield.AbstractVectorToSimplicesMap], in turn, defines the
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

tensor_on_simplex = tensorfield.InvLinearScalarSimplexTensor(dimension=vertices.shape[1])
tensor_field_mapping = tensorfield.LinearScalarMap()
tensor_field_object = tensorfield.TensorField(
    num_simplices=simplices.shape[0],
    vector_to_simplices_map=tensor_field_mapping,
    simplex_tensor=tensor_on_simplex,
)
```

The `tensor_field_object` is an intelligent mapping for any valid input vector $\mathbf{m}$. For
demonstration purposes, we simply create a random input vector and build the tensor field with the
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
initial_sites = preprocessing.InitialSites(inds=(0,), values=(0,))
eikonax_solver = solver.Solver(mesh_data, solver_data, initial_sites)
solution = eikonax_solver.run(tensor_field_instance)
```

## Partial Derivatives

Evaluating the gradient $g(\mathbf{m})$ is a two-step procedure in Eikonax. Firstly, we evaluate, for a
given parameter vector $\mathbf{m}$ and associated solution $\mathbf{u}$, the
partial derivatives $\mathbf{G}_u$ and $\mathbf{G}_M$ of the global update operator in the
[iterative solver][eikonax.solver.Solver]. This can be done with the
[`PartialDerivator`][eikonax.derivator.PartialDerivator] object. Its configuration in 
[`PartialDerivatorData`][eikonax.derivator.PartialDerivatorData] is analogous to that of the
[forward solver](./solve.md).

!!! note
    First derivatives do not require a differentiable transformation for the update parameter $\lambda$.
    Second derivatives do, however.

```py
from eikonax import derivator

derivator_data = derivator.PartialDerivatorData(
    use_soft_update=False,
    softminmax_order=None,
    softminmax_cutoff=None,
)
eikonax_derivator = derivator.PartialDerivator(mesh_data, derivator_data, initial_sites)
```

We can obtain sparse representations (through local dependencies) of the partial derivatives via
the [`compute_partial_derivatives`][eikonax.derivator.PartialDerivator.compute_partial_derivatives]
method,
```py
output_partial_solution, output_partial_tensor = eikonax_derivator.compute_partial_derivatives(
    solution.values, tensor_field_instance
)
```

$\mathbf{G}_u$ and $\mathbf{G}_M$ are both returned as
[sparse COO matrices](https://sparse.pydata.org/en/stable/api/COO/). This is a sparse data format
analogous to the one in [scipy](https://scipy.org/), but extensible to higher order tensors.

$\mathbf{G}_u$ is a matrix with shape $N_V \times N_V$, whereas $\mathbf{G}_M$ is four-dimensional with 
shape $N_V \times N_S \times d \times d$

To compute the total derivative according to the chain rule, we require the tensor field derivative
$\frac{d\mathbf{M}}{d\mathbf{m}}$. This is provided by the tensor field component through the
[`assemble_jacobian`][eikonax.tensorfield.TensorField.assemble_jacobian] method,
```py
tensor_partial_parameter = tensor_field_object.assemble_jacobian(parameter_vector)
```
$\frac{d\mathbf{M}}{d\mathbf{m}}$ is a again a fourth order tensor in [sparse COO](https://sparse.pydata.org/en/stable/api/COO/)
format, with shape $N_S \times d \times d \times M$.
We can now obtain $\mathbf{G}_M$ via a simple tensor dot-product,
```py
import sparse as spa

output_partial_parameter = spa.einsum(
    "ijkl,jklm->im", output_partial_tensor, tensor_partial_parameter
)
```
which returns a sparse $N_V \times M$ matrix


## Derivative Solver
With the partial derivatives, we can now set up a sparse, triangular equation system for computing discrete adjoints,
and subsequently the gradient $\mathbf{g}$. The rational behind this procedure is explained in more detail 
[here][eikonax.derivator.DerivativeSolver]. We set up the solver for the equation system with the
[`DerivativeSolver`][eikonax.derivator.DerivativeSolver] component,
```py
derivative_solver = derivator.DerivativeSolver(solution.values, output_partial_solution)
```

We can now evaluate the discrete adjoint from a given loss gradient $\frac{dl}{d\mathbf{u}}$,
```py
loss_grad = np.ones(solution.values.size)
adjoint = derivative_solver.solve(loss_grad)
```

We then obtain the gradient by simple multiplication of the adjoint with $\mathbf{G}_m$,

```py
total_grad = output_partial_parameter.T @ adjoint
```

!!! tip "Reusability of the gradient copmutation"
    The setup of the derivative solver and $\mathbf{G}_m$ basically constitutes a sparse representation
    of the entire parametric Jacobian $\mathbf{J}$ at a point $\mathbf{m}$. Once assembled, arbitrary portions of
    the Jacobian can be constructed with negligible cost, even for large systems.

## Comparison to Finite Differences

As a proof-of-concept, we compare the Jacobian $\mathbf{J}$ at the point $\mathbf{m}$ produced by Eikonax
to forward finite differences. With Eikonax, we simply evaluate the gradient for all unit vectors
$\mathbf{e}_i,\ i=1,\ldots,N_V$ as "loss gradient" $\frac{dl}{d\mathbf{u}}$. Each such computation
yields a row $\mathbf{J}_i$ of the Jacobian.
To this end, Eikonax provides the utility function [`compute_eikonax_jacobian`][eikonax.derivator.compute_eikonax_jacobian],
```python
eikonax_jacobian = derivator.compute_eikonax_jacobian(derivative_solver, output_partial_parameter)
```

With finite difference, we evaluate a column $\mathbf{J}_j$
of the jacobian as

$$
    \mathbf{J}_j = \frac{\partial\mathbf{u}}{\partial m_j} \approx \frac{\mathbf{u}(\mathbf{m} + h\mathbf{e}_j) - \mathbf{u}(\mathbf{m})}{h},
$$

with $j=1,\ldots,N_S$, and $h$ denotes the step width of the finite difference scheme. Again,
we use an Eikonax utility module, [`finitediff`][eikonax.finitediff],

```python
step_widths = np.logspace(-5, -1, 101)
errors = []
for step_width in step_widths:
    finite_diff_jacobian = finitediff.compute_fd_jacobian(
        eikonax_solver=eikonal_solver,
        tensor_field=tensor_field_object,
        stencil=finitediff.finite_diff_1_forward,
        eval_point=parameter_vector,
        step_width=step_width,
    )
    error = np.linalg.norm(finite_diff_jacobian - eikonax_jacobian)
    errors.append(error)
```

The figure below shows the difference of the Jacobian matrices in Frobenius norm for $h\in[1\mathrm{e}{-5}, 1\mathrm{e}{-1}]$.
As can be expected, the error decreases linearly for decreasing $h$ down to the square root of the floating point precision
(32 bit in this case). Beyond this threshold, the error increases again due to round-off errors.
<figure markdown>
![samples](../images/ug_derivative_fd.png){ width="500" style="display: inline-block" }
</figure>