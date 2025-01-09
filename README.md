# Eikonax: A Fully Differentiable Solver for the Anisotropic Eikonal Equation

Eikonax is a pure Python implementation of a solver for the anisotropic eikonal equation on triangulated meshes. In particular, it focuses on domains $\Omega$ either in 2D Euclidean space, or 2D manifolds in 3D Euclidean space. For a given, space-dependent parameter tensor field $\mathbf{M}$, Eikonax computes the arrival times $u$ according to

$$
\begin{gather*}
\sqrt{\big(\nabla u(\mathbf{x}),\mathbf{M}(\mathbf{x})\nabla u(\mathbf{x})\big)} = 1,\quad \mathbf{x}\in\Omega \\
\nabla u(\mathbf{x}) \cdot \mathbf{n}(\mathbf{x}) \geq 0,\quad \mathbf{x}\in\partial\Omega.
\end{gather*}
$$

The iterative solver is based on Godunov Upwind schemes and employs global Jacobi updates, which can be efficiently ported to SIMD architectures.
In addition, Eikonax implements an efficient algorithm for the evaluation of parametric derivatives, meaning the derivative of the solution vector with respect to the parameter tensor field, $\frac{du}{d\mathbf{M}}$. Through exploitation of causality in the forward solution, Eikonax can compute this derivatives through discrete adjoint on timescales much smaller than those for the forward solve.

Eikonax is mainly based on the [JAX](https://jax.readthedocs.io/en/latest/) software library. This allows for GPU offloading off relevant computations. In addition, Eikonax makes extensive use of JAX`s just-in-time compilation and automatic differentiation capabilities.


## Getting Started

Eikonax is deployed as a **python package**, simply install via
```bash
pip install eikonax
```

For **development**, we recommend using the great [**uv**](https://docs.astral.sh/uv/) project management tool, for with Eikonax provides a universal lock file. To set up a reproducible environment, run
```bash
uv sync
```
in the project root directory.

The [**documentation**](docs/build/index.html) provides further information regarding usage, theoretical background, technical setup and API. Alternatively, you can check out the notebooks under [`examples`](examples/)


## Acknowledgement and License

Eikonax is being developed in the research group [**Uncertainty Quantification**](https://www.scc.kit.edu/forschung/uq.php) at KIT. Itis distributed as free software under the [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)
