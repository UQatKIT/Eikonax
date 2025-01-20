
![CI](https://img.shields.io/github/actions/workflow/status/UQatKIT/Eikonax/main.yaml?label=CI)
![Docs](https://img.shields.io/github/actions/workflow/status/UQatKIT/Eikonax/main.yaml?label=Docs)
![Coverage](https://img.shields.io/codecov/c/github/UQatKIT/Eikonax)
![Version](https://img.shields.io/pypi/v/Eikonax)
![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FUQatKIT%2FEikonax%2Fmain%2Fpyproject.toml)
![License](https://img.shields.io/github/license/UQatKIT/Eikonax) <br>
![JAX](https://img.shields.io/badge/JAX-Accelerated-9cf.svg)
![Beartype](https://github.com/beartype/beartype-assets/raw/main/badge/bear-ified.svg)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

# Eikonax: A Fully Differentiable Solver for the Anisotropic Eikonal Equation

Eikonax is a pure Python implementation of a solver for the anisotropic eikonal equation on triangulated meshes. In particular, it focuses on domains $\Omega$ either in 2D Euclidean space, or 2D manifolds in 3D Euclidean space. For a given, space-dependent parameter tensor field $\mathbf{M}$, and a set $\Gamma$ of initially active points, Eikonax computes the arrival times $u$ according to

$$
\begin{gather*}
\sqrt{\big(\nabla u(\mathbf{x}),\mathbf{M}(\mathbf{x})\nabla u(\mathbf{x})\big)} = 1,\quad \mathbf{x}\in\Omega, \\
\nabla u(\mathbf{x}) \cdot \mathbf{n}(\mathbf{x}) \geq 0,\quad \mathbf{x}\in\partial\Omega, \\
u(\mathbf{x}_0) = u_0,\quad \mathbf{x}_0 \in \Gamma.
\end{gather*}
$$

The iterative solver is based on *Godunov-type upwinding* and employs global *Jacobi updates*, which can be efficiently ported to SIMD architectures.
In addition, Eikonax implements an efficient algorithm for the evaluation of *parametric derivatives*, meaning the derivative of the solution vector with respect to the parameter tensor field, $\frac{du}{d\mathbf{M}}$. More precisely, we assume that the tensor field is parameterized through some vector $\mathbf{m}$, s.th. we compute $\frac{du}{d\mathbf{m}} = \frac{du}{d\mathbf{M}}\frac{d\mathbf{M}}{d\mathbf{m}}$. This make Eikonax particularly suitable for the inverse problem setting, where derivative information is typically indispensable for efficient solution procedures.
Through exploitation of causality in the forward solution, Eikonax can compute these derivatives through discrete adjoints on timescales much smaller than those for the forward solve.

### Key Features
- **Supports anisotropic conductivity tensors**
- **Works on irregular meshes**
- **GPU offloading of performance-relevant computations**
- **Super fast derivatives through causality-informed adjoints**


>Eikonax is mainly based on the [JAX](https://jax.readthedocs.io/en/latest/) software library. This allows for GPU offloading of relevant computations. In addition, Eikonax makes extensive use of JAX`s just-in-time compilation and automatic differentiation capabilities.



## Getting Started

Eikonax is deployed as a python package, simply install via
```bash
pip install eikonax
```

For development, we recommend using the great [uv](https://docs.astral.sh/uv/) project management tool, for which Eikonax provides a universal lock file. To set up a reproducible environment, run
```bash
uv sync --all-groups
```
in the project root directory.

## Documentation

The [documentation](https://uqatkit.github.io/Eikonax/) provides further information regarding usage, theoretical background, technical setup and API. Alternatively, you can check out the notebooks under [`examples`](https://github.com/UQatKIT/Eikonax/tree/main/examples)


## Acknowledgement and License

Eikonax is being developed in the research group [Uncertainty Quantification](https://www.scc.kit.edu/forschung/uq.php) at KIT.
It is partially based on the excellent [FIM-Python](https://fim-python.readthedocs.io/en/latest/) tool. Eikonax is distributed as free software under the [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/).