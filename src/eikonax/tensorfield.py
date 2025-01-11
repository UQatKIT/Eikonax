r"""Composable and differentiable parameter tensor fields.

This module provides ABCs and implementations for the creation of differentiable parameter fields
used in Eikonax. Recall that for the Eikonax solver, and particularly parameteric derivatives, we
require an input tensor field
$\mathbf{M}: \mathbb{R}^M \times \mathbb{N}_0 \to \mathbb{S}_+^{d\times d}$. This means that the
tensor field is a mapping $\mathbf{M}(\mathbf{m},s)$ that assigns, given a global parameter vector
$\mathbf{m}$, an s.p.d tensor to every simplex $s$ in the mesh. To allow for sufficient flexibility
in the choice of tensor field, we implement it as a composition of two main components.

1. [`BaseVectorToSimplicesMap`][eikonax.tensorfield.BaseVectorToSimplicesMap] provides the interface
    for a mapping from the global parameter vector $\mathbf{m}$ to the local parameter values
    $\mathbf{m}_s$ required to assemble the tensor $\mathbf{M}_s$ for simplex $s$.
2. [`BaseSimplexTensor`][eikonax.tensorfield.BaseSimplexTensor] provides the interface for the
    assembly of the local tensor $\mathbf{M}_s$, given the local contributions $\mathbf{m}_s$ and a
    simplex s.

Concrete implementations of both components are used to initialize the
[`TensorField`][eikonax.tensorfield.TensorField] object, which vectorizes and differentiates them
using JAX, to provide the mapping $\mathbf{M}(\mathbf{m})$ and its Jacobian tensor
$\frac{d \mathbf{M}}{d \mathbf{m}}$.

Classes:
    BaseVectorToSimplicesMap: ABC interface contract for vector-to-simplices maps
    LinearScalarMap: Simple one-to-one map from global to simplex parameters
    BaseSimplexTensor: ABC interface contract for assembly of the tensor field
    LinearScalarSimplexTensor: SimplexTensor implementation relying on one parameter per simplex
    InvLinearScalarSimplexTensor: SimplexTensor implementation relying on one parameter per simplex
    TensorField: Tensor field component
"""

from abc import ABC, abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy.typing as npt
import scipy as sp
from jaxtyping import Float as jtFloat
from jaxtyping import Int as jtInt
from jaxtyping import Real as jtReal


# ==================================================================================================
class BaseVectorToSimplicesMap(ABC, eqx.Module):
    """ABC interface contract for vector-to-simplices maps.

    Every component derived from this class needs to implement the `map` method, which maps returns
    the relevant parameters for a given simplex from the global parameter vector.

    !!! note
        Eikonax assumes that the mapping from global to local parameters is linear, so that
        a parametric derivatives does not have to be provided.

    Methods:
        map: Interface for vector-so-simplex mapping
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def map(
        self, simplex_ind: jtInt[jax.Array, ""], parameters: jtReal[jax.Array, "num_parameters"]
    ) -> jtReal[jax.Array, "..."]:
        """Interface for vector-so-simplex mapping.

        For the given `simplex_ind`, return those parameters from the global parameter vector that
        are relevant for the simplex. This methods need to be broadcastable over `simplex_ind` by
        JAX (with `vmap`).

        Args:
            simplex_ind (jax.Array): Index of the simplex under consideration
            parameters (jax.Array): Global parameter vector

        Raises:
            NotImplementedError: ABC error indicating that the method needs to be implemented
                in subclasses

        Returns:
            jax.Array: Relevant parameters for the simplex
        """
        raise NotImplementedError


# --------------------------------------------------------------------------------------------------
class LinearScalarMap(BaseVectorToSimplicesMap):
    r"""Simple one-to-one map from global to simplex parameters.

    Every simplex takes exactly one parameter $m_s$, which is sorted in the global parameter
    in the same order as the simplices, meaning that $m_s = \mathbf{m}[s]$.
    """

    # ----------------------------------------------------------------------------------------------
    def map(
        self,
        simplex_ind: jtInt[jax.Array, ""],
        parameters: jtReal[jax.Array, "num_parameters"],
    ) -> jtReal[jax.Array, ""]:
        """Return relevant parameters for a given simplex.

        Args:
            simplex_ind (jax.Array): Index of the simplex under consideration
            parameters (jax.Array): Global parameter vector

        Returns:
            jax.Array: relevant parameter (only one)
        """
        parameter = parameters[simplex_ind]
        return parameter


# ==================================================================================================
class BaseSimplexTensor(ABC, eqx.Module):
    """ABC interface contract for assembly of the tensor field.

    `SimplexTensor` components assemble the tensor field for a given simplex and a set of parameters
    for that simplex. The relevant parameters are provided by the `VectorToSimplicesMap` component
    from the global parameter vector.

    !!! note
        Tis class provides the metric tensor as used in the inner product for the update stencil of
        the eikonal equation. This is the **INVERSE** of the conductivity tensor, which is the
        actual tensor field in the eikonal equation.

    Methods:
        assemble: Assemble the tensor field for a given simplex and parameters
        derivative: Parametric derivative of the `assemble` method
    """

    # Equinox modules are data classes, so we have to define attributes at the class level
    _dimension: int

    # ----------------------------------------------------------------------------------------------
    def __init__(self, dimension: int) -> None:
        """Constructor, simply fixes the dimension of the tensor field."""
        self._dimension = dimension

    @abstractmethod
    def assemble(
        self,
        simplex_ind: jtInt[jax.Array, ""],
        parameters: jtFloat[jax.Array, "num_parameters_local"],
    ) -> jtFloat[jax.Array, "dim dim"]:
        r"""Assemble the tensor field for given simplex and parameters.

        Given a parameter array of size $m_s$, the methods returns a tensor of size $d\times d$.
        The method needs to be broadcastable over `simplex_ind` by JAX (with `vmap`).

        Args:
            simplex_ind (jax.Array): Index of the simplex under consideration
            parameters (jax.Array): Parameters for the simplex

        Raises:
            NotImplementedError: ABC error indicating that the method needs to be implemented
                in subclasses

        Returns:
            jax.Array: Tensor field for the simplex under consideration
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(
        self,
        simplex_ind: jtInt[jax.Array, ""],
        parameters: jtFloat[jax.Array, "num_parameters_local"],
    ) -> jtFloat[jax.Array, "dim dim num_parameters_local"]:
        r"""Parametric derivative of the `assemble` method.

        Given a parameter array of size $m_s$, the methods returns a Jacobian tensor of size
        $d\times d\times m_s$. The method needs to be broadcastable over `simplex_ind` by JAX
        (with `vmap`).

        Args:
            simplex_ind (jax.Array): Index of the simplex under consideration
            parameters (jax.Array): Parameters for the simplex

        Raises:
            NotImplementedError: ABC error indicating that the method needs to be implemented
                in subclasses

        Returns:
            jax.Array: Jacobian tensor for the simplex under consideration
        """
        raise NotImplementedError


# --------------------------------------------------------------------------------------------------
class LinearScalarSimplexTensor(BaseSimplexTensor):
    r"""SimplexTensor implementation relying on one parameter per simplex.

    Given a scalar parameter $m_s$, the tensor field is assembled as $m_s \cdot \mathbf{I}$, where
    $\mathbf{I}$ is the identity matrix.

    Methods:
        assemble: Assemble the tensor field for a parameter vector
        derivative: Parametric derivative of the `assemble` method
    """

    def assemble(
        self, _simplex_ind: jtInt[jax.Array, ""], parameters: jtFloat[jax.Array, ""]
    ) -> jtFloat[jax.Array, "dim dim"]:
        """Assemble tensor for given simplex.

        the `parameters` argument is a scalar here, and `_simplex_ind` is not used.

        Args:
            _simplex_ind (jax.Array): Index of simplex under consideration (not used)
            parameters (jax.Array): Parameter (scalar) for tensor assembly

        Returns:
            jax.Array: Tensor for the simplex
        """
        tensor = parameters * jnp.identity(self._dimension, dtype=jnp.float32)
        return tensor

    def derivative(
        self, _simplex_ind: jtInt[jax.Array, ""], _parameters: jtFloat[jax.Array, ""]
    ) -> jtFloat[jax.Array, "dim dim num_parameters_local"]:
        """Parametric derivative of the `assemble` method.

        Args:
            _simplex_ind (jax.Array): Index of simplex under consideration (not used)
            _parameters (jax.Array): Parameter (scalar) for tensor assembly

        Returns:
            jax.Array: Jacobian tensor for the simplex under consideration
        """
        derivative = jnp.expand_dims(jnp.identity(self._dimension, dtype=jnp.float32), axis=-1)
        return derivative


# --------------------------------------------------------------------------------------------------
class InvLinearScalarSimplexTensor(BaseSimplexTensor):
    r"""SimplexTensor implementation relying on one parameter per simplex.

    Given a scalar parameter $m_s$, the tensor field is assembled as
    $\frac{1}{m_s} \cdot \mathbf{I}$, where $\mathbf{I}$ is the identity matrix.

    Methods:
        assemble: Assemble the tensor field for a parameter vector
        derivative: Parametric derivative of the `assemble` method
    """

    def assemble(
        self, _simplex_ind: jtInt[jax.Array, ""], parameters: jtFloat[jax.Array, ""]
    ) -> jtFloat[jax.Array, "dim dim"]:
        """Assemble tensor for given simplex.

        The `parameters` argument is a scalar here, and `_simplex_ind` is not used.

        Args:
            _simplex_ind (jax.Array): Index of simplex under consideration (not used)
            parameters (jax.Array): Parameter (scalar) for tensor assembly

        Returns:
            jax.Array: Tensor for the simplex
        """
        tensor = 1 / parameters * jnp.identity(self._dimension, dtype=jnp.float32)
        return tensor

    def derivative(
        self, _simplex_ind: jtInt[jax.Array, ""], parameters: jtFloat[jax.Array, ""]
    ) -> jtFloat[jax.Array, "dim dim num_parameters_local"]:
        """Parametric derivative of the `assemble` method.

        Args:
            _simplex_ind (jax.Array): Index of simplex under consideration (not used)
            parameters (jax.Array): Parameter (scalar) for tensor assembly

        Returns:
            jax.Array: Jacobian tensor for the simplex under consideration
        """
        derivative = (
            -1
            / jnp.square(parameters)
            * jnp.expand_dims(jnp.identity(self._dimension, dtype=jnp.float32), axis=-1)
        )
        return derivative


# ==================================================================================================
class TensorField(eqx.Module):
    r"""Tensor field component.

    Tensor fields combine the functionality of vector-to-simplices maps and simplex tensors
    according to the composition over inheritance principle. They constitute the full mapping
    $\mathbf{M}(\mathbf{m})$ from the global parameter vector to the tensor field over all mesh
    faces (simplices). In addition, they provide the parametric derivative
    $\frac{d\mathbf{M}}{\mathbf{m}}$ of that mapping, and assemble the full parameter-to-solution
    partial Jacobian $\mathbf{G}_m$ from a given partial derivative of the solution vector w.r.t.
    the tensor field $\mathbf{G}_M$. This introduces some degree of coupling to the eikonax solver,
    but is the simplest interface for computation of the total derivative according to the chain
    rule. More detailed explanations are given in the
    [`assemble_jacobian`][eikonax.tensorfield.TensorField.assemble_jacobian] method.

    Methods:
        assemble_field: Assemble the tensor field for the given parameter vector
        assemble_jacobian: Assemble the parametric derivative of a solution vector for a given
            parameter vector and derivative of the solution vector w.r.t. the tensor field
    """

    # Equinox modules are data classes, so we have to define attributes at the class level
    _num_simplices: int
    _simplex_inds: jtFloat[jax.Array, "num_simplices"]
    _vector_to_simplices_map: BaseVectorToSimplicesMap
    _simplex_tensor: BaseSimplexTensor

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        num_simplices: int,
        vector_to_simplices_map: BaseVectorToSimplicesMap,
        simplex_tensor: BaseSimplexTensor,
    ) -> None:
        """Constructor.

        Takes information about the mesh simplices, a vector-to-simplices map, and a simplex tensor
        map.

        Args:
            num_simplices (int): Number of simplices in the mesh
            vector_to_simplices_map (BaseVectorToSimplicesMap): Mapping from global to simplex
                parameters
            simplex_tensor (BaseSimplexTensor): Tensor field assembly for a given simplex
        """
        self._num_simplices = num_simplices
        self._simplex_inds = jnp.arange(num_simplices, dtype=jnp.int32)
        self._vector_to_simplices_map = vector_to_simplices_map
        self._simplex_tensor = simplex_tensor

    # ----------------------------------------------------------------------------------------------
    @eqx.filter_jit
    def assemble_field(
        self, parameter_vector: jtFloat[jax.Array | npt.NDArray, "num_parameters_global"]
    ) -> jtFloat[jax.Array, "num_simplex dim dim"]:
        """Assemble global tensor field from global parameter vector.

        This method simply chains calls to the vector-to-simplices map and the simplex tensor
        objects, vectorized over all simplices.

        Args:
            parameter_vector (jax.Array | npt.NDArray): Global parameter vector

        Returns:
            jax.Array: Global tensor field
        """
        parameter_vector = jnp.array(parameter_vector, dtype=jnp.float32)
        simplex_map = jax.vmap(self._vector_to_simplices_map.map, in_axes=(0, None))
        field_assembly = jax.vmap(self._simplex_tensor.assemble, in_axes=(0, 0))
        simplex_params = simplex_map(self._simplex_inds, parameter_vector)
        tensor_field = field_assembly(self._simplex_inds, simplex_params)

        return tensor_field

    # ----------------------------------------------------------------------------------------------
    def assemble_jacobian(
        self,
        number_of_vertices: int,
        derivative_solution_tensor: tuple[
            jtInt[jax.Array, "num_values"],
            jtInt[jax.Array, "num_values"],
            jtFloat[jax.Array, "num_values dim dim"],
        ],
        parameter_vector: jtFloat[jax.Array | npt.NDArray, "num_parameters_global"],
    ) -> sp.sparse.coo_matrix:
        r"""Assemble partial derivative of the Eikonax solution vector w.r.t. parameters.

        The total derivative of the Update operator w.r.t. the global parameter vector is given by
        the chain rule of differentiation,
        $\mathbf{G}_m = \mathbf{G}_M \frac{d\mathbf{M}}{d\mathbf{m}}$
        The Eikonax [`PartialDerivator`][eikonax.derivator.PartialDerivator] component evaluates
        the derivative of the solution vector w.r.t. the tensor field.
        The tensor field assembles the Jacobian tensor of the tensor field w.r.t. to the global
        parameter vector, and chains it with the solution-to-tensor derivative in a vectorized form.
        All computations are done in a sparse matrix format.
        Consider given a solution-to-tensor derivative of $\mathbf{G}_M$ of shape
        $N \times K \times d \times d$, where $N$ is the number of vertices, $K$ is the number of
        simplices, and $d$ is the physical dimension of the tensor field. This method internally
        assembles the tensor-to-parameter derivative $\frac{d\mathbf{M}}{d\mathbf{m}}$ of
        shape $K \times d \times d \times M$, where $M$ is the number of parameters. The total
        derivative is then computed as a tensor product of
        $\mathbf{G}_M$ and $\frac{d\mathbf{M}}{d\mathbf{m}}$ over their last and first three
        dimensions,respectively. The output is a sparse matrix of shape N x M, returned as a
        `scipy COO matrix`. The assembly is rather involved, so we handle it internally in this
        component, at the expense of introducing some additional coupling to the Eikonax Derivator

        !!! warning "Will be changed"
            The `PartialDerivator` returns a compressed representation of $\mathbf{G}_M$, which
            is hard to handle with standardized tensor product operations. Reducing the compression
            might allow for a more transparent interface at this point.


        Args:
            number_of_vertices (int): Number of vertices in the mesh
            derivative_solution_tensor (tuple[jax.Array, jax.Array, jax.Array]):
                Solution-tensor derivative of shape N x K x D x D. Provided as a tuple of row
                indices, simplex indices, and values, already in sparsified format. The row indices
                are the indices of the relevant vertices, and can be seen as one half of the index
                set of the resulting sparse matrix. For each row index, the corresponding simplex
                index indicates the simplex whose tensor values influence the solution at that
                vertex by means of the derivative.
            parameter_vector (jax.Array): Global parameter vector

        Returns:
            sp.sparse.coo_matrix: Sparse derivative of the Eikonax solution vector w.r.t. the
                global parameter vector, of shape N x M
        """
        row_inds, simplex_inds, derivative_solution_tensor_values = derivative_solution_tensor
        parameter_vector = jnp.array(parameter_vector, dtype=jnp.float32)
        values, col_inds = self._assemble_jacobian(
            simplex_inds,
            derivative_solution_tensor_values,
            parameter_vector,
        )

        # Multiple values may be assigned to the same row-col pair, the coo_matrix constructor
        # automatically sums up these duplicates
        jacobian = sp.sparse.coo_matrix(
            (values, (row_inds, col_inds)),
            shape=(number_of_vertices, parameter_vector.size),
        )
        return jacobian

    # ----------------------------------------------------------------------------------------------
    @eqx.filter_jit
    def _assemble_jacobian(
        self,
        simplex_inds: jtFloat[jax.Array, "num_values"],
        derivative_solution_tensor_values: jtFloat[jax.Array, "num_values"],
        parameter_vector: jtFloat[jax.Array, "num_parameters_global"],
    ) -> tuple[jtFloat[jax.Array, "..."], jtInt[jax.Array, "..."]]:
        r"""Compute the partial derivative $\frac{d\mathbf{M}}{d\mathbf{m}}$.

        Simplex-level derivatives are computed for all provided `simplex_inds` to match the
        solution-tensor derivatives obtained from the Eikonax derivator.

        Args:
            simplex_inds (jax.Array): Indices of simplices under consideration
            derivative_solution_tensor_values (jax.Array): Solution-tensor derivative values
            parameter_vector (jax.Array): Global parameter vector

        Returns:
            tuple[jax.Array, jax.Array]: Values and column indices of the Jacobian
        """
        simplex_map = jax.vmap(self._vector_to_simplices_map.map, in_axes=(0, None))
        field_derivative = jax.vmap(self._simplex_tensor.derivative, in_axes=(0, 0))
        simplex_params = simplex_map(simplex_inds, parameter_vector)
        derivative_tensor_parameter_values = field_derivative(simplex_inds, simplex_params)
        total_derivative = jnp.einsum(
            "ijk,ijkl->il", derivative_solution_tensor_values, derivative_tensor_parameter_values
        )
        total_derivative = total_derivative.flatten()
        ind_array = jnp.arange(parameter_vector.size, dtype=jnp.int32)
        col_inds = simplex_map(simplex_inds, ind_array).flatten()

        return total_derivative, col_inds
