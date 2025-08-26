r"""Composable and differentiable parameter tensor fields.

This module provides ABCs and implementations for the creation of differentiable parameter fields
used in Eikonax. Recall that for the Eikonax solver, and particularly parameteric derivatives, we
require an input tensor field
$\mathbf{M}: \mathbb{R}^M \times \mathbb{N}_0 \to \mathbb{S}_+^{d\times d}$. This means that the
tensor field is a mapping $\mathbf{M}(\mathbf{m},s)$ that assigns, given a global parameter vector
$\mathbf{m}$, an s.p.d tensor to every simplex $s$ in the mesh. To allow for sufficient flexibility
in the choice of tensor field, we implement it as a composition of two main components.

1. [`AbstractVectorToSimplicesMap`][eikonax.tensorfield.AbstractVectorToSimplicesMap] provides the
    interface for a mapping from the global parameter vector $\mathbf{m}$ to the local parameter
    values $\mathbf{m}_s$ required to assemble the tensor $\mathbf{M}_s$ for simplex $s$.
2. [`AbstractSimplexTensor`][eikonax.tensorfield.AbstractSimplexTensor] provides the interface for
    the assembly of the local tensor $\mathbf{M}_s$, given the local contributions $\mathbf{m}_s$
    and a simplex s.

Concrete implementations of both components are used to initialize the
[`TensorField`][eikonax.tensorfield.TensorField] object, which vectorizes and differentiates them
using JAX, to provide the mapping $\mathbf{M}(\mathbf{m})$ and its Jacobian tensor
$\frac{d \mathbf{M}}{d \mathbf{m}}$.

Classes:
    AbstractVectorToSimplicesMap: ABC interface contract for vector-to-simplices maps
    LinearScalarMap: Simple one-to-one map from global to simplex parameters
    AbstractSimplexTensor: ABC interface contract for assembly of the tensor field
    LinearScalarSimplexTensor: SimplexTensor implementation relying on one parameter per simplex
    InvLinearScalarSimplexTensor: SimplexTensor implementation relying on one parameter per simplex
    TensorField: Tensor field component
"""

from abc import abstractmethod
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import sparse as spa
from jaxtyping import Float as jtFloat
from jaxtyping import Int as jtInt
from jaxtyping import Real as jtReal


# ==================================================================================================
class AbstractVectorToSimplicesMap(eqx.Module):
    """ABC interface contract for vector-to-simplices maps.

    Every component derived from this class needs to implement the `map`  and `derivative`  methods.
    The `map` method is responsible for returning the relevant parameters for a given simplex from
    the global parameter vector. The `derivative` method computes the derivative of the mapping with
    respect to the global parameters.

    Methods:
        map: Interface for vector-to-simplex mapping
        derivative: Interface for vector-to-simplex mapping derivative
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def map(
        self, simplex_ind: jtInt[jax.Array, ""], parameters: jtReal[jax.Array, "num_parameters"]
    ) -> jtReal[jax.Array, "num_parameters_local"]:
        """Interface for vector-to-simplex mapping.

        For the given `simplex_ind`, return those parameters from the global parameter vector that
        are relevant for the simplex. This methods need to be broadcastable over `simplex_ind` by
        JAX (with `vmap`).

        Args:
            simplex_ind (int): Index of the simplex under consideration
            parameters (jax.Array): Global parameter vector

        Raises:
            NotImplementedError: ABC error indicating that the method needs to be implemented
                in subclasses

        Returns:
            jax.Array: Relevant parameters for the simplex
        """
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def derivative(
        self, simplex_ind: jtInt[jax.Array, ""], parameters: jtReal[jax.Array, "num_parameters"]
    ) -> tuple[
        jtReal[jax.Array, "num_parameters_local"],
        jtReal[jax.Array, "num_parameters_local"],
    ]:
        r"""Interface for vector-to-simplex mapping derivative.

        The derivative of the simplex mapping for a given `simplex_ind` is a `num_local_parameters`
        $\times$ `num_parameters` Jacobian matrix, where the only non-zero entries are in the
        columns corresponding to the values in the global parameter that are used for the current
        simplex. The corresponding entries are typically identical to one. This method needs to
        return the Jacobian matrix in sparse COO format, i.e. one vector comprising the column
        numbers, and one vector comprising the matrix entries.

        Args:
            simplex_ind (int): Index of the simplex under consideration
            parameters (jax.Array): Global parameter vector

        Raises:
            NotImplementedError: ABC error indicating that the method needs to be implemented
                in subclasses

        Returns:
            tuple[jax.Array, jax.Array]: Jacobi matrix, expressed via relevant global indices and
                matrix entries
        """
        raise NotImplementedError


# --------------------------------------------------------------------------------------------------
@final
class LinearScalarMap(AbstractVectorToSimplicesMap):
    r"""Simple one-to-one map from global to simplex parameters.

    Every simplex takes exactly one parameter $m_s$, which is sorted in the global parameter
    in the same order as the simplices, meaning that $m_s = \mathbf{m}[s]$.
    """

    # ----------------------------------------------------------------------------------------------
    def map(
        self,
        simplex_ind: jtInt[jax.Array, ""],
        parameters: jtReal[jax.Array, "num_parameters"],
    ) -> jtReal[jax.Array, "num_parameters_local"]:
        """Return relevant parameters for a given simplex.

        Args:
            simplex_ind (int): Index of the simplex under consideration
            parameters (jax.Array): Global parameter vector

        Returns:
            jax.Array: relevant parameter (only one)
        """
        parameter = jnp.expand_dims(parameters[simplex_ind], axis=-1)
        return parameter

    # ----------------------------------------------------------------------------------------------
    def derivative(
        self, simplex_ind: jtInt[jax.Array, ""], _parameters: jtReal[jax.Array, "num_parameters"]
    ) -> tuple[
        jtReal[jax.Array, "num_parameters_local"],
        jtReal[jax.Array, "num_parameters_local"],
    ]:
        """Return sparse representation of Jacobi matrix, in this case two arrays of size one.

        Args:
            simplex_ind (int): Index of the simplex under consideration
            _parameters (jax.Array): Global parameter vector (not used)

        Returns:
            tuple[jax.Array, jax.Array]: Jacobi matrix, expressed via relevant global indices and
                matrix entries
        """
        global_parameter_inds = jnp.array((simplex_ind,), dtype=jnp.int32)
        derivative_vals = jnp.ones_like(global_parameter_inds, dtype=jnp.float32)
        return global_parameter_inds, derivative_vals


# ==================================================================================================
class AbstractSimplexTensor(eqx.Module):
    """ABC interface contract for assembly of the tensor field.

    `SimplexTensor` components assemble the tensor field for a given simplex and a set of parameters
    for that simplex. The relevant parameters are provided by the `VectorToSimplicesMap` component
    from the global parameter vector.

    !!! note
        This class provides the metric tensor as used in the inner product for the update stencil of
        the eikonal equation. This is the **inverse** of the conductivity tensor, which is the
        actual tensor field in the eikonal equation.

    Methods:
        assemble: Assemble the tensor field for a given simplex and parameters
        derivative: Parametric derivative of the `assemble` method
    """

    # Equinox modules are data classes, so we have to define attributes at the class level
    dimension: int

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def assemble(
        self,
        simplex_ind: jtInt[jax.Array, ""],
        parameters: jtFloat[jax.Array, "num_parameters_local"],
    ) -> jtFloat[jax.Array, "dim dim"]:
        r"""Assemble the tensor field for given simplex and parameters.

        Given a parameter array of size $m_s$, the method returns a tensor of size $d\times d$.
        The method needs to be broadcastable over `simplex_ind` by JAX (with `vmap`).

        Args:
            simplex_ind (int): Index of the simplex under consideration
            parameters (jax.Array): Parameters for the simplex

        Raises:
            NotImplementedError: ABC error indicating that the method needs to be implemented
                in subclasses

        Returns:
            jax.Array: Tensor field for the simplex under consideration
        """
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def derivative(
        self,
        simplex_ind: jtInt[jax.Array, ""],
        parameters: jtFloat[jax.Array, "num_parameters_local"],
    ) -> jtFloat[jax.Array, "dim dim num_local_parameters"]:
        r"""Parametric derivative of the `assemble` method.

        Given a parameter array of size $m_s$, the methods returns a Jacobian tensor of size
        $d\times d\times m_s$. The method needs to be broadcastable over `simplex_ind` by JAX
        (with `vmap`).

        Args:
            simplex_ind (int): Index of the simplex under consideration
            parameters (jax.Array): Parameters for the simplex

        Raises:
            NotImplementedError: ABC error indicating that the method needs to be implemented
                in subclasses

        Returns:
            jax.Array: Jacobian tensor for the simplex under consideration
        """
        raise NotImplementedError


# ==================================================================================================
@final
class LinearScalarSimplexTensor(AbstractSimplexTensor):
    r"""SimplexTensor implementation relying on one parameter per simplex.

    Given a scalar parameter $m_s$, the tensor field is assembled as $m_s \cdot \mathbf{I}$, where
    $\mathbf{I}$ is the identity matrix.

    Methods:
        assemble: Assemble the tensor field for a parameter vector
        derivative: Parametric derivative of the `assemble` method
    """

    # ----------------------------------------------------------------------------------------------
    def assemble(
        self,
        _simplex_ind: jtInt[jax.Array, ""],
        parameters: jtFloat[jax.Array, "num_parameters_local"],
    ) -> jtFloat[jax.Array, "dim dim"]:
        """Assemble tensor for given simplex.

        the `parameters` argument is a scalar here, and `_simplex_ind` is not used.

        Args:
            _simplex_ind (int): Index of simplex under consideration (not used)
            parameters (jax.Array): Parameter (scalar) for tensor assembly

        Returns:
            jax.Array: Tensor for the simplex
        """
        tensor = parameters * jnp.identity(self.dimension, dtype=jnp.float32)
        return tensor

    # ----------------------------------------------------------------------------------------------
    def derivative(
        self,
        _simplex_ind: jtInt[jax.Array, ""],
        _parameters: jtFloat[jax.Array, "num_parameters_local"],
    ) -> jtFloat[jax.Array, "dim dim num_local_parameters"]:
        """Parametric derivative of the `assemble` method.

        Args:
            _simplex_ind (int): Index of simplex under consideration (not used)
            _parameters (jax.Array): Parameter (scalar) for tensor assembly

        Returns:
            jax.Array: Jacobian tensor for the simplex under consideration
        """
        derivative = jnp.expand_dims(jnp.identity(self.dimension, dtype=jnp.float32), axis=-1)
        return derivative


# ==================================================================================================
@final
class InvLinearScalarSimplexTensor(AbstractSimplexTensor):
    r"""SimplexTensor implementation relying on one parameter per simplex.

    Given a scalar parameter $m_s$, the tensor field is assembled as
    $\frac{1}{m_s} \cdot \mathbf{I}$, where $\mathbf{I}$ is the identity matrix.

    Methods:
        assemble: Assemble the tensor field for a parameter vector
        derivative: Parametric derivative of the `assemble` method
    """

    # ----------------------------------------------------------------------------------------------
    def assemble(
        self,
        _simplex_ind: jtInt[jax.Array, ""],
        parameters: jtFloat[jax.Array, "num_parameters_local"],
    ) -> jtFloat[jax.Array, "dim dim"]:
        """Assemble tensor for given simplex.

        The `parameters` argument is a scalar here, and `_simplex_ind` is not used.

        Args:
            _simplex_ind (int): Index of simplex under consideration (not used)
            parameters (jax.Array): Parameter (scalar) for tensor assembly

        Returns:
            jax.Array: Tensor for the simplex
        """
        tensor = 1 / parameters * jnp.identity(self.dimension, dtype=jnp.float32)
        return tensor

    # ----------------------------------------------------------------------------------------------
    def derivative(
        self,
        _simplex_ind: jtInt[jax.Array, ""],
        parameters: jtFloat[jax.Array, "num_local_parameters"],
    ) -> jtFloat[jax.Array, "dim dim num_local_parameters"]:
        """Parametric derivative of the `assemble` method.

        Args:
            _simplex_ind (int): Index of simplex under consideration (not used)
            parameters (jax.Array): Parameter (scalar) for tensor assembly

        Returns:
            jax.Array: Jacobian tensor for the simplex under consideration
        """
        derivative = (
            -1
            / jnp.square(parameters)
            * jnp.expand_dims(jnp.identity(self.dimension, dtype=jnp.float32), axis=-1)
        )
        return derivative


# ==================================================================================================
class TensorField(eqx.Module):
    r"""Tensor field component.

    Tensor fields combine the functionality of vector-to-simplices maps and simplex tensors
    according to the composition over inheritance principle. They constitute the full mapping
    $\mathbf{M}(\mathbf{m})$ from the global parameter vector to the tensor field over all mesh
    faces (simplices). In addition, they provide the parametric derivative
    $\frac{d\mathbf{M}}{\mathbf{m}}$ of that mapping. Tensor fields are completely independent from
    the Eikonax solver and derivator, but the output of these two components can be used to compute
    the partial derivative $\mathbf{G}_m = \frac{du}{d\mathbf{M}}\frac{d\mathbf{M}}{\mathbf{m}}$.

    Methods:
        assemble_field: Assemble the tensor field for the given parameter vector
        assemble_jacobian: Assemble the parametric derivative $\frac{d\mathbf{M}}{\mathbf{m}}$
            of the tensor field
    """

    # Equinox modules are data classes, so we have to define attributes at the class level
    _num_simplices: int
    _simplex_inds: jtInt[jax.Array, "num_simplices"]
    _vector_to_simplices_map: AbstractVectorToSimplicesMap
    _simplex_tensor: AbstractSimplexTensor

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        num_simplices: int,
        vector_to_simplices_map: AbstractVectorToSimplicesMap,
        simplex_tensor: AbstractSimplexTensor,
    ) -> None:
        """Constructor.

        Takes information about the mesh simplices, a vector-to-simplices map, and a simplex tensor
        map.

        Args:
            num_simplices (int): Number of simplices in the mesh
            vector_to_simplices_map (AbstractVectorToSimplicesMap): Mapping from global to simplex
                parameters
            simplex_tensor (AbstractSimplexTensor): Tensor field assembly for a given simplex
        """
        self._num_simplices = num_simplices
        self._simplex_inds = jnp.arange(num_simplices, dtype=jnp.int32)
        self._vector_to_simplices_map = vector_to_simplices_map
        self._simplex_tensor = simplex_tensor

    # ----------------------------------------------------------------------------------------------
    @eqx.filter_jit
    def assemble_field(
        self, parameter_vector: jtFloat[jax.Array | npt.NDArray, "num_parameters"]
    ) -> jtFloat[jax.Array, "num_simplices dim dim"]:
        """Assemble global tensor field from global parameter vector.

        This method simply chains calls to the vector-to-simplices map and the simplex tensor
        objects, vectorized over all simplices.

        Args:
            parameter_vector (jax.Array | npt.NDArray): Global parameter vector

        Returns:
            jax.Array: Global tensor field
        """
        parameter_vector = jnp.array(parameter_vector, dtype=jnp.float32)
        vector_to_simplices_global = jax.vmap(self._vector_to_simplices_map.map, in_axes=(0, None))
        simplex_tensor_assembly_global = jax.vmap(self._simplex_tensor.assemble, in_axes=(0, 0))
        simplex_params = vector_to_simplices_global(self._simplex_inds, parameter_vector)
        tensor_field = simplex_tensor_assembly_global(self._simplex_inds, simplex_params)

        return tensor_field

    # ----------------------------------------------------------------------------------------------
    def assemble_jacobian(
        self,
        parameter_vector: jtFloat[jax.Array | npt.NDArray, "num_parameters"],
    ) -> spa.COO:
        r"""Assemble the Jacobian $\frac{d\mathbf{M}}{d\mathbf{m}}$.

        The assembly of the Jacobian matrix works via a local chaining of the `derivative` calls
        to the simplex tensor and the vector-to-simplices map. A sparse COO representation of the
        call output is then vectorized over all simplices with JAX's `vmap`.

        Args:
            parameter_vector (jax.Array | npt.NDArray): Global parameter vector

        Returns:
            spa.COO: Jacobian tensor in sparse COO format,
                has dimension $N_S \times d \times d \times M$
        """
        parameter_vector = jnp.array(parameter_vector)
        coords, values = self._assemble_jacobian_global(parameter_vector)
        coords = [np.array(co.flatten(), dtype=np.int32) for co in coords]
        values = np.array(values.flatten(), dtype=np.float32)
        jacobian = spa.COO(
            coords,
            values,
            shape=(
                self._num_simplices,
                self._simplex_tensor.dimension,
                self._simplex_tensor.dimension,
                parameter_vector.size,
            ),
        )

        return jacobian

    # ----------------------------------------------------------------------------------------------
    @eqx.filter_jit
    def _assemble_jacobian_global(
        self, parameter_vector: jtFloat[jax.Array, "num_parameters"]
    ) -> tuple[
        list[jtInt[jax.Array, "num_matrix_entries"]], jtFloat[jax.Array, "num_matrix_entries"]
    ]:
        """Intermediate call for vectorization with `vmap` in `jit` context."""
        assemble_jacobian_global = jax.vmap(self._assemble_jacobian_local, in_axes=(0, None))
        *coords, values = assemble_jacobian_global(self._simplex_inds, parameter_vector)

        return coords, values

    # ----------------------------------------------------------------------------------------------
    def _assemble_jacobian_local(
        self,
        simplex_ind: jtInt[jax.Array, ""],
        parameter_vector: jtFloat[jax.Array, "num_parameters"],
    ) -> tuple[
        jtInt[jax.Array, "num_matrix_entries"],
        jtInt[jax.Array, "num_matrix_entries"],
        jtInt[jax.Array, "num_matrix_entries"],
        jtInt[jax.Array, "num_matrix_entries"],
        jtFloat[jax.Array, "derivative_dim_1 derivative_dim_2"],
    ]:
        """Assembly of sparse jacobian representation for single simplex."""
        local_parameters = self._vector_to_simplices_map.map(simplex_ind, parameter_vector)
        partial_derivative_simplex = self._simplex_tensor.derivative(simplex_ind, local_parameters)
        global_parameter_inds, map_derivative_values = self._vector_to_simplices_map.derivative(
            simplex_ind, parameter_vector
        )
        tensor_dim = self._simplex_tensor.dimension
        num_tensor_entries = tensor_dim**2
        num_global_parameters = global_parameter_inds.size

        simplex_inds = simplex_ind * jnp.ones(
            (num_tensor_entries * num_global_parameters,), dtype=jnp.int32
        )
        tensor_inds = jnp.arange(self._simplex_tensor.dimension)
        tensor_d1_inds = jnp.tile(jnp.repeat(tensor_inds, tensor_dim), num_global_parameters)
        tensor_d2_inds = jnp.tile(jnp.tile(tensor_inds, tensor_dim), num_global_parameters)
        parameters_inds = jnp.repeat(global_parameter_inds, num_tensor_entries)
        partial_derivative_values = partial_derivative_simplex @ map_derivative_values

        return (
            simplex_inds,
            tensor_d1_inds,
            tensor_d2_inds,
            parameters_inds,
            partial_derivative_values,
        )
