"""Composable and differentiable parameter fields.

This module provides ABCs and implementations for the creation of differentiable parameter fields
used in Eikonax. To provide sufficient flexibility, the actual tensor field component, implemented
in `TensorField`, comprises two main sub-components. Firstly, we implement a vector-to-simplices
map, adhering to the protocol defined in `BaseVectorToSimplicesMap`. This component maps the global
parameter vector to the local parameters of a given simplex. Secondly, we implement a simplex tensor
component, adhering to the protocol defined in `BaseSimplexTensor`. This component assembles the
tensor field for a given simplex and a set of parameters for that simplex. The relevant parameters
are provided by the `VectorToSimplicesMap` component from the global parameter vector.

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
        JAX (with vmap).

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
    """Simple one-to-one map from global to simplex parameters.

    Every simplex takes exactly one parameter, which is sorted in the global parameter in the same
    order as the simplices.
    """

    # ----------------------------------------------------------------------------------------------
    def map(
        self,
        simplex_ind: jtInt[jax.Array, ""],
        parameters: jtReal[jax.Array, "num_parameters_local"],
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
    from the global parameter vector. Note that this class provides the metric tensor as used in the
    inner product for the update stencil of the eikonal equation. This is the INVERSE of the
    conductivity tensor, which is the actual tensor field in the eikonal equation.

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
        """Assemble the tensor field for given simplex and parameters.

        Given a parameter array of size n_local, the methods returns a tensor of size dim x dim.
        The method needs to be broadcastable over `simplex_ind` by JAX (with vmap).

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
        """Parametric derivative of the `assemble` method.

        Given a parameter array of size n_local, the methods returns a Jacobian tensor of size
        dim x dim x n_local. The method needs to be broadcastable over `simplex_ind` by JAX
        (with vmap).

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
    """SimplexTensor implementation relying on one parameter per simplex.

    The simplex tensor is assembled as parameter * Identity for each simplex.

    Methods:
        assemble: Assemble the tensor field for a parameter vector
        derivative: Parametric derivative of the `assemble` method
    """

    def assemble(
        self, _simplex_ind: jtInt[jax.Array, ""], parameters: jtFloat[jax.Array, ""]
    ) -> jtFloat[jax.Array, "dim dim"]:
        """Assemble tensor for given simplex as parameter*Identity.

        the `parameters` argument is a scalar here, and `_simplex_ind` is not used. The method needs
        to be broadcastable over `simplex_ind` by JAX (with vmap).

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

        The method needs to be broadcastable over `simplex_ind` by JAX (with vmap).

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
    """SimplexTensor implementation relying on one parameter per simplex.

    The simplex tensor is assembled as 1/parameter * Identity for each simplex.

    Methods:
        assemble: Assemble the tensor field for a parameter vector
        derivative: Parametric derivative of the `assemble` method
    """

    def assemble(
        self, _simplex_ind: jtInt[jax.Array, ""], parameters: jtFloat[jax.Array, ""]
    ) -> jtFloat[jax.Array, "dim dim"]:
        """Assemble tensor for given simplex as 1/parameter*Identity.

        the `parameters` argument is a scalar here, and `_simplex_ind` is not used. The method needs
        to be broadcastable over `simplex_ind` by JAX (with vmap).

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

        The method needs to be broadcastable over `simplex_ind` by JAX (with vmap).

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
    """Tensor field component.

    Tensor fields combine the functionality of vector-to-simplices maps and simplex tensors
    according to the composition over inheritance principle. They constitute the full mapping
    from the global parameter vector to the tensor field over all mesh faces (simplices). In
    addition, they provide the parametric derivative of that mapping, and assemble the full
    parameter-to-solution Jacobian of the Eikonax solver from a given partial derivative of
    the solution vector w.r.t. the tensor field. This introduces some degree of coupling to the
    eikonax solver, but is the simplest interface for computation of the total derivative
    according to the chain rule. More detailed explanations are given in the `assemble_jacobian`
    method.

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
        """Assemble partial derivative of the Eikonax solution vector w.r.t. parameters.

        The total derivative of the solution vector w.r.t. the global parameter vector is given by
        the chain rule of differentiation. The Eikonax Derivator component evaluates the derivative
        of the solution vector w.r.t. the tensor field, which is the output of this component.
        The tensor field assembles the Jacobian tensor of the tensor field w.r.t. to the global
        parameter vector, and chains it with the solution-tensor derivative in a vectorized form.
        All computations are done in a sparse matrix format.
        Consider given a solution-tensor derivative of G_1 of shape N x K x D x D, where N is the
        number of vertices, K is the number of simplices, and D is the physical dimension of the
        tensor field. This component internally assembles the tensor-parameter derivative G_2 of
        shape K x D x D x M, where M is the number of parameters. The total derivative is then
        computed as a tensor product of G_1 and G_2 over their last and first three dimensions,
        respectively. The output is a sparse matrix of shape N x M, returned as a scipy COO matrix.
        The assembly is rather involved, so we handle it internally in this component, a the
        expense of introducing some additional coupling to the Eikonax Derivator

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
        """Compute the partial derivative of the the tensor field w.r.t. to global parameter vector.

        Simplex-level derivatives are computed for all provided `simplex_inds' to match the
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
