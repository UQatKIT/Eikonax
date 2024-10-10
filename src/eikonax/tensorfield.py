from abc import ABC, abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import scipy as sp


# ==================================================================================================
class BaseVectorToSimplicesMap(ABC, eqx.Module):
    @abstractmethod
    def map(self, simplex_ind: int, parameters: jnp.ndarray) -> int | jnp.ndarray:
        pass


# --------------------------------------------------------------------------------------------------
class LinearScalarMap(BaseVectorToSimplicesMap):
    def map(self, simplex_ind: int, parameters: jnp.ndarray) -> int:
        parameter = parameters[simplex_ind]
        return parameter


# ==================================================================================================
class BaseSimplexTensor(ABC, eqx.Module):
    _dimension: int

    def __init__(self, dimension: int):
        self._dimension = dimension

    @abstractmethod
    def assemble(self, simplex_ind: int, parameters: float | jnp.ndarray) -> jnp.ndarray:
        pass

    @abstractmethod
    def derivative(self, simplex_ind: int, parameters: float | jnp.ndarray) -> jnp.ndarray:
        pass


# --------------------------------------------------------------------------------------------------
class LinearScalarSimplexTensor(BaseSimplexTensor):
    def assemble(self, simplex_ind: int, parameters: float) -> jnp.ndarray:
        tensor = parameters * jnp.identity(self._dimension)
        return tensor

    def derivative(self, simplex_ind: int, parameters: float) -> jnp.ndarray:
        derivative = jnp.expand_dims(jnp.identity(self._dimension), axis=-1)
        return derivative


# ==================================================================================================
class BaseTensorField(ABC, eqx.Module):
    _num_simplices: int
    _simplex_inds: jnp.ndarray
    _vector_to_simplices_map: BaseVectorToSimplicesMap
    _simplex_tensor: BaseSimplexTensor

    def __init__(
        self,
        num_simplices: int,
        vector_to_simplices_map: BaseVectorToSimplicesMap,
        simplex_tensor: BaseSimplexTensor,
    ):
        self._num_simplices = num_simplices
        self._simplex_inds = jnp.arange(num_simplices)
        self._vector_to_simplices_map = vector_to_simplices_map
        self._simplex_tensor = simplex_tensor

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def assemble_field(self, parameter_vector: jnp.ndarray) -> jnp.ndarray:
        pass

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def assemble_jacobian(
        self, partial_derivative_tensor_field: jnp.array, parameter_vector: jnp.ndarray
    ) -> jnp.ndarray:
        pass


# ==================================================================================================
class LinearTensorField(BaseTensorField):
    # ----------------------------------------------------------------------------------------------
    @eqx.filter_jit
    def assemble_field(self, parameter_vector: jnp.ndarray) -> jnp.ndarray:
        simplex_map = jax.vmap(self._vector_to_simplices_map.map, in_axes=(0, None))
        field_assembly = jax.vmap(self._simplex_tensor.assemble, in_axes=(0, 0))
        simplex_params = simplex_map(self._simplex_inds, parameter_vector)
        tensor_field = field_assembly(self._simplex_inds, simplex_params)

        return tensor_field

    # ----------------------------------------------------------------------------------------------
    def assemble_jacobian(
        self,
        number_of_vertices: int,
        partial_derivative_tensor_field: jnp.array,
        parameter_vector: jnp.ndarray,
    ) -> jnp.array:
        row_inds, simplex_inds, tensor_values = partial_derivative_tensor_field
        values, col_inds = self._assemble_jacobian(
            simplex_inds,
            tensor_values,
            parameter_vector,
        )

        jacobian = sp.sparse.coo_matrix(
            (values, (row_inds, col_inds)),
            shape=(number_of_vertices, parameter_vector.size),
        )
        return jacobian

    # ----------------------------------------------------------------------------------------------
    @eqx.filter_jit
    def _assemble_jacobian(
        self,
        simplex_inds: jnp.array,
        values: jnp.array,
        parameter_vector: jnp.ndarray,
    ) -> jnp.array:
        simplex_map = jax.vmap(self._vector_to_simplices_map.map, in_axes=(0, None))
        field_derivative = jax.vmap(self._simplex_tensor.derivative, in_axes=(0, 0))
        simplex_params = simplex_map(simplex_inds, parameter_vector)
        tensor_array = field_derivative(simplex_inds, simplex_params)
        partial_derivative = jnp.einsum("ijk,ijkl->il", values, tensor_array)
        partial_derivative = partial_derivative.flatten()
        ind_array = jnp.arange(parameter_vector.size)
        col_inds = simplex_map(simplex_inds, ind_array).flatten()

        return partial_derivative, col_inds
