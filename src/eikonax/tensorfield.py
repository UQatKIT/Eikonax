from abc import ABC, abstractmethod

import chex
import equinox as eqx
import jax
import jax.numpy as jnp


# ==================================================================================================
@chex.dataclass
class BaseTensorFieldData(ABC):
    dimension: int
    num_simplices: int


@chex.dataclass
class LinearTensorFieldData(BaseTensorFieldData):
    pass


# ==================================================================================================
class BaseTensorField(ABC):
    # ----------------------------------------------------------------------------------------------
    @staticmethod
    @abstractmethod
    def assemble(parameter_vector: jnp.ndarray, field_data: BaseTensorFieldData) -> jnp.ndarray:
        pass

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    @abstractmethod
    def gradient(parameter_vector: jnp.ndarray, field_data: BaseTensorFieldData) -> jnp.ndarray:
        pass

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    @abstractmethod
    def hessian_vector_product(
        parameter_vector: jnp.ndarray,
        direction_vector: jnp.ndarray,
        field_data: BaseTensorFieldData,
    ) -> jnp.ndarray:
        pass


# ==================================================================================================
class LinearTensorField(BaseTensorField):
    # ----------------------------------------------------------------------------------------------
    @staticmethod
    @eqx.filter_jit
    def assemble(parameter_vector: jnp.array, field_data: LinearTensorFieldData) -> jnp.ndarray:
        assert parameter_vector.size == field_data.dimension * field_data.num_simplices, (
            f"Size of parameter vector ({parameter_vector.size}) "
            f"does not match field dimension ({field_data.dimension}) "
            f"times number of simplices ({field_data.num_simplices})"
        )
        parameter_vector = jnp.reshape(
            parameter_vector, (field_data.num_simplices, field_data.dimension)
        )
        vectorized_assembly = jax.vmap(jnp.diag, in_axes=0)
        tensor_field = vectorized_assembly(parameter_vector)

        assert tensor_field.shape == (
            field_data.num_simplices,
            field_data.dimension,
            field_data.dimension,
        ), (
            f"tensor field shape {tensor_field.shape} does not match"
            f"number of simplices {field_data.num_simplices} "
            f"and dimension {field_data.dimension} squared"
        )
        return tensor_field

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def gradient(parameter_vector: jnp.ndarray, field_data: BaseTensorFieldData) -> jnp.array:
        pass

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def hessian_vector_product(
        parameter_vector: jnp.ndarray,
        direction_vector: jnp.ndarray,
        field_data: BaseTensorFieldData,
    ) -> jnp.ndarray:
        pass
