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
    def assemble_field(
        parameter_vector: jnp.ndarray, field_data: BaseTensorFieldData
    ) -> jnp.ndarray:
        pass

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    @abstractmethod
    def assemble_jacobian(
        parameter_vector: jnp.ndarray, field_data: BaseTensorFieldData
    ) -> jnp.ndarray:
        pass

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    @abstractmethod
    def assemble_hessian(
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
    def assemble_field(
        parameter_vector: jnp.array, field_data: LinearTensorFieldData
    ) -> jnp.ndarray:
        # This is a compile time exception, as all field-data members are static
        if parameter_vector.size != field_data.dimension * field_data.num_simplices:
            raise ValueError(
                f"Size of parameter vector ({parameter_vector.size}) "
                f"does not match field dimension ({field_data.dimension}) "
                f"times number of simplices ({field_data.num_simplices})"
            )

        parameter_vector = jnp.reshape(
            parameter_vector, (field_data.num_simplices, field_data.dimension)
        )
        vectorized_assembly = jax.vmap(jnp.diag, in_axes=0)
        tensor_field = vectorized_assembly(parameter_vector)

        # Compile-time assertions for tensor field shape
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
    def assemble_jacobian(
        parameter_vector: jnp.ndarray, field_data: BaseTensorFieldData,
    ) -> jnp.array:
        pass

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def assemble_hessian(
        parameter_vector: jnp.ndarray,
        direction_vector: jnp.ndarray,
        field_data: BaseTensorFieldData,
    ) -> jnp.ndarray:
        pass
