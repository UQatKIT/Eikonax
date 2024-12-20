import jax.numpy as jnp
import numpy as np
import pytest


# ===================================== Single Simplex Setup =======================================
@pytest.fixture(scope="module")
def simplex_geometry_and_tensor():
    edges = (
        jnp.array([1, 0], dtype=jnp.float32),
        jnp.array([0, 1], dtype=jnp.float32),
        jnp.array([1, -1], dtype=jnp.float32),
    )
    parameter_tensor = jnp.identity(2, dtype=jnp.float32)
    return parameter_tensor, edges


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="function")
def simplex_setup_and_optimal_lambda_candidates(simplex_geometry_and_tensor):
    parameter_tensor, edges = simplex_geometry_and_tensor
    solution_values = jnp.array([0.0, 0.0], jnp.float32)
    lambda_values = (jnp.array(0.5, dtype=jnp.float32), jnp.array(0.5, dtype=jnp.float32))
    return solution_values, parameter_tensor, edges, lambda_values


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def simplex_setup_and_update(simplex_geometry_and_tensor):
    parameter_tensor, edges = simplex_geometry_and_tensor
    solution_values = jnp.array([0.1, 0.7], dtype=jnp.float32)
    lambda_value = jnp.array(0.4, dtype=jnp.float32)
    update_value = jnp.array(0.34 + jnp.sqrt(0.52), dtype=jnp.float32)
    return (solution_values, parameter_tensor, lambda_value, edges, update_value)


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def simplex_data_update_derivative(simplex_setup_and_update):
    grad_update_solution = jnp.array([0.6, 0.4], dtype=jnp.float32)
    grad_update_parameter = jnp.array(
        [[0.2496151, 0.16641007], [0.16641006, 0.11094004]], dtype=jnp.float32
    )
    grad_update_lambda = jnp.array(0.32264987, dtype=jnp.float32)
    return (
        simplex_setup_and_update,
        grad_update_solution,
        grad_update_parameter,
        grad_update_lambda,
    )


# ==================================== Single Global Update Step ===================================
@pytest.fixture(scope="function")
def global_update_setup(mesh_and_adjacency_data_small):
    mesh, adjacency_data = mesh_and_adjacency_data_small
    vertices, simplices, _ = mesh
    tensor_field = jnp.repeat(
        jnp.identity(2, dtype=jnp.float32)[jnp.newaxis, :, :], simplices.shape[0], axis=0
    )
    softminmax_order = 20
    softminmax_cutoff = 1

    return vertices, adjacency_data, tensor_field, softminmax_order, softminmax_cutoff


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="function")
def update_candidates_without_softminmax(global_update_setup):
    use_soft_update = False
    solution_values = jnp.array(
        (0.0, 0.5, 1.0, 0.5, 0.8535534, 1.2071068, 1.0, 1.2071068, 1.5606602), dtype=jnp.float32
    )
    global_update_candidates = jnp.array(
        [
            [
                [1.0, 1.0, 0.8535534, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
            ],
            [
                [1.2071068, 1.3535534, jnp.inf, jnp.inf],
                [1.2071068, 0.5, jnp.inf, jnp.inf],
                [1.9142137, 1.3535534, jnp.inf, jnp.inf],
                [1.9142137, 1.5, 1.6435943, jnp.inf],
            ],
            [
                [1.0, 1.7071068, jnp.inf, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
            ],
            [
                [1.2071068, 1.3535534, jnp.inf, jnp.inf],
                [1.2071068, 0.5, jnp.inf, jnp.inf],
                [1.9142137, 1.3535534, jnp.inf, jnp.inf],
                [1.9142137, 1.5, 1.6435943, jnp.inf],
            ],
            [
                [1.0, 1.0, 0.8535534, jnp.inf],
                [1.7071068, 1.0, jnp.inf, jnp.inf],
                [1.0, 1.7071068, jnp.inf, jnp.inf],
                [1.7071068, 1.7071068, 1.5606602, jnp.inf],
            ],
            [
                [1.2071068, 1.3535534, jnp.inf, jnp.inf],
                [1.2071068, 1.5, jnp.inf, jnp.inf],
                [1.9142137, 1.3535534, jnp.inf, jnp.inf],
                [1.9142137, 2.0606604, jnp.inf, jnp.inf],
            ],
            [
                [1.7071068, 1.0, jnp.inf, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
            ],
            [
                [1.2071068, 1.3535534, jnp.inf, jnp.inf],
                [1.2071068, 1.5, jnp.inf, jnp.inf],
                [1.9142137, 1.3535534, jnp.inf, jnp.inf],
                [1.9142137, 2.0606604, jnp.inf, jnp.inf],
            ],
            [
                [1.7071068, 1.7071068, 1.5606602, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
            ],
        ],
        dtype=jnp.float32,
    )

    return global_update_setup, use_soft_update, solution_values, global_update_candidates


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="function")
def update_candidates_with_softminmax(global_update_setup):
    use_soft_update = True
    solution_values = jnp.array(
        (0.0, 0.5, 1.0, 0.5, 0.8535534, 1.2071068, 1.0, 1.2071068, 1.5606602), dtype=jnp.float32
    )
    global_update_candidates = jnp.array(
        [
            [
                [1.0, 1.0, 0.8535534, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
            ],
            [
                [1.2071068, 1.3535534, 1.3535534, 1.2072148],
                [1.2071068, 0.5, jnp.inf, jnp.inf],
                [1.9142137, 1.3535534, 1.8898153, 1.3535534],
                [1.9142137, 1.5, 1.6435962, 1.5000012],
            ],
            [
                [1.0, 1.7071068, jnp.inf, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
            ],
            [
                [1.2071068, 1.3535534, 1.3535534, 1.2072148],
                [1.2071068, 0.5, jnp.inf, jnp.inf],
                [1.9142137, 1.3535534, 1.8898153, 1.3535534],
                [1.9142137, 1.5, 1.6435962, 1.5000012],
            ],
            [
                [1.0, 1.0, 0.8535534, jnp.inf],
                [1.7071068, 1.0, jnp.inf, jnp.inf],
                [1.0, 1.7071068, jnp.inf, jnp.inf],
                [1.7071068, 1.7071068, 1.5606602, jnp.inf],
            ],
            [
                [1.2071068, 1.3535534, 1.3535534, 1.2072148],
                [1.2071068, 1.5, jnp.inf, jnp.inf],
                [1.9142137, 1.3535534, 1.8898153, 1.3535534],
                [1.9142137, 2.0606604, 2.0606604, 1.9143217],
            ],
            [
                [1.7071068, 1.0, jnp.inf, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
            ],
            [
                [1.2071068, 1.3535534, 1.3535534, 1.2072148],
                [1.2071068, 1.5, jnp.inf, jnp.inf],
                [1.9142137, 1.3535534, 1.8898153, 1.3535534],
                [1.9142137, 2.0606604, 2.0606604, 1.9143217],
            ],
            [
                [1.7071068, 1.7071068, 1.5606602, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
                [jnp.inf, jnp.inf, jnp.inf, jnp.inf],
            ],
        ],
        dtype=jnp.float32,
    )

    return global_update_setup, use_soft_update, solution_values, global_update_candidates
