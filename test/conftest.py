import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial import Delaunay


# ==================================== Fixtures for Unit Tests =====================================
@pytest.fixture(scope="function")
def test_mesh_small():
    meta_data = {
        "mesh_bounds_x": [0, 1],
        "mesh_bounds_y": [0, 1],
        "num_points_x": 3,
        "num_points_y": 3,
    }

    vertices = jnp.array(
        [
            [0.0, 0.0],
            [0.0, 0.5],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.5, 1.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0],
        ],
        dtype=jnp.float32,
    )

    simplices = jnp.array(
        [[1, 3, 4], [3, 1, 0], [5, 1, 4], [1, 5, 2], [3, 7, 4], [7, 3, 6], [7, 5, 4], [5, 7, 8]],
        dtype=jnp.int32,
    )

    return vertices, simplices, meta_data


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="function")
def adjacency_data_for_test_mesh_small():
    adjacency_data = jnp.array(
        [
            [[0, 3, 1, 1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
            [[1, 3, 4, 0], [1, 3, 0, 1], [1, 5, 4, 2], [1, 5, 2, 3]],
            [[2, 1, 5, 3], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
            [[3, 1, 4, 0], [3, 1, 0, 1], [3, 7, 4, 4], [3, 7, 6, 5]],
            [[4, 1, 3, 0], [4, 5, 1, 2], [4, 3, 7, 4], [4, 7, 5, 6]],
            [[5, 1, 4, 2], [5, 1, 2, 3], [5, 7, 4, 6], [5, 7, 8, 7]],
            [[6, 7, 3, 5], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
            [[7, 3, 4, 4], [7, 3, 6, 5], [7, 5, 4, 6], [7, 5, 8, 7]],
            [[8, 5, 7, 7], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
        ]
    )

    return adjacency_data


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="function")
def mock_simplex_data():
    edges = (jnp.array([1, 0]), jnp.array([0, 1]), jnp.array([1, -1]))
    parameter_tensor = jnp.identity(2)
    solution_values = jnp.array([0, 0])
    return solution_values, parameter_tensor, edges


# --------------------------------------------------------------------------------------------------
@pytest.fixture(
    scope="function",
    params=[(jnp.array([0.0, 0.0]), [0.5, 0.5]), (jnp.array([0.0, 1.0]), [1.0, 0.0])],
)
def simplex_data_for_lambda(request):
    edges = (jnp.array([1, 0]), jnp.array([0, 1]), jnp.array([1, -1]))
    parameter_tensor = jnp.identity(2)
    solution_values, lambda_values = request.param
    return solution_values, parameter_tensor, edges, lambda_values


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="function")
def simplex_data_for_update():
    edges = (jnp.array([1, 0]), jnp.array([0, 1]), jnp.array([1, -1]))
    parameter_tensor = jnp.identity(2)
    solution_values = jnp.array([0.1, 0.7])
    lambda_value = 0.4
    update_value = 0.34 + jnp.sqrt(0.52)
    return solution_values, parameter_tensor, lambda_value, edges, update_value


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="function")
def simplex_data_for_derivatives():
    edges = (jnp.array([1, 0]), jnp.array([0, 1]), jnp.array([1, -1]))
    parameter_tensor = jnp.identity(2)
    solution_values = jnp.array([0.1, 0.7])
    lambda_value = 0.4
    simplex_data = solution_values, parameter_tensor, lambda_value, edges
    grad_update_solution = jnp.array([0.6, 0.4])
    grad_update_parameter = jnp.array([[0.2496151, 0.16641007], [0.16641006, 0.11094004]])
    grad_update_lambda = jnp.array(0.32264987)
    return simplex_data, grad_update_solution, grad_update_parameter, grad_update_lambda


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="function")
def vertex_update_data(test_mesh_small, adjacency_data_for_test_mesh_small):
    vertices, simplices, _ = test_mesh_small
    adjacency_data = adjacency_data_for_test_mesh_small
    tensor_field = jnp.repeat(jnp.identity(2)[jnp.newaxis, :, :], simplices.shape[0], axis=0)
    softminmax_order = 20
    softminmax_cutoff = 1

    return vertices, adjacency_data, tensor_field, softminmax_order, softminmax_cutoff


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="function")
def vertex_update_without_softmin(vertex_update_data):
    use_soft_update = False
    solution_values = jnp.array(
        (0.0, 0.5, 1.0, 0.5, 0.8535534, 1.2071068, 1.0, 1.2071068, 1.5606602)
    )
    vertex_update_candidates = jnp.array(
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

    return vertex_update_data, use_soft_update, solution_values, vertex_update_candidates


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="function")
def vertex_update_with_softmin(vertex_update_data):
    use_soft_update = True
    solution_values = jnp.array(
        (0.0, 0.5, 1.0, 0.5, 0.8535534, 1.2071068, 1.0, 1.2071068, 1.5606602)
    )
    vertex_update_candidates = jnp.array(
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

    return vertex_update_data, use_soft_update, solution_values, vertex_update_candidates


# ================================= Fixtures for Integration Tests =================================
@pytest.fixture(scope="module", params=[10, 100])
def test_mesh_for_runs(request):
    mesh_bounds_x = (0, 1)
    mesh_bounds_y = (0, 1)
    num_points_x = request.param
    num_points_y = request.param

    mesh_points_x = np.linspace(*mesh_bounds_x, num_points_x)
    mesh_points_y = np.linspace(*mesh_bounds_y, num_points_y)
    mesh_points = np.column_stack(
        (np.repeat(mesh_points_x, num_points_x), np.tile(mesh_points_y, num_points_y))
    )
    triangulation = Delaunay(mesh_points)
    vertices = triangulation.points
    simplices = triangulation.simplices

    return vertices, simplices


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mesh_and_tensorfield_2D_uniform(test_mesh_for_runs):
    vertices, simplices = test_mesh_for_runs
    tensor_field = np.repeat(np.identity(2)[np.newaxis, :, :], simplices.shape[0], axis=0)
    return vertices, simplices, tensor_field


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mesh_and_tensorfield_2D_random(test_mesh_for_runs):
    vertices, simplices = test_mesh_for_runs
    num_simplices = simplices.shape[0]
    rng = np.random.default_rng(seed=0)
    inv_speed_values = rng.uniform(0.5, 1.5, num_simplices)
    vertices, simplices = test_mesh_for_runs
    tensor_field = np.repeat(np.identity(2)[np.newaxis, :, :], simplices.shape[0], axis=0)
    tensor_field = np.einsum("i,ijk->ijk", inv_speed_values, tensor_field)
    return vertices, simplices, tensor_field


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mesh_and_tensorfield_2D_function(test_mesh_for_runs):
    vertices, simplices = test_mesh_for_runs
    simplex_centers = np.mean(vertices[simplices], axis=1)
    inv_speed_values = 1 / (
        1 + 10 * np.exp(-50 * np.linalg.norm(simplex_centers - np.array([[0.65, 0.65]]), axis=-1) ** 2)
    )
    vertices, simplices = test_mesh_for_runs
    tensor_field = np.repeat(np.identity(2)[np.newaxis, :, :], simplices.shape[0], axis=0)
    tensor_field = np.einsum("i,ijk->ijk", inv_speed_values, tensor_field)
    return vertices, simplices, tensor_field


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module", params=[True, False])
def eikonax_solver_data(request):
    solver_data = {
        "tolerance": 1e-8,
        "max_num_iterations": 1000,
        "loop_type": "jitted_while",
        "max_value": 1000,
        "use_soft_update": request.param,
        "softminmax_order": 20,
        "softminmax_cutoff": 1,
        "log_interval": None,
    }
    return solver_data

