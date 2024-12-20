import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial import Delaunay

from eikonax import corefunctions, preprocessing, solver, tensorfield


# ================================= Setup for Tensor Field Check ===================================
@pytest.fixture(scope="function")
def small_tensorfield_setup_linear_scalar_map_linear_scalar_simplex_tensor():
    dimension = 2
    num_simplices = 3
    parameter_vector = jnp.array((1, 2, 3), dtype=jnp.float32)
    expected_tensor_field = jnp.array(
        (
            1 * jnp.identity(dimension),
            1 / 2 * jnp.identity(dimension),
            1 / 3 * jnp.identity(dimension),
        ),
        dtype=jnp.float32,
    )
    expected_field_derivative = -jnp.expand_dims(jnp.square(expected_tensor_field), axis=-1)
    data = (
        dimension,
        num_simplices,
        parameter_vector,
        expected_tensor_field,
        expected_field_derivative,
    )
    object_types = (tensorfield.LinearScalarMap, tensorfield.LinearScalarSimplexTensor)

    return data, object_types


# ================================ Setup for Forward Solver Runs ===================================
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


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module", params=[10, 100])
def meshes_for_2D_forward_evaluation(request):
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
def configurations_for_2D_forward_evaluation(meshes_for_2D_forward_evaluation, eikonax_solver_data):
    vertices, simplices = meshes_for_2D_forward_evaluation
    initial_sites = corefunctions.InitialSites(inds=(0,), values=(0,))
    adjacency_data = preprocessing.get_adjacent_vertex_data(simplices, vertices.shape[0])
    mesh_data = corefunctions.MeshData(vertices=vertices, adjacency_data=adjacency_data)
    solver_data = solver.SolverData(**eikonax_solver_data)
    return simplices, vertices, mesh_data, solver_data, initial_sites



# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def configurations_and_tensorfields_2D_uniform(configurations_for_2D_forward_evaluation):
    simplices, *_ = configurations_for_2D_forward_evaluation
    tensor_field = np.repeat(np.identity(2)[np.newaxis, :, :], simplices.shape[0], axis=0)
    return configurations_for_2D_forward_evaluation, tensor_field


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def configurations_and_tensorfields_2D_random(configurations_for_2D_forward_evaluation):
    simplices, *_ = configurations_for_2D_forward_evaluation
    num_simplices = simplices.shape[0]
    rng = np.random.default_rng(seed=0)
    inv_speed_values = rng.uniform(0.5, 1.5, num_simplices)
    tensor_field = np.repeat(np.identity(2)[np.newaxis, :, :], simplices.shape[0], axis=0)
    tensor_field = np.einsum("i,ijk->ijk", inv_speed_values, tensor_field)
    return configurations_for_2D_forward_evaluation, tensor_field


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def configurations_and_tensorfields_2D_function(configurations_for_2D_forward_evaluation):
    simplices, vertices, *_ = configurations_for_2D_forward_evaluation
    simplex_centers = np.mean(vertices[simplices], axis=1)
    inv_speed_values = 1 / (
        1
        + 10
        * np.exp(-50 * np.linalg.norm(simplex_centers - np.array([[0.65, 0.65]]), axis=-1) ** 2)
    )
    tensor_field = np.repeat(np.identity(2)[np.newaxis, :, :], simplices.shape[0], axis=0)
    tensor_field = np.einsum("i,ijk->ijk", inv_speed_values, tensor_field)
    return configurations_for_2D_forward_evaluation, tensor_field


# ============================== Setup for Paramettric Derivatives =================================
@pytest.fixture(scope="module")
def mesh_and_tensorfield_for_analytical_derivative_check(mesh_small):
    vertices, simplices = mesh_small
    tensor_field = np.repeat(np.identity(2)[np.newaxis, :, :], simplices.shape[0], axis=0)
    return vertices, simplices, tensor_field


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def setup_analytical_partial_derivative_tests(
    small_tensorfield_setup_linear_scalar_map_linear_scalar_simplex_tensor,
):
    derivator_data = {"softmin_order": 20, "softminmax_order": 20, "softminmax_cutoff": 1}
    initial_sites = {"inds": (0,), "values": (0,)}
    input_data = (
        *small_tensorfield_setup_linear_scalar_map_linear_scalar_simplex_tensor,
        initial_sites,
        derivator_data,
    )
    fwd_solution = jnp.array(
        (0.0, 0.5, 1.0, 0.5, 0.8535534, 1.2071068, 1.0, 1.2071068, 1.5606602), dtype=jnp.float32
    )
    expected_sparse_partial_solution = (
        jnp.array([0, 0, 1, 2, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8], dtype=jnp.int32),
        jnp.array([3, 1, 0, 1, 0, 1, 3, 1, 1, 3, 3, 3, 5, 7], dtype=jnp.int32),
        jnp.array(
            [0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5],
            dtype=jnp.float32,
        ),
    )
    expected_sparse_partial_tensor = (
        jnp.array([0, 1, 2, 3, 4, 5, 5, 6, 7, 7, 8], dtype=jnp.int32),
        jnp.array([1, 1, 3, 1, 0, 2, 3, 5, 4, 5, 7], dtype=jnp.int32),
        jnp.array(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.25]],
                [[0.0, 0.0], [0.0, 0.25]],
                [[0.25, 0.0], [0.0, 0.0]],
                [[0.08838835, 0.08838835], [0.08838835, 0.08838835]],
                [[0.08838835, 0.08838835], [0.08838835, 0.08838835]],
                [[0.08838835, 0.08838835], [0.08838835, 0.08838835]],
                [[0.25, 0.0], [0.0, 0.0]],
                [[0.08838835, 0.08838835], [0.08838835, 0.08838835]],
                [[0.08838835, 0.08838835], [0.08838835, 0.08838835]],
                [[0.08838835, 0.08838835], [0.08838835, 0.08838835]],
            ],
            dtype=jnp.float32,
        ),
    )
    expected_partial_derivatives = (
        expected_sparse_partial_solution,
        expected_sparse_partial_tensor,
    )

    return input_data, fwd_solution, expected_partial_derivatives
