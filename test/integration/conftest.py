import jax.numpy as jnp
import numpy as np
import pytest
import sparse as spa
from scipy.spatial import Delaunay

from eikonax import derivator, preprocessing, solver, tensorfield


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

    return data, tensorfield.LinearScalarMap, tensorfield.InvLinearScalarSimplexTensor


# ================================ Setup for Forward Solver Runs ===================================
@pytest.fixture(scope="module", params=[True, False], ids=["soft_update", "no_soft_update"])
def eikonax_solver_data(request):
    solver_data = {
        "tolerance": 1e-8,
        "max_num_iterations": 1000,
        "loop_type": "jitted_while",
        "max_value": 1000,
        "use_soft_update": request.param,
        "softminmax_order": 20,
        "softminmax_cutoff": 1,
        "log_interval": 1,
    }
    return solver_data


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module", params=[10, 100], ids=["small_mesh", "large_mesh"])
def meshes_for_2d_forward_evaluation(request):
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
def configurations_for_2d_forward_evaluation(meshes_for_2d_forward_evaluation, eikonax_solver_data):
    vertices, simplices = meshes_for_2d_forward_evaluation
    initial_sites = preprocessing.InitialSites(inds=(0,), values=(0,))
    mesh_data = preprocessing.MeshData(vertices=vertices, simplices=simplices)
    solver_data = solver.SolverData(**eikonax_solver_data)
    return simplices, vertices, mesh_data, solver_data, initial_sites


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def configurations_and_tensorfields_2d_uniform(configurations_for_2d_forward_evaluation):
    simplices, *_ = configurations_for_2d_forward_evaluation
    tensor_field = np.repeat(np.identity(2)[np.newaxis, :, :], simplices.shape[0], axis=0)
    return configurations_for_2d_forward_evaluation, tensor_field


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def configurations_and_tensorfields_2d_random(configurations_for_2d_forward_evaluation):
    simplices, *_ = configurations_for_2d_forward_evaluation
    num_simplices = simplices.shape[0]
    rng = np.random.default_rng(seed=0)
    inv_speed_values = rng.uniform(0.5, 1.5, num_simplices)
    tensor_field = np.repeat(np.identity(2)[np.newaxis, :, :], simplices.shape[0], axis=0)
    tensor_field = np.einsum("i,ijk->ijk", inv_speed_values, tensor_field)
    return configurations_for_2d_forward_evaluation, tensor_field


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def configurations_and_tensorfields_2d_function(configurations_for_2d_forward_evaluation):
    simplices, vertices, *_ = configurations_for_2d_forward_evaluation
    simplex_centers = np.mean(vertices[simplices], axis=1)
    inv_speed_values = 1 / (
        1
        + 10
        * np.exp(-50 * np.linalg.norm(simplex_centers - np.array([[0.65, 0.65]]), axis=-1) ** 2)
    )
    tensor_field = np.repeat(np.identity(2)[np.newaxis, :, :], simplices.shape[0], axis=0)
    tensor_field = np.einsum("i,ijk->ijk", inv_speed_values, tensor_field)
    return configurations_for_2d_forward_evaluation, tensor_field


# ============================== Setup for Parametric Derivatives ==================================
@pytest.fixture(scope="module")
def setup_analytical_partial_derivative_tests(
    mesh_small,
):
    vertices, simplices, _ = mesh_small
    tensor_dim = 2
    tensor_field = np.repeat(np.identity(tensor_dim)[np.newaxis, :, :], simplices.shape[0], axis=0)
    derivator_data = {
        "use_soft_update": True,
        "softminmax_order": 20,
        "softminmax_cutoff": 1,
    }
    initial_sites = {"inds": (0,), "values": (0,)}
    input_data = (vertices, simplices, tensor_field, initial_sites, derivator_data)

    fwd_solution = jnp.array(
        (0.0, 0.5, 1.0, 0.5, 0.8535534, 1.2071068, 1.0, 1.2071068, 1.5606602), dtype=jnp.float32
    )
    expected_sparse_partial_solution = spa.COO(
        coords=(
            jnp.array([0, 0, 1, 2, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8], dtype=jnp.int32),
            jnp.array([3, 1, 0, 1, 0, 1, 3, 1, 1, 3, 3, 3, 5, 7], dtype=jnp.int32),
        ),
        data=jnp.array(
            [0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5],
            dtype=jnp.float32,
        ),
        shape=(9, 8),
    )

    # fmt: off
    expected_sparse_partial_tensor = spa.COO(
        coords=(
            jnp.array([0, 0, 0 ,0 ,1 ,2 ,3 ,4, 4, 4 ,4 ,5, 5, 5, 5,
                       5, 5, 5, 5 ,6 ,7 ,7 ,7, 7, 7, 7, 7 ,7, 8, 8, 8, 8], dtype=jnp.int32),
            jnp.array([1, 1, 1, 1, 1, 3, 1, 0, 0, 0, 0, 2, 2, 2, 2,
                       3, 3, 3, 3, 5, 4, 4, 4, 4, 5, 5, 5 ,5 ,7 ,7 ,7 ,7], dtype=jnp.int32),
            jnp.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1,
                       0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=jnp.int32),
            jnp.array([0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1,
                       0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=jnp.int32),
        ),
        data=jnp.array([0.,         0.,         0.,         0.,         0.25,       0.25,
                        0.25,       0.08838835, 0.08838835, 0.08838835, 0.08838835, 0.08838835,
                        0.08838835, 0.08838835, 0.08838835, 0.08838835, 0.08838835, 0.08838835,
                        0.08838835, 0.25,       0.08838835, 0.08838835, 0.08838835, 0.08838835,
                        0.08838835, 0.08838835, 0.08838835, 0.08838835, 0.08838835, 0.08838835,
                        0.08838835, 0.08838835], dtype=jnp.float32),
        shape=(9, 8, 2, 2),
    )
    # fmt: on

    expected_partial_derivatives = (
        expected_sparse_partial_solution,
        expected_sparse_partial_tensor,
    )

    return input_data, fwd_solution, expected_partial_derivatives


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def setup_derivative_solve_checks(mesh_small):
    solver_data = solver.SolverData(
        tolerance=1e-8,
        max_num_iterations=1000,
        loop_type="jitted_while",
        max_value=1000,
        use_soft_update=True,
        softminmax_order=100,
        softminmax_cutoff=1,
        log_interval=1,
    )
    derivator_data = derivator.PartialDerivatorData(
        use_soft_update=True,
        softminmax_order=100,
        softminmax_cutoff=1,
    )
    vertices, simplices, _ = mesh_small
    mesh_data = preprocessing.MeshData(vertices=vertices, simplices=simplices)
    initial_sites = preprocessing.InitialSites(inds=(0,), values=(0,))
    rng = np.random.default_rng(seed=0)
    parameter_vector = rng.uniform(0.5, 1.5, simplices.shape[0])
    parameter_vector = jnp.array(parameter_vector)
    tensor_on_simplex = tensorfield.InvLinearScalarSimplexTensor(vertices.shape[1])
    tensor_field_mapping = tensorfield.LinearScalarMap()
    tensor_field = tensorfield.TensorField(
        simplices.shape[0], tensor_field_mapping, tensor_on_simplex
    )
    parameter_field = tensor_field.assemble_field(parameter_vector)
    eikonal_solver = solver.Solver(mesh_data, solver_data, initial_sites)
    solution = eikonal_solver.run(parameter_field)
    eikonax_derivator = derivator.PartialDerivator(mesh_data, derivator_data, initial_sites)
    output_partial_solution, output_partial_tensor = eikonax_derivator.compute_partial_derivatives(
        solution.values, parameter_field
    )
    tensor_partial_parameter = tensor_field.assemble_jacobian(parameter_vector)
    output_partial_parameter = spa.einsum(
        "ijkl,jklm->im", output_partial_tensor, tensor_partial_parameter
    )
    derivative_solver = derivator.DerivativeSolver(solution.values, output_partial_solution)

    return (
        parameter_vector,
        solution.values,
        tensor_field,
        eikonal_solver,
        derivative_solver,
        output_partial_parameter,
    )
