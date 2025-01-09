import copy

import jax.numpy as jnp
import numpy as np
import pytest
from fimpy.solver import create_fim_solver

from eikonax import (
    corefunctions,
    derivator,
    finitediff,
    logging,
    preprocessing,
    solver,
    tensorfield,
)

pytestmark = pytest.mark.integration


# =============================== Integration Tests for Tensor Field ===============================
def test_tensor_field_assemble(
    small_tensorfield_setup_linear_scalar_map_linear_scalar_simplex_tensor,
):
    data, MapObject, SimplexObject = (
        small_tensorfield_setup_linear_scalar_map_linear_scalar_simplex_tensor
    )
    dimension, num_simplices, parameter_vector, expected_tensor_field, _ = data
    map_object = MapObject()
    simplex_object = SimplexObject(dimension)
    tensorfield_object = tensorfield.TensorField(num_simplices, map_object, simplex_object)
    field = tensorfield_object.assemble_field(parameter_vector)
    assert jnp.allclose(field, expected_tensor_field)


# ================================== Integration Tests for Solver ==================================
@pytest.mark.slow
def test_solver_loop_types(configurations_and_tensorfields_2D_uniform):
    logger_data = logging.LoggerSettings(
        log_to_console=False,
        logfile_path=None,
    )
    logger = logging.Logger(logger_data)
    config, tensor_field = configurations_and_tensorfields_2D_uniform
    *_, mesh_data, solver_data, initial_sites = config
    solver_data_jitted_while = solver_data
    solver_data_nonjitted_while = copy.deepcopy(solver_data)
    solver_data_nonjitted_while.loop_type = "nonjitted_while"
    solver_data_jitted_for = copy.deepcopy(solver_data)
    solver_data_jitted_for.loop_type = "jitted_for"
    eikonax_solver = solver.Solver(mesh_data, solver_data_jitted_while, initial_sites)
    solution_jitted_while = eikonax_solver.run(np.linalg.inv(tensor_field))
    eikonax_solver = solver.Solver(mesh_data, solver_data_nonjitted_while, initial_sites, logger)
    solution_nonjitted_while = eikonax_solver.run(np.linalg.inv(tensor_field))
    eikonax_solver = solver.Solver(mesh_data, solver_data_jitted_for, initial_sites)
    solution_jitted_for = eikonax_solver.run(np.linalg.inv(tensor_field))
    assert np.allclose(solution_jitted_while.values, solution_nonjitted_while.values)
    assert np.allclose(solution_jitted_while.values, solution_jitted_for.values)


# --------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_solver_run_2D_uniform_tensorfield(configurations_and_tensorfields_2D_uniform):
    config, tensor_field = configurations_and_tensorfields_2D_uniform
    simplices, vertices, mesh_data, solver_data, initial_sites = config
    fimpython_solver = create_fim_solver(vertices, simplices, tensor_field, use_active_list=False)
    fimpython_solution = fimpython_solver.comp_fim(initial_sites.inds, initial_sites.values)
    eikonax_solver = solver.Solver(mesh_data, solver_data, initial_sites)
    eikonax_solution = eikonax_solver.run(np.linalg.inv(tensor_field))
    assert np.allclose(fimpython_solution, np.array(eikonax_solution.values), atol=1e-4)


# --------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_solver_run_2D_random_tensorfield(configurations_and_tensorfields_2D_random):
    config, tensor_field = configurations_and_tensorfields_2D_random
    simplices, vertices, mesh_data, solver_data, initial_sites = config
    fimpython_solver = create_fim_solver(vertices, simplices, tensor_field, use_active_list=False)
    fimpython_solution = fimpython_solver.comp_fim(initial_sites.inds, initial_sites.values)
    eikonax_solver = solver.Solver(mesh_data, solver_data, initial_sites)
    eikonax_solution = eikonax_solver.run(np.linalg.inv(tensor_field))
    assert np.allclose(fimpython_solution, np.array(eikonax_solution.values), atol=1e-4)


# --------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_solver_run_2D_function_tensorfield(configurations_and_tensorfields_2D_function):
    config, tensor_field = configurations_and_tensorfields_2D_function
    simplices, vertices, mesh_data, solver_data, initial_sites = config
    fimpython_solver = create_fim_solver(vertices, simplices, tensor_field, use_active_list=False)
    fimpython_solution = fimpython_solver.comp_fim(initial_sites.inds, initial_sites.values)
    eikonax_solver = solver.Solver(mesh_data, solver_data, initial_sites)
    eikonax_solution = eikonax_solver.run(np.linalg.inv(tensor_field))
    assert np.allclose(fimpython_solution, np.array(eikonax_solution.values), atol=1e-4)


# ================================= Integration Tests for Derivator ================================
@pytest.mark.slow
def test_compute_partial_derivatives(setup_analytical_partial_derivative_tests):
    input_data, fwd_solution, expected_partial_derivatives = (
        setup_analytical_partial_derivative_tests
    )
    vertices, simplices, tensor_field, initial_sites, derivator_data = input_data
    adjacency_data = preprocessing.get_adjacent_vertex_data(simplices, vertices.shape[0])
    initial_sites = corefunctions.InitialSites(**initial_sites)
    derivator_data = derivator.PartialDerivatorData(**derivator_data)
    mesh_data = corefunctions.MeshData(vertices=vertices, adjacency_data=adjacency_data)
    eikonax_derivator = derivator.PartialDerivator(mesh_data, derivator_data, initial_sites)
    sparse_partial_solution, sparse_partial_tensor = eikonax_derivator.compute_partial_derivatives(
        fwd_solution, tensor_field
    )
    expected_sparse_partial_solution, expected_sparse_partial_tensor = expected_partial_derivatives
    for sps, expected_sps in zip(
        sparse_partial_solution, expected_sparse_partial_solution, strict=True
    ):
        assert jnp.allclose(sps, expected_sps)
    for spt, expected_spt in zip(
        sparse_partial_tensor, expected_sparse_partial_tensor, strict=True
    ):
        assert jnp.allclose(spt, expected_spt)


# --------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_derivative_solver_constructor_viable(setup_derivative_solve_checks):
    _, solution_vector, _, _, derivative_solver, _ = setup_derivative_solve_checks
    system_matrix = derivative_solver.sparse_system_matrix.toarray()
    permutation_matrix = derivative_solver.sparse_permutation_matrix.toarray()
    assert system_matrix.shape == (solution_vector.size, solution_vector.size)
    assert permutation_matrix.shape == (solution_vector.size, solution_vector.size)
    assert np.allclose(system_matrix, np.triu(system_matrix))
    assert np.allclose(permutation_matrix @ permutation_matrix.T, np.identity(solution_vector.size))


@pytest.mark.slow
def test_partial_derivative_parameter_viable(setup_derivative_solve_checks):
    parameter_vector, solution_vector, _, _, _, partial_derivative_parameter = (
        setup_derivative_solve_checks
    )
    assert partial_derivative_parameter.shape == (
        solution_vector.size,
        parameter_vector.size,
    )


# --------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_derivative_solver_solve_viable(setup_derivative_solve_checks):
    parameter_vector, solution_vector, _, _, derivative_solver, partial_derivative_parameter = (
        setup_derivative_solve_checks
    )
    rhs_adjoint = np.ones(solution_vector.size)
    adjoint = derivative_solver.solve(rhs_adjoint)
    gradient = partial_derivative_parameter.T @ adjoint
    assert gradient.size == parameter_vector.size


# --------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_derivative_solver_vs_finite_differences(setup_derivative_solve_checks):
    (
        parameter_vector,
        _,
        tensor_field,
        eikonal_solver,
        derivative_solver,
        partial_derivative_parameter,
    ) = setup_derivative_solve_checks
    eikonax_jacobian = derivator.compute_eikonax_jacobian(
        derivative_solver, partial_derivative_parameter
    )
    finite_diff_jacobian = finitediff.compute_fd_jacobian(
        eikonax_solver=eikonal_solver,
        tensor_field=tensor_field,
        stencil=finitediff.finite_diff_1_forward,
        eval_point=parameter_vector,
        step_width=1e-3,
    )
    error = np.linalg.norm(finite_diff_jacobian - eikonax_jacobian)
    assert error < 2e-3
