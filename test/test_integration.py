import jax.numpy as jnp
import numpy as np
import pytest
from fimpy.solver import create_fim_solver

from eikonax import corefunctions, derivator, preprocessing, solver, tensorfield

pytestmark = pytest.mark.integration


# =============================== Integration Tests for Tensor Field ===============================
def test_tensor_field_assemble(tensorfield_setup_linear_scalar_map_linear_scalar_simplex_tensor):
    data, object_Types = tensorfield_setup_linear_scalar_map_linear_scalar_simplex_tensor
    dimension, num_simplices, parameter_vector, expected_tensor_field, _ = data
    MapObject, SimplexObject = object_Types
    map_object = MapObject()
    simplex_object = SimplexObject(dimension)
    tensorfield_object = tensorfield.TensorField(num_simplices, map_object, simplex_object)
    field = tensorfield_object.assemble_field(parameter_vector)
    assert jnp.allclose(field, expected_tensor_field)


# ================================== Integration Tests for Solver ==================================
@pytest.mark.slow
def test_solver_run_2D_uniform_tensorfield(mesh_and_tensorfield_2D_uniform, eikonax_solver_data):
    vertices, simplices, tensor_field = mesh_and_tensorfield_2D_uniform
    initial_sites = corefunctions.InitialSites(inds=(0,), values=(0,))
    adjacency_data = preprocessing.get_adjacent_vertex_data(simplices, vertices.shape[0])
    mesh_data = corefunctions.MeshData(vertices=vertices, adjacency_data=adjacency_data)
    eikonax_tensor_field = np.linalg.inv(tensor_field)
    solver_data = solver.SolverData(**eikonax_solver_data)

    fimpython_solver = create_fim_solver(vertices, simplices, tensor_field, use_active_list=False)
    fimpython_solution = fimpython_solver.comp_fim(initial_sites.inds, initial_sites.values)
    eikonax_solver = solver.Solver(mesh_data, solver_data, initial_sites)
    eikonax_solution = eikonax_solver.run(eikonax_tensor_field)
    assert np.allclose(fimpython_solution, np.array(eikonax_solution.values), atol=1e-4)


# --------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_solver_run_2D_random_tensorfield(mesh_and_tensorfield_2D_random, eikonax_solver_data):
    vertices, simplices, tensor_field = mesh_and_tensorfield_2D_random
    initial_sites = corefunctions.InitialSites(inds=(0,), values=(0,))
    adjacency_data = preprocessing.get_adjacent_vertex_data(simplices, vertices.shape[0])
    mesh_data = corefunctions.MeshData(vertices=vertices, adjacency_data=adjacency_data)
    eikonax_tensor_field = np.linalg.inv(tensor_field)
    solver_data = solver.SolverData(**eikonax_solver_data)

    fimpython_solver = create_fim_solver(vertices, simplices, tensor_field, use_active_list=False)
    fimpython_solution = fimpython_solver.comp_fim(initial_sites.inds, initial_sites.values)
    eikonax_solver = solver.Solver(mesh_data, solver_data, initial_sites)
    eikonax_solution = eikonax_solver.run(eikonax_tensor_field)
    assert np.allclose(fimpython_solution, np.array(eikonax_solution.values), atol=1e-4)


# --------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_solver_run_2D_function_tensorfield(mesh_and_tensorfield_2D_function, eikonax_solver_data):
    vertices, simplices, tensor_field = mesh_and_tensorfield_2D_function
    initial_sites = corefunctions.InitialSites(inds=(0,), values=(0,))
    adjacency_data = preprocessing.get_adjacent_vertex_data(simplices, vertices.shape[0])
    mesh_data = corefunctions.MeshData(vertices=vertices, adjacency_data=adjacency_data)
    eikonax_tensor_field = np.linalg.inv(tensor_field)
    solver_data = solver.SolverData(**eikonax_solver_data)

    fimpython_solver = create_fim_solver(vertices, simplices, tensor_field, use_active_list=False)
    fimpython_solution = fimpython_solver.comp_fim(initial_sites.inds, initial_sites.values)
    eikonax_solver = solver.Solver(mesh_data, solver_data, initial_sites)
    eikonax_solution = eikonax_solver.run(eikonax_tensor_field)
    assert np.allclose(fimpython_solution, np.array(eikonax_solution.values), atol=1e-4)


# ================================= Integration Tests for Derivator ================================
@pytest.mark.slow
def test_compute_partial_derivatives(setup_derivative_tests):
    input_data, fwd_solution, expected_partial_derivatives = setup_derivative_tests
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
def test_derivative_solver_constructor_viable():
    assert False


# --------------------------------------------------------------------------------------------------
def test_derivative_solver_solve_viable():
    assert False
