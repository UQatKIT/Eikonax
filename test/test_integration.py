import jax.numpy as jnp
import numpy as np
import pytest
from fimpy.solver import create_fim_solver

from eikonax import corefunctions, derivator, preprocessing, solver, tensorfield

pytestmark = pytest.mark.integration


# ================================== Integration Tests for Solver ==================================
@pytest.mark.slow
def test_solver_run_2D_uniform_tensorfield(mesh_and_tensorfield_2D_uniform, eikonax_solver_data):
    vertices, simplices, tensor_field = mesh_and_tensorfield_2D_uniform
    initial_sites = corefunctions.InitialSites(inds=jnp.array((0,)), values=jnp.array((0,)))
    adjacency_data = preprocessing.get_adjacent_vertex_data(simplices, vertices.shape[0])
    mesh_data = corefunctions.MeshData(vertices=vertices, adjacency_data=adjacency_data)
    eikonax_tensor_field = jnp.array(np.linalg.inv(tensor_field))
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
    initial_sites = corefunctions.InitialSites(inds=jnp.array((0,)), values=jnp.array((0,)))
    adjacency_data = preprocessing.get_adjacent_vertex_data(simplices, vertices.shape[0])
    mesh_data = corefunctions.MeshData(vertices=vertices, adjacency_data=adjacency_data)
    eikonax_tensor_field = jnp.array(np.linalg.inv(tensor_field))
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
    initial_sites = corefunctions.InitialSites(inds=jnp.array((0,)), values=jnp.array((0,)))
    adjacency_data = preprocessing.get_adjacent_vertex_data(simplices, vertices.shape[0])
    mesh_data = corefunctions.MeshData(vertices=vertices, adjacency_data=adjacency_data)
    eikonax_tensor_field = jnp.array(np.linalg.inv(tensor_field))
    solver_data = solver.SolverData(**eikonax_solver_data)

    fimpython_solver = create_fim_solver(vertices, simplices, tensor_field, use_active_list=False)
    fimpython_solution = fimpython_solver.comp_fim(initial_sites.inds, initial_sites.values)
    eikonax_solver = solver.Solver(mesh_data, solver_data, initial_sites)
    eikonax_solution = eikonax_solver.run(eikonax_tensor_field)
    assert np.allclose(fimpython_solution, np.array(eikonax_solution.values), atol=1e-4)


# =============================== Integration Tests for Tensor Field ===============================
def test_assemble_linear_scalar_simplex_tensor():
    assert False


def test_derivative_linear_scalar_simplex_tensor():
    assert False


# ================================= Integration Tests for Derivator ================================
@pytest.mark.slow
def test_compute_partial_derivatives():
    assert False


@pytest.mark.slow
def test_derivative_solver_constructor():
    assert False


def test_derivative_solver_solve():
    assert False
