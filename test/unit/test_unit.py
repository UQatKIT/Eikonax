import jax.numpy as jnp
import numpy as np
import pytest

from eikonax import corefunctions, preprocessing

pytestmark = pytest.mark.unit


# =================================== Unit Tests for Preprocssing ==================================
def test_create_test_mesh(mesh_small):
    expected_vertices, expected_simplices, meta_data = mesh_small
    created_vertices, created_simplices = preprocessing.create_test_mesh(**meta_data)
    assert np.allclose(expected_vertices, created_vertices)
    assert np.allclose(expected_simplices, created_simplices)


# --------------------------------------------------------------------------------------------------
@staticmethod
def test_get_adjacent_vertex_data(mesh_and_adjacency_data_small):
    benchmark_mesh, expected_adjacency_data = mesh_and_adjacency_data_small
    benchmark_vertices, benchmark_simplices, _ = benchmark_mesh
    adjacency_data = preprocessing.get_adjacent_vertex_data(
        benchmark_simplices, benchmark_vertices.shape[0]
    )
    assert np.allclose(adjacency_data, expected_adjacency_data)


# ================================== Unit Tests for Core Functions =================================
@pytest.mark.parametrize("order", [1, 10], ids=["order_1", "order_10"])
@pytest.mark.parametrize(
    "input_values, expected_output",
    [(jnp.array((2.57,)), 2.57), (jnp.array((2, 2, jnp.inf)), 2)],
)
def test_compute_softmin(input_values, expected_output, order):
    min_arg = jnp.min(input_values)
    output = corefunctions.compute_softmin(input_values, min_arg, order=order)
    assert jnp.allclose(output, expected_output)


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("order", [1, 10, 1000], ids=["order_1", "order_10", "order_1000"])
@pytest.mark.parametrize(
    "input_value",
    [
        jnp.array(0),
        jnp.array(1),
        jnp.array(0.25),
        jnp.array(0.75),
        jnp.array(-1000),
        jnp.array(1000),
        jnp.array(-jnp.inf),
        jnp.array(jnp.inf),
    ],
)
def test_bounds_compute_softminmax(input_value, order):
    output = corefunctions.compute_softminmax(input_value, order)
    assert jnp.all(output >= 0)
    assert jnp.all(output <= 1)


# --------------------------------------------------------------------------------------------------
def test_values_compute_softminmax():
    order = 10
    input_values = jnp.linspace(-1, 2, 10, dtype=jnp.float32)
    expected_output_values = jnp.array(
        [
            4.5417501e-06,
            1.2718633e-04,
            3.5050693e-03,
            6.9309868e-02,
            3.3670986e-01,
            6.6328585e-01,
            9.3068784e-01,
            9.9649477e-01,
            9.9987280e-01,
            9.9999547e-01,
        ],
        dtype=jnp.float32,
    )
    output_values = corefunctions.compute_softminmax(input_values, order)
    assert jnp.allclose(output_values, expected_output_values)


# --------------------------------------------------------------------------------------------------
def test_compute_edges():
    test_vertices = jnp.array(([-1, -1], [1, 0], [0.5, 2]))
    expected_edges = (jnp.array([-2, -1]), jnp.array([-1.5, -3]), jnp.array([-0.5, 2]))
    output_edges = corefunctions.compute_edges(
        jnp.array(0), jnp.array(1), jnp.array(2), test_vertices
    )
    for output_edge, expected_edge in zip(output_edges, expected_edges, strict=True):
        assert jnp.allclose(output_edge, expected_edge)


# --------------------------------------------------------------------------------------------------
def test_compute_optimal_update_parameters(simplex_setup_and_optimal_lambda_candidates):
    *input_data, expected_lambda_values = simplex_setup_and_optimal_lambda_candidates
    output_lambda_values = corefunctions._compute_optimal_update_parameters(*input_data)
    for output_lambda, expected_lambda in zip(
        output_lambda_values, expected_lambda_values, strict=True
    ):
        assert jnp.allclose(output_lambda, expected_lambda)


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "lambda_input, lambda_expected",
    [
        ([0.25, 0.75], jnp.array((0, 1, 0.25, 0.75))),
        ([0.4, 0.4], jnp.array((0, 1, 0.4, -1))),
        ([0, 0.5], jnp.array((0, 1, -1, 0.5))),
        ([0.5, 0], jnp.array((0, 1, 0.5, -1))),
        ([-10, 0.1], jnp.array((0, 1, -1, 0.1))),
        ([0.1, 10], jnp.array((0, 1, 0.1, -1))),
        ([-10, 10], jnp.array((0, 1, -1, -1))),
    ],
)
def test_compute_optimal_update_parameters_hard(
    simplex_setup_and_optimal_lambda_candidates, lambda_input, lambda_expected, monkeypatch
):
    *input_data, _ = simplex_setup_and_optimal_lambda_candidates
    monkeypatch.setattr(
        "eikonax.corefunctions._compute_optimal_update_parameters", lambda *_: lambda_input
    )
    output_lambda_values = corefunctions.compute_optimal_update_parameters_hard(*input_data)
    assert jnp.allclose(output_lambda_values, lambda_expected)


# --------------------------------------------------------------------------------------------------
def test_compute_optimal_update_parameters_soft(
    simplex_setup_and_optimal_lambda_candidates, monkeypatch
):
    order, cutoff = 1, 1
    lambda_input = [jnp.array(-1.1, dtype=jnp.float32), jnp.array(2.1, dtype=jnp.float32)]
    *input_data, _ = simplex_setup_and_optimal_lambda_candidates
    monkeypatch.setattr(
        "eikonax.corefunctions._compute_optimal_update_parameters", lambda *_: lambda_input
    )
    output_lambda_values = corefunctions.compute_optimal_update_parameters_soft(
        *input_data, order, cutoff
    )
    assert jnp.allclose(output_lambda_values, jnp.array((0, 1, -1, -1)))


# --------------------------------------------------------------------------------------------------
def test_compute_fixed_update(simplex_setup_and_update):
    *input_data, expected_update_value = simplex_setup_and_update
    update_value = corefunctions.compute_fixed_update(*input_data)
    assert jnp.allclose(update_value, expected_update_value)


# --------------------------------------------------------------------------------------------------
def test_grad_update_solution(simplex_data_update_derivative):
    simplex_setup_and_update, expected_grad_update_solution, *_ = simplex_data_update_derivative
    *input_data, _ = simplex_setup_and_update
    grad_update_solution = corefunctions.grad_update_solution(*input_data)
    assert jnp.allclose(grad_update_solution, expected_grad_update_solution)


# --------------------------------------------------------------------------------------------------
def test_grad_update_parameter(simplex_data_update_derivative):
    simplex_setup_and_update, _, expected_grad_update_parameter, _ = simplex_data_update_derivative
    *input_data, _ = simplex_setup_and_update
    grad_update_parameter = corefunctions.grad_update_parameter(*input_data)
    assert jnp.allclose(grad_update_parameter, expected_grad_update_parameter)


# --------------------------------------------------------------------------------------------------
def test_grad_update_lambda(simplex_data_update_derivative):
    simplex_setup_and_update, *_, expected_grad_update_lambda = simplex_data_update_derivative
    *input_data, _ = simplex_setup_and_update
    grad_update_lambda = corefunctions.grad_update_lambda(*input_data)
    assert jnp.allclose(grad_update_lambda, expected_grad_update_lambda)


# --------------------------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize(
    "update_fixture", ["update_candidates_without_softminmax", "update_candidates_with_softminmax"]
)
def test_compute_global_update_candidates(update_fixture, request):
    update_setup, use_soft_update, solution_values, expected_update_candidates = (
        request.getfixturevalue(update_fixture)
    )
    vertices, adjacency_data, tensor_field, softminmax_order, softminmax_cutoff = update_setup
    vertices = jnp.array(vertices, dtype=jnp.float32)
    adjacency_data = jnp.array(adjacency_data, dtype=jnp.int32)

    for i in range(vertices.shape[0]):
        adj_data = adjacency_data[i, ...]
        exp_update_candidates = expected_update_candidates[i, ...]
        update_candidates = corefunctions.compute_vertex_update_candidates(
            solution_values,
            tensor_field,
            adj_data,
            vertices,
            use_soft_update,
            softminmax_order,
            softminmax_cutoff,
        )
        assert jnp.allclose(update_candidates, exp_update_candidates)
