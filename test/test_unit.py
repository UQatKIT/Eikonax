import jax.numpy as jnp
import numpy as np
import pytest

from eikonax import corefunctions, preprocessing, tensorfield

pytestmark = pytest.mark.unit


# ==================================================================================================
class TestPreprocessing:
    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def test_create_test_mesh(test_mesh_small):
        benchmark_vertices, benchmark_simplices, meta_data = test_mesh_small
        created_vertices, created_simplices = preprocessing.create_test_mesh(**meta_data)
        assert jnp.allclose(benchmark_vertices, created_vertices)
        assert jnp.allclose(benchmark_simplices, created_simplices)

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def test_get_adjacent_vertex_data(test_mesh_small, adjacency_data_for_test_mesh_small):
        benchmark_vertices, benchmark_simplices, _ = test_mesh_small
        adjacency_data = preprocessing.get_adjacent_vertex_data(
            benchmark_simplices, benchmark_vertices.shape[0]
        )
        assert jnp.allclose(adjacency_data, adjacency_data_for_test_mesh_small)


# ==================================================================================================
class TestCoreFunctions:
    # ----------------------------------------------------------------------------------------------
    @staticmethod
    @pytest.mark.parametrize("order", [1, 10])
    @pytest.mark.parametrize(
        "input_values, expected_output",
        [(jnp.array((2.57,)), 2.57), (jnp.array((2, 2, jnp.inf)), 2)],
    )
    def test_compute_softmin(input_values, expected_output, order):
        min_arg = jnp.min(input_values)
        output = corefunctions.compute_softmin(input_values, min_arg, order=order)
        assert jnp.allclose(output, expected_output)

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    @pytest.mark.parametrize("order", [1, 10, 1000])
    @pytest.mark.parametrize("input_value", [0, 1, 0.25, 0.75, -1000, 1000, -jnp.inf, jnp.inf])
    def test_bounds_compute_softminmax(input_value, order):
        output = corefunctions.compute_softminmax(input_value, order)
        assert jnp.all(output >= 0)
        assert jnp.all(output <= 1)

    # ----------------------------------------------------------------------------------------------
    @staticmethod
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
            ], dtype=jnp.float32
        )
        output_values = corefunctions.compute_softminmax(input_values, order)
        assert jnp.allclose(output_values, expected_output_values)

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def test_compute_edges():
        test_vertices = jnp.array(([-1, -1], [1, 0], [0.5, 2]))
        expected_edges = (jnp.array([-2, -1]), jnp.array([-1.5, -3]), jnp.array([-0.5, 2]))
        output_edges = corefunctions.compute_edges(0, 1, 2, test_vertices)
        for output_edge, expected_edge in zip(output_edges, expected_edges, strict=True):
            assert jnp.allclose(output_edge, expected_edge)

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def test_compute_optimal_update_parameters(simplex_data_for_lambda):
        *input_parameters, expected_lambda_values = simplex_data_for_lambda
        output_lambda_values = corefunctions._compute_optimal_update_parameters(*input_parameters)
        for output_lambda, expected_lambda in zip(
            output_lambda_values, expected_lambda_values, strict=True
        ):
            assert jnp.allclose(output_lambda, expected_lambda)

    # ----------------------------------------------------------------------------------------------
    @staticmethod
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
        monkeypatch, mock_simplex_data, lambda_input, lambda_expected
    ):
        monkeypatch.setattr(
            "eikonax.corefunctions._compute_optimal_update_parameters", lambda *_: lambda_input
        )
        output_lambda_values = corefunctions.compute_optimal_update_parameters_hard(
            *mock_simplex_data
        )
        assert jnp.allclose(output_lambda_values, lambda_expected)

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def test_compute_optimal_update_parameters_soft(monkeypatch, mock_simplex_data):
        order, cutoff = 1, 1
        lambda_input = [-1.1, 2.1]
        monkeypatch.setattr(
            "eikonax.corefunctions._compute_optimal_update_parameters", lambda *_: lambda_input
        )
        output_lambda_values = corefunctions.compute_optimal_update_parameters_soft(
            *mock_simplex_data, order, cutoff
        )
        assert jnp.allclose(output_lambda_values, jnp.array((0, 1, -1, -1)))

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def test_compute_fixed_update(simplex_data_for_update):
        *input_data, expected_update_value = simplex_data_for_update
        update_value = corefunctions.compute_fixed_update(*input_data)
        assert jnp.allclose(update_value, expected_update_value)

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "vertex_update_fixture", ["vertex_update_without_softmin", "vertex_update_with_softmin"]
    )
    def test_compute_vertex_update_candidates(vertex_update_fixture, request):
        vertex_update_data, use_soft_update, solution_values, expected_update_candidates = (
            request.getfixturevalue(vertex_update_fixture)
        )
        vertices, adjacency_data, tensor_field, softminmax_order, softminmax_cutoff = (
            vertex_update_data
        )

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

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def test_grad_update_solution():
        assert False

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def test_grad_update_parameter():
        assert False

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def test_grad_update_lambda():
        assert False

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def test_jac_lambda_soft_solution():
        assert False

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def test_jac_lambda_soft_parameter():
        assert False

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def test_get_adjacent_vertex_data():
        assert False


# ==================================================================================================
class TestTensorField:
    @staticmethod
    def test_assemble_linear_scalar_simplex_tensor():
        assert False

    @staticmethod
    def test_derivative_linear_scalar_simplex_tensor():
        assert False

    @staticmethod
    def test_map_linear_scalar_map():
        assert False
