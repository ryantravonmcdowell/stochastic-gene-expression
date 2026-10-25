import numpy as np
import pytest
from scipy.special import comb

from burstkin import models
from burstkin.network import Reaction, ReactionNetwork


def test_telegraph_stoichiometry():
    net = models.telegraph()
    expected = np.array([
        [-1, 1, 0, 0],  # G_off
        [1, -1, 0, 0],  # G_on
        [0, 0, 1, -1],  # M
    ])
    np.testing.assert_array_equal(net.stoichiometry, expected)
    assert net.is_linear()
    assert not models.annihilation().is_linear()


def test_combinatorial_propensities_count_reactant_combinations():
    net = ReactionNetwork(
        species=("A", "B"),
        reactions=(
            Reaction({"A": 3}, {"B": 1}, "c1"),
            Reaction({"A": 1, "B": 1}, {}, "c2"),
            Reaction({}, {"A": 1}, "c3"),
        ),
        parameters={"c1": 2.0, "c2": 0.5, "c3": 7.0},
    )
    x = np.array([[a, b] for a in range(6) for b in range(4)])
    a = net.propensities(x)
    np.testing.assert_allclose(a[:, 0], 2.0 * comb(x[:, 0], 3))
    np.testing.assert_allclose(a[:, 1], 0.5 * x[:, 0] * x[:, 1])
    np.testing.assert_allclose(a[:, 2], 7.0)


def test_macroscopic_jacobian_matches_finite_differences():
    net = models.autoregulation()
    phi = np.array([0.3, 0.7, 12.0])
    jac = net.macroscopic_jacobian(phi)
    h = 1e-6
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        fd = (net.macroscopic_propensities(phi + e) - net.macroscopic_propensities(phi - e)) / (2 * h)
        np.testing.assert_allclose(jac[:, k], fd, rtol=1e-7, atol=1e-9)


def test_parameter_overrides_and_validation():
    net = models.telegraph(k_on=0.5)
    assert net.rate_vector({"k_on": 2.0})[0] == 2.0
    assert net.with_parameters(k_syn=99.0).parameters["k_syn"] == 99.0
    with pytest.raises(KeyError):
        net.rate_vector({"nope": 1.0})
    with pytest.raises(ValueError):
        ReactionNetwork(("A",), (Reaction({"B": 1}, {}, "k"),), {"k": 1.0})
    with pytest.raises(ValueError):
        ReactionNetwork(("A",), (Reaction({"A": 1}, {}, "missing"),), {"k": 1.0})


def test_state_builder():
    net = models.autoregulation()
    np.testing.assert_array_equal(net.state(G=1, P=5), [1, 0, 5])
