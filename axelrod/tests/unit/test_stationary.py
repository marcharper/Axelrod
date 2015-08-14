"""Test for the memoryone strategies."""

import unittest

from numpy.testing import assert_array_equal, assert_array_almost_equal

import axelrod
from axelrod import Game, simulate_play
from .test_player import TestHeadsUp, TestPlayer, test_four_vector

from axelrod.strategies.stationary import *

C, D = 'C', 'D'
game = Game()
(R, P, S, T) = game.RPST()
payoffs = numpy.array([R, S, T, P])


class TestStationaryFunctions(unittest.TestCase):

    def test_stationary(self):
        ep = 0.01
        vec = [1. - ep, ep, 1. - ep, ep]

        # Test approx_stationary
        transitions = compute_transitions(vec, vec)
        stationary = approximate_stationary(transitions)
        expected = numpy.array([0.25] * 4)
        assert_array_almost_equal(stationary, expected)

        # Test exact_stationary
        stationary = exact_stationary(vec, vec)
        assert_array_almost_equal(stationary, expected)

        # Test compute_stationary
        stationary = compute_stationary(vec, vec)
        assert_array_almost_equal(stationary, expected)

        # Test stationary_payoff
        payoff = stationary_payoff(vec, vec, payoffs)
        self.assertEqual(payoff, numpy.dot(stationary, payoffs))

    def test_response(self):
        vec = [1, 1, 1, 1]
        expected_response = [0, 0.8, 0, 0]
        response = compute_response_four_vector(vec, mode='t')
        assert_array_almost_equal(response, expected_response, decimal=2)
        response = compute_response_four_vector(vec, mode='d')
        assert_array_almost_equal(response, expected_response, decimal=2)

        vec = [1, 0, 0, 0]
        expected_response = [1, 0, 1, 1]
        response = compute_response_four_vector(vec, mode='t')
        assert_array_almost_equal(response, expected_response, decimal=2)
        expected_response = [0.85, 0, 0, 0]
        response = compute_response_four_vector(vec, mode='d')
        assert_array_almost_equal(response, expected_response, decimal=2)

        vec = [0, 0, 0, 1]
        expected_response = [0.1562, 0, 0, 0]
        response = compute_response_four_vector(vec, mode='t')
        assert_array_almost_equal(response, expected_response, decimal=2)
        expected_response = [0.13322, 0, 0, 0]
        response = compute_response_four_vector(vec, mode='d')
        assert_array_almost_equal(response, expected_response, decimal=2)


        vec = [0, 0, 0, 0]
        expected_response = [0.8, 0, 0, 0]
        response = compute_response_four_vector(vec, mode='t')
        assert_array_almost_equal(response, expected_response, decimal=2)
        response = compute_response_four_vector(vec, mode='d')
        assert_array_almost_equal(response, expected_response, decimal=2)


        vec = [1, 0, 1, 0]
        expected_response = [1, 1, 0.92787, 0.82735]
        response = compute_response_four_vector(vec, mode='t')
        assert_array_almost_equal(response, expected_response, decimal=2)
        expected_response = [0, 0, 0, 0]
        response = compute_response_four_vector(vec, mode='d')
        assert_array_almost_equal(response, expected_response, decimal=2)

        vec = [1, 0, 0, 1]
        expected_response = [1, 0, 0.73367, 1]
        response = compute_response_four_vector(vec, mode='t')
        assert_array_almost_equal(response, expected_response, decimal=2)
        expected_response = [1, 0, 0, 0]
        response = compute_response_four_vector(vec, mode='d')
        assert_array_almost_equal(response, expected_response, decimal=2)

        vec = [8./9, 0.5, 1./3, 0]
        expected_response = [1, 1, 0.8638, 0.858]
        response = compute_response_four_vector(vec, mode='t')
        assert_array_almost_equal(response, expected_response, decimal=2)
        expected_response = [0, 0, 0, 0]
        response = compute_response_four_vector(vec, mode='d')
        assert_array_almost_equal(response, expected_response, decimal=2)

        vec = [1, 1./8, 1., 0.25]
        expected_response = [1, 1, 0.925, 0.8208]
        response = compute_response_four_vector(vec, mode='t')
        assert_array_almost_equal(response, expected_response, decimal=2)
        expected_response = [0., 0.35456, 0, 0]
        response = compute_response_four_vector(vec, mode='d')
        assert_array_almost_equal(response, expected_response, decimal=2)


class TestStationaryMax(TestPlayer):

    name = "Stationary Max"
    player = axelrod.StationaryMax
    stochastic = True

    def test_strategy(self):
        self.first_play_test(C)
        # Test initial TFT
        p1 = self.player()
        p2 = axelrod.Cooperator()
        p1.tournament_length = 200
        for i in range(15):
            simulate_play(p1, p2)
        # Should have computed a response strategy
        simulate_play(p1, p2)
        self.assertIsNotNone(p1._response_four_vector)


class TestStationaryMaxDiff(TestPlayer):

    name = "Stationary Max Diff"
    player = axelrod.StationaryMaxDiff
    stochastic = True

    def test_strategy(self):
        self.first_play_test(C)


class TestStationaryMaxvsTFT(TestHeadsUp):
    """Test TFT vs WSLS"""
    def test_rounds(self):
        outcomes = [(C, C)] * 15 + [(D, C), (D, D), (D, D)]
        self.versus_test(axelrod.StationaryMax, axelrod.TitForTat, outcomes)

class TestStationaryMaxvsALLC(TestHeadsUp):
    """Test TFT vs WSLS"""
    def test_rounds(self):
        outcomes = [(C, C)] * 15 + [(D, C)]
        self.versus_test(axelrod.StationaryMax, axelrod.Cooperator, outcomes)
