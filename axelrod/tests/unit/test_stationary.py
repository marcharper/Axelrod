"""Test for the memoryone strategies."""

import unittest

from numpy.testing import assert_array_equal, assert_array_almost_equal

import axelrod
from axelrod import Game, simulate_play
from .test_player import TestPlayer, test_four_vector

from axelrod.strategies.stationary import *

C, D = 'C', 'D'
game = Game()
(R, P, S, T) = game.RPST()
payoffs = numpy.array([R, S, T, P])


class TestStationaryFunctions(unittest.TestCase):

    def test_stationary(self):
        ep = 0.05
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
        expected_response = [0, 0.5, 0, 0.5]
        response = compute_response_four_vector(vec, mode='t')
        assert_array_almost_equal(response, expected_response)

        vec = [0, 0, 0, 0]
        expected_response = [0.5, 0, 0.5, 0]
        response = compute_response_four_vector(vec, mode='t')
        assert_array_almost_equal(response, expected_response)

        vec = [1, 0, 1, 0]
        expected_response = [1, 0.85, 1, 1]
        response = compute_response_four_vector(vec, mode='t')
        assert_array_almost_equal(response, expected_response)
        expected_response = [0.5] * 4
        response = compute_response_four_vector(vec, mode='d')
        assert_array_almost_equal(response, expected_response)

        vec = [1, 0, 0, 1]
        expected_response = [2. / 3, 0, 0, 2. / 3]
        response = compute_response_four_vector(vec, mode='t')
        assert_array_almost_equal(response, expected_response)
        expected_response = [0.15106, 0, 0, 0]
        response = compute_response_four_vector(vec, mode='d')
        assert_array_almost_equal(response, expected_response)

        vec = [8./9, 0.5, 1./3, 0]
        expected_response = [1, 1, 0.8325788, 1]
        response = compute_response_four_vector(vec, mode='t')
        assert_array_almost_equal(response, expected_response)
        expected_response = [0.32424251, 0.06666674, 0.281818, 0]
        response = compute_response_four_vector(vec, mode='d')
        assert_array_almost_equal(response, expected_response)

        vec = [1, 1./8, 1., 0.25]
        expected_response = [1, 0.755986, 1, 0.83158]
        response = compute_response_four_vector(vec, mode='t')
        assert_array_almost_equal(response, expected_response)
        expected_response = [0.42398642, 0.25958896, 0, 0]
        response = compute_response_four_vector(vec, mode='d')
        assert_array_almost_equal(response, expected_response)


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
        #self.assertEqual(p1.history, p2.history)
        # Should have computed a response strategy
        simulate_play(p1, p2)
        simulate_play(p1, p2)
        self.assertIsNotNone(p1._response_four_vector)


class TestStationaryMaxDiff(TestPlayer):

    name = "Stationary Max Diff"
    player = axelrod.StationaryMaxDiff
    stochastic = True

    def test_strategy(self):
        self.first_play_test(C)

