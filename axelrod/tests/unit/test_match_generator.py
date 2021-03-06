import unittest
import axelrod

from hypothesis import given, example
from hypothesis.strategies import floats, random_module

test_strategies = [
    axelrod.Cooperator,
    axelrod.TitForTat,
    axelrod.Defector,
    axelrod.Grudger,
    axelrod.GoByMajority
]
test_turns = 100
test_game = axelrod.Game()


class TestTournamentType(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.players = [s() for s in test_strategies]

    def test_init_with_clone(self):
        tt = axelrod.MatchGenerator(
            self.players, test_turns, test_game, axelrod.DeterministicCache())
        self.assertEqual(tt.players, self.players)
        self.assertEqual(tt.turns, test_turns)
        player = tt.players[0]
        opponent = tt.opponents[0]
        self.assertEqual(player.name, opponent.name)
        self.assertNotEqual(player, opponent)
        # Check that the two player instances are wholly independent
        opponent.name = 'Test'
        self.assertNotEqual(player.name, opponent.name)


class TestRoundRobin(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.players = [s() for s in test_strategies]

    def test_build_single_match(self):
        rr = axelrod.RoundRobinMatches(
            self.players, test_turns, test_game, axelrod.DeterministicCache())
        match = rr.build_single_match((self.players[0], self.players[1]))
        self.assertIsInstance(match, axelrod.Match)
        self.assertEqual(len(match), rr.turns)


    def test_build_matches(self):
        rr = axelrod.RoundRobinMatches(
            self.players, test_turns, test_game, axelrod.DeterministicCache())
        matches = rr.build_matches()
        match_definitions = [
            (index_pair) for index_pair, match in matches]
        expected_match_definitions = [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 2),
            (2, 3),
            (2, 4),
            (3, 3),
            (3, 4),
            (4, 4)
        ]
        self.assertEqual(sorted(match_definitions), expected_match_definitions)


class TestProbEndRoundRobin(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.players = [s() for s in test_strategies]

    @given(prob_end=floats(min_value=0, max_value=1), rm=random_module())
    def test_build_matches(self, prob_end, rm):
        rr = axelrod.ProbEndRoundRobinMatches(
            self.players, prob_end, test_game, axelrod.DeterministicCache())
        matches = rr.build_matches()
        match_definitions = [
            (index_pair) for index_pair, match in matches]
        expected_match_definitions = [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 2),
            (2, 3),
            (2, 4),
            (3, 3),
            (3, 4),
            (4, 4)
        ]
        self.assertEqual(sorted(match_definitions), expected_match_definitions)

    @given(prob_end=floats(min_value=0.1, max_value=0.5), rm=random_module())
    def test_build_matches_different_length(self, prob_end, rm):
        """
        If prob end is not 0 or 1 then the matches should all have different
        length

        Theoretically, this test could fail as it's probabilistically possible
        to sample all games with same length.
        """
        rr = axelrod.ProbEndRoundRobinMatches(
            self.players, prob_end, test_game, axelrod.DeterministicCache())
        matches = rr.build_matches()
        match_lengths = [len(match) for index_pair, match in matches]
        self.assertNotEqual(min(match_lengths), max(match_lengths))

    @given(prob_end=floats(min_value=0, max_value=1), rm=random_module())
    def test_sample_length(self, prob_end, rm):
        rr = axelrod.ProbEndRoundRobinMatches(
            self.players, prob_end, test_game, axelrod.DeterministicCache())
        self.assertGreaterEqual(rr.sample_length(prob_end), 1)
        try:
            self.assertIsInstance(rr.sample_length(prob_end), int)
        except AssertionError:
            self.assertEqual(rr.sample_length(prob_end), float("inf"))

    @given(prob_end=floats(min_value=.1, max_value=1), rm=random_module())
    def test_build_single_match(self, prob_end, rm):
        rr = axelrod.ProbEndRoundRobinMatches(
            self.players, prob_end, test_game, axelrod.DeterministicCache())
        match = rr.build_single_match((self.players[0], self.players[1]))

        self.assertIsInstance(match, axelrod.Match)
        self.assertGreaterEqual(len(match), 1)
        self.assertLessEqual(len(match), float("inf"))
