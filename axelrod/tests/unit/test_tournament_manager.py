import unittest
import axelrod


class TestTournamentManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_output_directory = './assets/'
        cls.test_with_ecological = True
        cls.test_tournament_name = 'test_tournament'
        cls.test_file_name = 'test_file_name'
        cls.test_file_extenstion = 'png'
        cls.test_strategies = [axelrod.Defector, axelrod.Cooperator]
        cls.test_players = [axelrod.Defector(), axelrod.Cooperator()]

        cls.expected_output_file_path = './assets/test_file_name.png'
        cls.mgr_class = axelrod.TournamentManager

    def test_init(self):
        mgr = self.mgr_class(
            self.test_output_directory,
            self.test_with_ecological,
            load_cache=False)
        self.assertEqual(mgr._output_directory, self.test_output_directory)
        self.assertEqual(mgr._tournaments, [])
        self.assertEqual(mgr._with_ecological, self.test_with_ecological)
        self.assertTrue(mgr._pass_cache)

    def test_one_player_per_strategy(self):
        mgr = self.mgr_class(
            self.test_output_directory,
            self.test_with_ecological,
            load_cache=False)
        players = mgr.one_player_per_strategy(self.test_strategies)
        self.assertIsInstance(players[0], axelrod.Defector)
        self.assertIsInstance(players[1], axelrod.Cooperator)

    def test_output_file_path(self):
        mgr = self.mgr_class(
            self.test_output_directory,
            self.test_with_ecological,
            load_cache=False)
        output_file_path = mgr._output_file_path(
            self.test_file_name, self.test_file_extenstion)
        self.assertEqual(output_file_path, self.expected_output_file_path)

    def test_add_tournament(self):
        mgr = self.mgr_class(
            self.test_output_directory,
            self.test_with_ecological,
            load_cache=False)
        mgr.add_tournament(
            players=self.test_players, name=self.test_tournament_name)
        self.assertEqual(len(mgr._tournaments), 1)
        self.assertIsInstance(mgr._tournaments[0], axelrod.Tournament)
        self.assertEqual(mgr._tournaments[0].name, self.test_tournament_name)

    def test_valid_cache(self):
        mgr = self.mgr_class(
            output_directory=self.test_output_directory,
            with_ecological=self.test_with_ecological,
            load_cache=False)
        mgr.add_tournament(
                players=self.test_players, name=self.test_tournament_name)
        self.assertTrue(mgr._valid_cache(200))
        mgr._deterministic_cache[(axelrod.Cooperator, axelrod.Defector)] = [('C', 'D')]
        self.assertFalse(mgr._valid_cache(200))
        mgr._deterministic_cache.turns = 500
        self.assertFalse(mgr._valid_cache(200))
        self.assertTrue(mgr._valid_cache(500))

    def test_tournament_label(self):
        tournament = axelrod.Tournament(self.test_players, turns=20,
                                        repetitions=2)
        mgr = self.mgr_class(
            output_directory=self.test_output_directory,
            with_ecological=self.test_with_ecological, load_cache=False)
        expected_label = "Turns: {}, Repetitions: {}, Strategies: {}.".format(tournament.turns,
                tournament.repetitions, len(tournament.players))

        self.assertEqual(mgr._tournament_label(tournament), expected_label)


class TestProbEndTournamentManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_output_directory = './assets/'
        cls.test_with_ecological = True
        cls.test_tournament_name = 'test_prob_tournament'
        cls.test_file_name = 'test_prob_end_file_name'
        cls.test_file_extenstion = 'png'
        cls.test_strategies = [axelrod.Defector, axelrod.Cooperator]
        cls.test_players = [axelrod.Defector(), axelrod.Cooperator()]

        cls.expected_output_file_path = './assets/test__prob_end_file_name.png'
        cls.mgr_class = axelrod.ProbEndTournamentManager

    def test_add_tournament(self):
        mgr = self.mgr_class(
            self.test_output_directory,
            self.test_with_ecological,
            load_cache=False)
        mgr.add_tournament(
            players=self.test_players, name=self.test_tournament_name)
        self.assertEqual(len(mgr._tournaments), 1)
        self.assertIsInstance(mgr._tournaments[0], axelrod.ProbEndTournament)
        self.assertEqual(mgr._tournaments[0].name, self.test_tournament_name)

    def test_tournament_label(self):
        tournament = axelrod.ProbEndTournament(self.test_players, prob_end=.5,
                                               repetitions=2)
        mgr = self.mgr_class(
            output_directory=self.test_output_directory,
            with_ecological=self.test_with_ecological, load_cache=False)
        expected_label = "Prob end: {}, Repetitions: {}, Strategies: {}.".format(tournament.prob_end,
                    tournament.repetitions,
                    len(tournament.players))
        self.assertEqual(mgr._tournament_label(tournament), expected_label)
