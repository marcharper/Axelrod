from __future__ import absolute_import, unicode_literals, print_function

import os

from .tournament import *
from .deterministic_cache import DeterministicCache
from .plot import *
from .ecosystem import *
from .utils import *


class TournamentManager(object):
    """A class to manage and create tournaments."""

    plot_types = {'boxplot': "Payoffs. ", 'payoff': "Payoffs. ",
                  'winplot': "Wins. ", 'sdvplot': "Std Payoffs. ",
                  'pdplot': "Payoff differences. ", 'lengthplot': "Lengths. "}

    ecoturns = {
            'basic_strategies': 1000,
            'cheating_strategies': 10,
            'ordinary_strategies': 1000,
            'strategies': 10,
        }

    def __init__(self, output_directory, with_ecological,
                 pass_cache=True, load_cache=True, save_cache=False,
                 cache_file='./cache.txt', image_format="svg"):
        self._tournaments = []
        self._ecological_variants = []
        self._logger = logging.getLogger(__name__)
        self._output_directory = output_directory
        self._with_ecological = with_ecological
        self._pass_cache = pass_cache
        self._save_cache = save_cache
        self._cache_file = cache_file
        self._deterministic_cache = DeterministicCache()
        self._load_cache = False
        self._image_format = image_format

        if load_cache and not save_cache:
            self.load_cache = self._load_cache_from_file(cache_file)

    @staticmethod
    def one_player_per_strategy(strategies):
        return [strategy() for strategy in strategies]

    def add_tournament(self, name, players, game=None, turns=200,
                       repetitions=10, processes=None, noise=0,
                       with_morality=True):
        tournament = Tournament(
            name=name,
            players=players,
            turns=turns,
            repetitions=repetitions,
            processes=processes,
            noise=noise,
            with_morality=with_morality)
        self._tournaments.append(tournament)

    def run_tournaments(self):
        t0 = time.time()
        for tournament in self._tournaments:
            self._run_single_tournament(tournament)
        if self._save_cache and not tournament.noise:
            self._save_cache_to_file(self._deterministic_cache, self._cache_file)
        self._logger.info(timed_message('Finished all tournaments', t0))

    def _run_single_tournament(self, tournament):
        self._logger.info(
                'Starting {} tournament: '.format(tournament.name) + self._tournament_label(tournament)
            )

        t0 = time.time()

        if not tournament.noise and self._pass_cache and self._valid_cache(tournament.turns):
            self._logger.debug('Passing cache with %d entries to %s tournament' %
                            (len(self._deterministic_cache), tournament.name))
            tournament.deterministic_cache = self._deterministic_cache
            if self._load_cache:
                tournament.prebuilt_cache = True
        else:
            self._logger.debug('Cache is not valid for %s tournament' %
                            tournament.name)
        tournament.play()

        self._logger.debug(timed_message('Finished %s tournament' % tournament.name, t0))

        if self._with_ecological:
            ecosystem = Ecosystem(tournament.result_set)
            self.run_ecological_variant(tournament, ecosystem)
        else:
            ecosystem = None

        self._generate_output_files(tournament, ecosystem)
        self._cache_valid_for_turns = tournament.turns

        self._logger.debug('Cache now has %d entries' %
                        len(self._deterministic_cache))

        self._logger.info(
            timed_message('Finished all %s tasks' % tournament.name, t0))

    def _valid_cache(self, turns):
        return ((len(self._deterministic_cache) == 0) or
                (len(self._deterministic_cache) > 0) and
                turns == self._deterministic_cache.turns)

    def run_ecological_variant(self, tournament, ecosystem):
        self._logger.debug(
            'Starting ecological variant of %s' % tournament.name)
        t0 = time.time()
        ecosystem.reproduce(self.ecoturns.get(tournament.name))
        self._logger.debug(
            timed_message('Finished ecological variant of %s' % tournament.name, t0))

    def _generate_output_files(self, tournament, ecosystem=None):
        self._save_csv(tournament)
        self._save_plots(tournament, ecosystem,
                         image_format=self._image_format)

    def _save_csv(self, tournament):
        csv = tournament.result_set.csv()
        file_name = self._output_file_path(
                tournament.name, 'csv')
        with open(file_name, 'w') as f:
            f.write(csv)

    def _save_plots(self, tournament, ecosystem=None, image_format="svg"):
        results = tournament.result_set
        plot = Plot(results)
        if not plot.matplotlib_installed:
            self._logger.error('The matplotlib library is not installed. '
                            'No plots will be produced')
            return
        label = self._tournament_label(tournament)
        for plot_type, name in self.plot_types.items():
            title = name + label
            figure = getattr(plot, plot_type)(title=title)
            file_name = self._output_file_path(
                tournament.name + '_' + plot_type, image_format)
            self._save_plot(figure, file_name)
        if ecosystem is not None:
            title = "Eco. " + label
            figure = plot.stackplot(ecosystem, title=title)
            file_name = self._output_file_path(
                    tournament.name + '_reproduce', image_format)
            self._save_plot(figure, file_name)

    def _tournament_label(self, tournament):
        """A label for the tournament for the corresponding title plots"""
        return "Turns: {}, Repetitions: {}, Strategies: {}.".format(tournament.turns,
                                                       tournament.repetitions,
                                                       len(tournament.players))

    def _output_file_path(self, file_name, file_extension):
        return os.path.join(
            self._output_directory,
            file_name + '.' + file_extension)

    @staticmethod
    def _save_plot(figure, file_name, dpi=400):
        figure.savefig(file_name, bbox_inches='tight', dpi=dpi)
        figure.clf()
        plt.close(figure)

    def _save_cache_to_file(self, cache, file_name):
        self._logger.debug(
            'Saving cache with %d entries to %s' % (len(cache), file_name))
        cache.save(file_name)
        return True

    def _load_cache_from_file(self, file_name):
        try:
            self._deterministic_cache.load(file_name)
            self._logger.debug(
                'Loaded cache with %d entries' % len(self._deterministic_cache))
            return True
        except IOError:
            self._logger.debug('Cache file not found. Starting with empty cache')
            return False


class ProbEndTournamentManager(TournamentManager):
    """A class to manage and create probabilistic ending tournaments."""

    ecoturns = {
            'basic_strategies_prob_end': 1000,
            'cheating_strategies_prob_end': 10,
            'ordinary_strategies_prob_end': 1000,
            'strategies_prob_end': 10,
        }

    def add_tournament(self, name, players, game=None, prob_end=.01,
                       repetitions=10, processes=None, noise=0,
                       with_morality=True):
        tournament = ProbEndTournament(
            name=name,
            players=players,
            prob_end=prob_end,
            repetitions=repetitions,
            processes=processes,
            noise=noise,
            with_morality=with_morality)
        self._tournaments.append(tournament)

    def _tournament_label(self, tournament):
        """A label for the tournament for the corresponding title plots"""
        return "Prob end: {}, Repetitions: {}, Strategies: {}.".format(tournament.prob_end,
                                                       tournament.repetitions,
                                                       len(tournament.players))
