"""Tests for the Ecosystem class"""

import unittest

import axelrod


class TestEcosystem(unittest.TestCase):

    cooperators = axelrod.Tournament('cooperators', [
        axelrod.Cooperator(),
        axelrod.Cooperator(),
        axelrod.Cooperator(),
        axelrod.Cooperator(),
    ])
    defector_wins = axelrod.Tournament('defector_wins', [
        axelrod.Cooperator(),
        axelrod.Cooperator(),
        axelrod.Cooperator(),
        axelrod.Defector(),
    ])

    def setUp(self):
        self.res_cooperators = self.cooperators.play()
        self.res_defector_wins = self.defector_wins.play()

    def test_init(self):
        """Are the populations created correctly?"""

        eco = axelrod.Ecosystem(self.res_cooperators)
        pops = eco.population_sizes
        self.assertEquals(eco.nplayers, 4)
        self.assertEquals(len(pops), 1)
        self.assertEquals(len(pops[0]), 4)
        self.assertAlmostEqual(sum(pops[0]), 1.0)
        self.assertEquals(list(set(pops[0])), [0.25])

    def test_cooperators(self):
        """Are cooperators stable over time?"""

        eco = axelrod.Ecosystem(self.res_cooperators)
        eco.reproduce(100)
        pops = eco.population_sizes
        self.assertEquals(len(pops), 101)
        for p in pops:
            self.assertEquals(len(p), 4)
            self.assertEquals(sum(p), 1.0)
            self.assertEquals(list(set(p)), [0.25])

    def test_defector_wins(self):
        """Does one defector win over time?"""

        eco = axelrod.Ecosystem(self.res_defector_wins)
        eco.reproduce(1000)
        pops = eco.population_sizes
        self.assertEquals(len(pops), 1001)
        for p in pops:
            self.assertEquals(len(p), 4)
            self.assertAlmostEquals(sum(p), 1.0)
        last = pops[-1]
        self.assertAlmostEquals(last[0], 0.0)
        self.assertAlmostEquals(last[1], 0.0)
        self.assertAlmostEquals(last[2], 0.0)
        self.assertAlmostEquals(last[3], 1.0)
