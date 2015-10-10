"""Tests for the hunter strategy."""

import random

import axelrod
from axelrod.strategies.noise_detector import upper_threshold
from .test_player import TestPlayer


C, D = 'C', 'D'


class TestNoiseDetector(TestPlayer):

    name = "NoiseDetector"
    player = axelrod.NoiseDetector
    expected_classifier = {
        'memory_depth': float('inf'),  # Long memory
        'stochastic' : False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def test_track_plays(self):
        p1 = self.player()
        p2 = axelrod.Cooperator()
        p1.play(p2) # plays 'C'
        p1.history = ['D']
        self.assertEqual(p1.mismatch_count, 0)
        self.assertEqual(p1.submitted_plays, [C])
        # Next play should trigger a mismatch detection
        p1.strategy(p2)
        self.assertEqual(p1.mismatch_count, 1)

    def test_upper_threshold(self):
        t = upper_threshold(0, 0)
        self.assertEqual(t, 0)
        t = upper_threshold(0, 1)
        self.assertEqual(t, 0.75)
        t = upper_threshold(1, 1)
        self.assertEqual(t, 1.)
        t = upper_threshold(1, 2)
        self.assertEqual(t, 0.7886751345948128)
        t = upper_threshold(0, 2)
        self.assertEqual(t, 0.3333333333333333)
        t = upper_threshold(2, 2)
        self.assertEqual(t, 1)
        t = upper_threshold(1, 10)
        self.assertEqual(t, 0.19748913904330553)

    def test_strategy(self):
        self.first_play_test(C)

        p1 = self.player()
        p2 = axelrod.Cooperator()
        p1.play(p2) # plays 'C'
        p1.play(p2) # plays 'C'
        self.assertEqual(p1.history, [C, C])
        self.assertEqual(p1.submitted_plays, [C, C])

        p1 = self.player()
        p2 = axelrod.Cooperator()
        p1.play(p2) # plays 'C'
        p1.history[-1] = D
        p1.play(p2) # plays 'C'
        p1.history[-1] = D

        # mismatches don't trigger defection
        play = p1.strategy(p2) # plays 'C'
        self.assertEqual(play, C)

        p1 = self.player()
        p2 = axelrod.Defector()
        p1.play(p2) # (C, D)
        p1.play(p2) # (C, D)
        # Too many defections, not enough mismatches
        p1.play(p2)
        play = p1.history[-1]
        self.assertEqual(play, D)
        # Throw in some mismatches due to noise
        p1.play(p2) # plays 'C'
        p1.history[-1] = C
        p1.play(p2) # plays 'C'
        p1.history[-1] = C
        p1.play(p2) # plays 'C'
        p1.history[-1] = C
        # Back to cooperating
        p1.play(p2)
        play = p1.history[-1]
        self.assertEqual(play, D)

    def test_reset(self):
        p1 = self.player()
        p1.reset()
        self.assertEqual(p1.mismatch_count, 0)
        self.assertEqual(p1.submitted_plays, [])
