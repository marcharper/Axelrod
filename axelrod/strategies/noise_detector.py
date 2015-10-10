import math

from axelrod import Player


def upper_threshold(mismatch_count, total_plays, z=1.):
    """Use a binomial confidence interval (Wilson score interval) to determine
    a reasonable noise threshold."""
    if total_plays == 0:
        return 0
    p = float(mismatch_count) / total_plays
    n = total_plays
    upper = (1 / (1. + z * z / n)) * (p + z * z / (n * n) + \
        z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n) ) )

    if upper > 1:
        upper = 1
    return upper


def track_plays(member_function):
    """A decorator to add history tracking to a strategy."""
    def decorator(self, opponent):
        play = member_function(self, opponent)
        self.submitted_plays.append(play)
        return play
    return decorator


class NoiseDetector(Player):
    """
    A player that tracks its own plays to determine if the environment is noisy.
    FoolMeOnce and be more forgiving because of the noise. Could be improved
    by maintaining the noise estimate, and by a more sophisticated statistical
    test.
    """

    name = 'NoiseDetector'
    classifier = {
        'memory_depth': float('inf'),
        'stochastic': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def __init__(self):
        Player.__init__(self)
        self.submitted_plays = []
        self.mismatch_count = 0
        self.is_defecting = False

    @track_plays
    def strategy(self, opponent):
        if not len(self.history):
            return 'C'
        if self.is_defecting:
            return 'D'

        # Update mismatch_count
        if self.history[-1] != self.submitted_plays[-1]:
            self.mismatch_count += 1

        # Defect if the opponent has defected more than can be reasonably
        # explained by noise
        defection_threshold = upper_threshold(self.mismatch_count,
                                                len(self.history))
        defection_proportion = float(opponent.defections) / len(opponent.history)
        if defection_proportion >= 0.5 * defection_threshold:
            self.is_defecting = True
            return 'D'

        return 'C'

    def reset(self):
        Player.reset(self)
        self.submitted_plays = []
        self.mismatch_count = 0
        self.is_defecting = False
