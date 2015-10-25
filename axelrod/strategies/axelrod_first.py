"""
Additional strategies from Axelrod's two tournaments.
"""

from math import sqrt
import random

from axelrod import Game, Player, Actions, random_choice, flip_action

from.memoryone import MemoryOnePlayer

C, D = Actions.C, Actions.D


class Davis(Player):
    """A player starts by cooperating for 10 rounds then plays Grudger,
    defecting if at any point the opponent has defected."""

    name = 'Davis'
    classifier = {
        'memory_depth': float('inf'),  # Long memory
        'stochastic': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def __init__(self, rounds_to_cooperate=10):
        """
        Parameters
        ----------
        rounds_to_cooperate: int, 10
           The number of rounds to cooperate initially
        """
        Player.__init__(self)
        self._rounds_to_cooperate = rounds_to_cooperate
        self.init_args = (self._rounds_to_cooperate,)

    def strategy(self, opponent):
        """Begins by playing C, then plays D for the remaining rounds if the
        opponent ever plays D."""
        if len(self.history) < self._rounds_to_cooperate:
            return C
        if opponent.defections:
            return D
        return C


class RevisedDowning(Player):
    """Revised Downing."""

    name = "Revised Downing"

    classifier = {
        'memory_depth': float('inf'),
        'stochastic': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def __init__(self, revised=True):
        Player.__init__(self)
        self.revised = revised
        self.good = 1.0
        self.bad = 0.0
        self.nice1 = 0
        self.nice2 = 0
        self.total_C = 0 # note the same as self.cooperations
        self.total_D = 0 # note the same as self.defections
        self.init_args = (revised,)

    def strategy(self, opponent):
        round_number = len(self.history) + 1
        # According to internet sources, the original implementation defected
        # on the first two moves. Otherwise it wins (if this code is removed
        # and the comment restored.
        # http://www.sci.brooklyn.cuny.edu/~sklar/teaching/f05/alife/notes/azhar-ipd-Oct19th.pdf

        if self.revised:
            if round_number == 1:
                self.move = C
                return self.move
        elif not self.revised:
            if round_number <= 2:
                self.move = D
                return self.move

        # Update various counts
        if round_number > 2:
            if self.history[-1] == D:
                if opponent.history[-1] == C:
                    self.nice2 += 1
                self.total_D += 1
                self.bad = float(self.nice2) / self.total_D
            else:
                if opponent.history[-1] == C:
                    self.nice1 += 1
                self.total_C += 1
                self.good = float(self.nice1) / self.total_C
        # Make a decision based on the accrued counts
        c = 6.0 * self.good - 8.0 * self.bad - 2
        alt = 4.0 * self.good - 5.0 * self.bad - 1
        if (c >= 0 and c >= alt):
            self.move = C
        elif (c >= 0 and c < alt) or (alt >= 0):
            self.move = flip_action(self.move)
        else:
            self.move = D
        return self.move

    def reset(self):
        Player.reset(self)
        self.good = 1.0
        self.bad = 0.0
        self.nice1 = 0
        self.nice2 = 0
        self.total_C = 0 # note the same as self.cooperations
        self.total_D = 0 # note the same as self.defections


class Feld(Player):
    """
    Defects when opponent defects. Cooperates with a probability that decreases
    to 0.5 at round 200.
    """

    name = "Feld"
    classifier = {
        'memory_depth': 200, # Varies actually, eventually becomes depth 1
        'stochastic': True,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def __init__(self, start_coop_prob=1.0, end_coop_prob=0.5,
                 rounds_of_decay=200):
        """
        Parameters
        ----------
        start_coop_prob, float
            The initial probability to cooperate
        end_coop_prob, float
            The final probability to cooperate
        rounds_of_decay, int
            The number of rounds to linearly decrease from start_coop_prob
            to end_coop_prob
        """
        Player.__init__(self)
        self._start_coop_prob = start_coop_prob
        self._end_coop_prob = end_coop_prob
        self._rounds_of_decay = rounds_of_decay
        self.init_args = (start_coop_prob,
                          end_coop_prob,
                          rounds_of_decay)

    def _cooperation_probability(self):
        """It's not clear what the interpolating function is, so we'll do
        something simple that decreases monotonically from 1.0 to 0.5 over
        200 rounds."""
        diff = (self._end_coop_prob - self._start_coop_prob)
        slope = diff / float(self._rounds_of_decay)
        rounds = len(self.history)
        return max(self._start_coop_prob + slope * rounds,
                   self._end_coop_prob)

    def strategy(self, opponent):
        if not opponent.history:
            return C
        if opponent.history[-1] == D:
            return D
        p = self._cooperation_probability()
        return random_choice(p)


class Graaskamp(Player):
    """
    This rule plays tit for tat for 50 moves, defects on move 51, and then plays
    5 more moves of tit for tat. A check is then made to see if the player seems
    to be RANDOM, in which case it defects from then on. A check is also made to
    see if the other is TIT FOR TAT, ANALOGY (a program from the preliminary
    tournament), and its own twin, in which case it plays tit for tat. Otherwise
    it randomly defects every 5 to 15 moves, hoping that enough trust has been
    built up so that the other player will not notice these defections.
    -- Axelrod, "Effective Choice in the Prisoner's Dilemma"

    Warning: this strategy is incomplete.
    Note: Not the same as "Graaskamp and Katzen"
    """

    name = "Graaskamp"
    classifier = {
        'memory_depth': float('inf'),  # Long memory
        'stochastic': True,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def __init__(self):
        self.is_defecting = False
        self.is_TFT = False
        self.reset_defection()
        self.init_args = ()

    def reset_defection(self):
        r = random.random()
        self.defect_in_rounds = int(10 * r + 5)

    def strategy(self, opponent):
        round_number = len(self.history) + 1
        # 50 rounds of TFT
        if round_number == 1:
            return C
        if round_number < 51:
            return D if opponent.history[-1] == D else C
        # Defect on round 51
        if round_number == 51:
            return D
        # 5 more rounds of TFT
        if round_number < 57:
            return D if opponent.history[-1] == D else C
        if round_number == 57:
            # Check if player seems random
            game = self.tournament_attributes["game"]
            opp_total_score = sum(x[0] for x in map(game.score,
                                    zip(opponent.history, self.history)))
            if opp_total_score <= 135:
                # Defect for the remainder.
                self.is_defecting = True
        if self.is_defecting:
            return D
        ### Not yet implemented ###
        # Check for TitForTat and Analogy (definition unknown)
        # If so, act as TFT
        # self.is_TFT = True
        ###
        if self.is_TFT:
            return D if opponent.history[-1] == D else C
        # If we've gotten this far, we are randomly defecting
        # every 5 to 15 rounds, cooperating otherwise
        if self.defect_in_rounds == 0:
            self.reset_defection()
            return D
        self.defect_in_rounds -= 1
        return C

    def reset(self):
        Player.reset(self)
        self.is_defecting = False
        self.is_TFT = False
        self.reset_defection()


class Grofman(Player):
    """
    Cooperate on the first 2 moves. Return opponent's move for the next 5.
    Then cooperate if the last round's moves were the same, otherwise cooperate
    with probability 2/7.
    """

    name = "Grofman"
    classifier = {
        'memory_depth': float('inf'),
        'stochastic': True,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def strategy(self, opponent):
        round_number = len(self.history) + 1
        if round_number < 3:
            return C
        if round_number < 8:
            return opponent.history[-1]
        if self.history[-1] == opponent.history[-1]:
            return C
        return random_choice(2./ 7)



class Joss(MemoryOnePlayer):
    """
    Cooperates with probability 0.9 when the opponent cooperates, otherwise
    emulates Tit-For-Tat.
    """

    name = "Joss"

    def __init__(self, p=0.9):
        """
        Parameters
        ----------
        p, float
            The probability of cooperating when the previous round was (C, C)
            or (D, C), i.e. the opponent cooperated.
        """
        four_vector = (p, 0, p, 0)
        self.p = p
        super(self.__class__, self).__init__(four_vector)
        self.init_args = (p,)

    def __repr__(self):
        return "%s: %s" % (self.name, round(self.p, 2))


class Nydegger(Player):
    """
    The program begins with tit for tat for the first three moves, except that
    if it was the only one to cooperate on the first move and the only one to
    defect on the second move, it defects on the third move. After the third move,
    its choice is determined from the 3 preceding outcomes in the following manner.
    Let A be the sum formed by counting the other's defection as 2 points and one's
    own as 1 point, and giving weights of 16, 4, and 1 to the preceding three
    moves in chronological order. The choice can be described as defecting only
    when A equals 1, 6, 7, 17, 22, 23, 26, 29, 30, 31, 33, 38, 39, 45, 49, 54,
    55, 58, or 61. Thus if all three preceding moves are mutual defection,
    A = 63 and the rule cooperates. This rule was designed for use in laboratory
    experiments as a stooge which had a memory and appeared to be trustworthy,
    potentially cooperative, but not gullible.

    -- Axelrod, "Effective Choice in the Prisoner's Dilemma"

    """

    name = "Nydegger"
    classifier = {
        'memory_depth': 3,
        'stochastic': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def __init__(self):
        self.As = [1, 6, 7, 17, 22, 23, 26, 29, 30, 31, 33, 38, 39, 45, 54, 55,
                   58, 61]
        self.score_map = {(C, C): 0,
                          (C, D): 2,
                          (D, C): 1,
                          (D, D): 3}
        super(self.__class__, self).__init__()

    @staticmethod
    def score_history(my_history, opponent_history, score_map):
        """Implements the Nydegger formula A = 16 a_1 + 4 a_2 + a_3"""
        a = 0
        for i, weight in [(-1, 16), (-2, 4), (-3, 1)]:
            plays = (my_history[i], opponent_history[i])
            a += weight * score_map[plays]
        return a

    def strategy(self, opponent):
        if len(self.history) == 0:
            return C
        if len(self.history) == 1:
            # TFT
            return D if opponent.history[-1] == D else C
        if len(self.history) == 2:
            if opponent.history[0: 2] == [D, C]:
                return D
            else:
                # TFT
                return D if opponent.history[-1] == D else C
        A = self.score_history(self.history[-3:], opponent.history[-3:],
                               self.score_map)
        if A in self.As:
            return D
        return C


class Shubik(Player):
    """
    Plays like Tit-For-Tat with the following modification. After
    each retaliation, the number of rounds that Shubik retaliates
    increases by 1.
    """

    name = 'Shubik'
    classifier = {
        'memory_depth': float('inf'),
        'stochastic': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def __init__(self):
        Player.__init__(self)
        self.is_retaliating = False
        self.retaliation_length = 0
        self.retaliation_remaining = 0

    def _decrease_retaliation_counter(self):
        """Lower the remaining owed retaliation count and flip to non-retaliate
        if the count drops to zero."""
        if self.is_retaliating:
            self.retaliation_remaining -= 1
            if self.retaliation_remaining == 0:
                self.is_retaliating = False

    def strategy(self, opponent):
        if not opponent.history:
            return C
        if opponent.history[-1] == D:
            # Retaliate against defections
            if self.history[-1] == C: # it's on now!
                # Lengthen the retaliation period
                self.is_retaliating = True
                self.retaliation_length += 1
                self.retaliation_remaining = self.retaliation_length
                self._decrease_retaliation_counter()
                return D
            else:
                # Just retaliate
                if self.is_retaliating:
                    self._decrease_retaliation_counter()
                return D
        if self.is_retaliating:
            # Are we retaliating still?
            self._decrease_retaliation_counter()
            return D
        return C

    def reset(self):
        Player.reset(self)
        self.is_retaliating = False
        self.retaliation_length = 0
        self.retaliation_remaining = 0


class SteinRapoport(Player):
    """This rule plays tit for tat except that it cooperates on the first four
    moves, it defects on the last two moves, and every fifteen moves it checks
    to see if the opponent seems to be playing randomly. This check uses a
    chi-squared test of the other's transition probabilities and also checks
    for alternating moves of CD and DC.
    -- Axelrod, "Effective Choice in the Prisoner's Dilemma"
    Note: Makes use of tournament length
    Warning: This strategy is incomplete
    """

    name = "Stein-Rapoport"
    classifier = {
        'memory_depth': float('inf'), # Depending on how the Chi^2 is done
        'stochastic': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def __init__(self):
        Player.__init__(self)
        self.isDefecting = False

    def strategy(self, opponent):
        # Cooperate on the first four rounds
        if len(self.history) < 4:
            return C
        # Defect on the last two rounds
        if self.tournament_attributes["length"] - len(self.history) <= 2:
            return D
        if len(self.history) % 15 == 0:
            # Every 15 rounds do a Chi-squared for random
            expected = len(self.history) / 2.
            chi = (opponent.defections - expected) ** 2 / expected + \
                  (opponent.cooperations - expected) ** 2 / expected)
            if chi < 2: # Arbitrary choice, 60:40 gives chi = 2
                self.isDefecting = True
            ## Not implemented
        # Also check for alternating moves of C, D and D, C
        ## Not implemented
        if zip(self.history[-2:0], opponent.history[-2:0]) == [(C, D), (D, C)]:
            return D
        if self.isDefecting:
            return D
        # Tit For Tat
        return D if opponent.history[-1] == D else C

    def reset(self):
        Player.reset(self)
        self.isDefecting = False


class TidemanChieruzzi(Player):
    """
    Every run of defections played by the opponent increases the number of\
    defections that this strategy retaliates with by 1.
    The opponent is given a fresh start if:
        it is 10 points behind this strategy
        and it has not just started a run of defections
        and it has been at least 20 rounds since the last fresh start
        and there are more than 10 rounds remaining in the tournament
        and the total number of defections differs from a 50-50 random sample by
        at least 3.0 standard deviations.
    """

    name = "Tideman-Chieruzzi"
    classifier = {
        'memory_depth': float('inf'), # Depending on how the Chi^2 is done
        'stochastic': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def __init__(self):
        Player.__init__(self)
        self.defection_run_count = 0
        self.is_retaliating = False
        self.retaliation_remaining = 0
        self.last_fresh_start = 0
        self.score_diff = 0
        self.fresh_start = False

    def strategy(self, opponent):
        # Fresh start yields two C, this is the second
        if self.fresh_start:
            self.fresh_start = False
            return C

        round_number = len(self.history) + 1
        if round_number == 1:
            return C
        # Update score difference
        game = self.tournament_attributes["game"]
        last_round = (self.history[-1], opponent.history[-1])
        scores = game.score(last_round)
        self.score_diff += scores[0] - scores[1]

        if round_number > 1:
            # Check for start of a defection run
            if opponent.history[-2:] == [C, D]:
                self.defection_run_count += 1
                if not self.is_retaliating:
                    self.is_retaliating == True
                    self.retaliation_remaining = self.defection_run_count

        # Check Fresh Start Criteria
        if  (opponent.history[-2:] != [C, D]) and \
            (round_number - 20 > self.last_fresh_start) and \
            (abs(opponent.defections - (round_number - 1) / 2) <= 1.5 * sqrt(round_number - 1)) and \
            (self.tournament_attributes['length'] - round_number > 10) and \
            (self.score_diff >= 10):
            # Fresh Start!
            self.is_retaliating = False
            self.retaliation_remaining = 0
            self.score_diff = 0
            self.last_fresh_start = round_number
            self.fresh_start = True
            return C

        if self.is_retaliating:
            self.retaliation_remaining -= 1
            if self.retaliation_remaining == 0:
                self.is_retaliating = False
            return D
        # TFT
        return D if opponent.history[-1] == D else C

    def reset(self):
        Player.reset(self)
        self.defection_run_count = 0
        self.is_retaliating = False
        self.retaliation_remaining = 0
        self.last_fresh_start = 0
        self.fresh_start = False
        self.score_diff = 0


class Tullock(Player):
    """
    Cooperates for the first 11 rounds then randomly cooperates 10% less often
    than the opponent has in previous rounds."""

    name = "Tullock"
    classifier = {
        'memory_depth': 11, # long memory, modified by init
        'stochastic': True,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def __init__(self, rounds_to_cooperate=11):
        """
        Parameters
        ----------
        rounds_to_cooperate: int, 10
           The number of rounds to cooperate initially
        """
        Player.__init__(self)
        self._rounds_to_cooperate = rounds_to_cooperate
        self.__class__.memory_depth = rounds_to_cooperate
        self.init_args = (rounds_to_cooperate,)

    def strategy(self, opponent):
        rounds = self._rounds_to_cooperate
        if len(self.history) < rounds:
            return C
        cooperate_count = opponent.history[-rounds:].count(C)
        prop_cooperate = cooperate_count / float(rounds)
        prob_cooperate = max(0, prop_cooperate - 0.10)
        return random_choice(prob_cooperate)


class UnnamedStrategy(Player):
    """Apparently written by a grad student in political science whose name was withheld, this strategy cooperates with a given probability P. This probability (which has initial value .3) is updated every 10 rounds based on whether the opponent seems to be random, very cooperative or very uncooperative. Furthermore, if after round 130 the strategy is losing then P is also adjusted.

    Fourteenth Place with 282.2 points is a 77-line program by a graduate
    student of political science whose dissertation is in game theory. This rule has
    a probability of cooperating, P, which is initially 30% and is updated every 10
    moves. P is adjusted if the other player seems random, very cooperative, or
    very uncooperative. P is also adjusted after move 130 if the rule has a lower
    score than the other player. Unfortunately, the complex process of adjustment
    frequently left the probability of cooperation in the 30% to 70% range, and
    therefore the rule appeared random to many other players.
    -- Axelrod, "Effective Choice in the Prisoner's Dilemma"

    Warning: This strategy is incomplete.
    """

    name = "Unnamed Strategy"
    classifier = {
        'memory_depth': 0,
        'stochastic': True,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def strategy(self, opponent):
        r = random.uniform(3, 7) / float(10)
        return random_choice(r)
