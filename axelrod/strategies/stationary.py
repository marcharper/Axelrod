
from collections import defaultdict
import functools

from scipy import stats, optimize
import numpy
from numpy import linalg

from axelrod.eigen import principal_eigenvector
from axelrod import Game, Player, random_choice


C, D = 'C', 'D'
#numpy.seterr(divide='ignore', invalid='ignore')


# Stationary distributions

def exact_stationary(p, q):
    """
    Using the Press and Dyson Formula where p and q are the conditional
    probability vectors:
    [P(C | CC), P(C | CD), P(C | DC), P(C | DD)]
    """

    s = []
    c1 = [-1 + p[0] * q[0], p[1] * q[2], p[2] * q[1], p[3] * q[3]]
    c2 = [-1 + p[0], -1 + p[1], p[2], p[3]]
    c3 = [-1 + q[0], q[2], -1 + q[1], q[3]]

    # Compute determinants
    for i in range(4):
        f = numpy.zeros(4)
        f[i] = 1
        m = numpy.matrix([c1,c2,c3,f])
        d = linalg.det(m)
        s.append(d)

    # Normalize
    n = sum(s)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        s = numpy.array(s) / n
    return s

def approximate_stationary(transitions, power=8, maximum_iterations=10):
    """
    Computes the stationary distribution using the principle eigenvector
    functionality of axelrod. The parameter `transitions` is the transition
    matrix of the Markov process for the four states CC, CD, DC, DD.
    """

    transitions_ = numpy.linalg.matrix_power(transitions, power)
    stationary, _ = principal_eigenvector(transitions,
                                          maximum_iterations=maximum_iterations)
    # Normalize to probability distribution
    stationary = stationary / sum(stationary)
    return stationary

def approximate_stationary2(transitions, epsilon=1e-4):
    """Computes the stationary distribution using eigenvector method."""
    w, v = linalg.eig(t.transpose())
    for i, eigenval in enumerate(w):
        s = numpy.real_if_close(v[:,i])
        s /= s.sum()
        if abs(eigenval - 1.) <= epsilon and (s >= 0.).sum() == len(s):
            return s
    raise ValueError('No stationary eigenvalue!')

def compute_transitions(v1, v2):
    """
    Computes the transition matrix of the Markov process for the four states
    CC, CD, DC, DD from the two strategy four-vectors.
    """

    mat = []
    for p, q in zip(v1, v2):
        row = (p * q, p * (1. - q), (1. - p) * q, (1. - p) * (1. - q))
        mat.append(row)
    return numpy.array(mat)

def compute_stationary(four_vector1, four_vector2):
    """
    Computes the stationary distribution of the Markov process for the four
    states CC, CD, DC, DD from the two strategy four-vectors. First attempts
    an exact calculation and falls back on an iterative approximation.
    """

    try:
        stationary = exact_stationary(four_vector1, four_vector2)
    except ZeroDivisionError:
        transitions = compute_transitions(four_vector1, four_vector2)
        stationary = approximate_stationary(transitions)
    return stationary

# Response strategy

def stationary_payoff(four_vector, opponent_four_vector, payoffs, s=None):
    """
    Computes the expected stationary payoff for two memory one players
    specified by their four-vectors
    """

    if s is None:
        s = compute_stationary(four_vector, opponent_four_vector)
    return numpy.dot(s, payoffs)

def stationary_difference(four_vector, opponent_four_vector, payoffs):
    s1 = compute_stationary(four_vector, opponent_four_vector)
    p1 = numpy.dot(s1, payoffs)
    s2 = [s1[0], s1[2], s1[1], s1[3]]
    p2 = numpy.dot(s2, payoffs)
    return p1 - p2

def perturb(p, ep=0.001):
    return (1 - ep) * p + (numpy.ones(4) - p) * ep

def compute_response_four_vector(opponent_four_vector, mode='t', maxfun=100):
    """
    Computes an optimal response strategy.
    """

    game = Game()
    (R, P, S, T) = game.RPST()
    payoffs = numpy.array([R, S, T, P])

    opponent_four_vector = perturb(numpy.array(opponent_four_vector))

    if mode == 't': # Maximize score
        to_optimize = functools.partial(stationary_payoff, payoffs=payoffs,
                                        opponent_four_vector=opponent_four_vector)
    if mode == 'd': # Maximize score difference
        to_optimize = functools.partial(stationary_difference, payoffs=payoffs,
                                        opponent_four_vector=opponent_four_vector)

    p, _, _ = optimize.fmin_tnc(lambda p: -1 * to_optimize(p), [0.8] * 4,
                                bounds=[(0, 1)] * 4, approx_grad=True,
                                messages=0, maxfun=maxfun, epsilon=0.01,
                                accuracy=1e-6, ftol=1e-4)
    return p


class StationaryMax(Player):
    """
    After an initial phase of gathering information, this strategy treats the
    opponent like a memory-one player and computes an optimal
    counter-strategy.
    """

    name = "Stationary Max"
    memory_depth = float('inf')
    stochastic = True

    def __init__(self, initial_four_vector=None, initial=C,
                 initial_phase_length=10, mode='t'):
        Player.__init__(self)
        self._initial = initial
        if not initial_four_vector:
            initial_four_vector = (1, 0.1, 1, 0.1)
        self.set_four_vector(initial_four_vector)
        self.initial_phase_length = initial_phase_length
        self.play_counts = defaultdict(int)
        self.play_cooperations = defaultdict(int)
        self._response_four_vector = None
        self.mode = mode
        self.stochastic = True

    def set_four_vector(self, vector, response=False):
        keys = [(C, C), (C, D), (D, C), (D, D)]
        values = map(float, vector)
        if response:
            self._response_four_vector = dict(zip(keys, values))
        else:
            self._four_vector = dict(zip(keys, values))

    def opponent_four_vector(self, const=0.5):
        # Note (C, D) and (D, C) order swapped for opponent's plays
        four_vector = []
        for key in [(C, C), (D, C), (C, D), (D, D)]:
            coop = (self.play_cooperations[key] + const) / (self.play_counts[key] + const)
            four_vector.append(coop)
        return four_vector

    def strategy(self, opponent):
        round_number = len(self.history)
        # Update play play counts
        if len(self.history) > 1:
            last_context = (self.history[-2], opponent.history[-2])
            last_round = (self.history[-1], opponent.history[-1])
            self.play_counts[last_context] += 1
            if last_round[1] == 'C':
                self.play_cooperations[last_context] += 1
        if not round_number:
            return self._initial
        if round_number >= max(self.tournament_length // 20, 15):
            # Compute the response strategy
            opponent_four_vector = self.opponent_four_vector()
            mod = self.tournament_length // 20
            # This is only to reduce the CPU footprint, in a competitive tournament
            # the player should be allowed to update every round
            if round_number % mod == 0 or (self._response_four_vector is None):
                response_vector = compute_response_four_vector(opponent_four_vector,
                                                            mode=self.mode)
                self.set_four_vector(response_vector, response=True)
            p = self._response_four_vector[(self.history[-1], opponent.history[-1])]
            return random_choice(p)
        else:
            p = self._four_vector[(self.history[-1], opponent.history[-1])]
            return random_choice(p)

    def reset(self):
        Player.reset(self)
        self.play_counts = defaultdict(int)
        self._response_four_vector = None


class StationaryMaxDiff(StationaryMax):

    name = "Stationary Max Diff"
    memory_depth = float('inf')
    stochastic = True

    def __init__(self):
        StationaryMax.__init__(self, mode='d')
