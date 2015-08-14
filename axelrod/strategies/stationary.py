
from collections import defaultdict
import functools

from scipy import stats, optimize
import numpy
from numpy import linalg

from axelrod.eigen import principal_eigenvector
from axelrod import Game, Player, random_choice

C, D = 'C', 'D'

## Todo: # set initial phase based on tournament length


# Stationary distributions

#def game_transition_matrix(myProbs, hisProbs0):
    #'compute transition rate matrix for a strategy pair'
    # have to swap moves for other player...
    #hisProbs = (hisProbs0[0], hisProbs0[2],hisProbs0[1], hisProbs0[3])
    #l = []
    #for i,myP in enumerate(myProbs):
        #hisP = hisProbs[i]
        #l.append((myP * hisP, myP * (1. - hisP), 
                  #(1. - myP) * hisP, (1. - myP) * (1. - hisP)))
    #return numpy.array(l)

#def stationary_rates(myProbs, hisProbs):
    #'compute expectation rates of all possible transitions for strategy pair'
    #t = game_transition_matrix(myProbs, hisProbs)
    #s = stationary_dist2(t)
    #return [p * t[i] for (i,p) in enumerate(s)]

#def stationary_score(myProbs, hisProbs, scores):
    #'compute expectation score for my strategy vs. opponent strategy'
    #rates = stationary_rates(myProbs, hisProbs)
    #l = [scores * vec for vec in rates]
    #return numpy.array(l).sum()


#def stationary_dist(t, epsilon=1e-10):
    #'compute stationary dist from transition matrix'
    #diff = 1.
    #while diff > epsilon:
        #t = linalg.matrix_power(t, 2)
        #w = t.sum(axis=1) # get row sums
        #t /= w.reshape((len(w), 1)) # normalize each row
        #m = numpy.mean(t, axis=0)
        #diff = numpy.dot(m, t) - m
        #diff = (diff * diff).sum()
    #return m

#def stationary_dist2(t, epsilon=.001):
    #'compute stationary distribution using eigenvector method'
    #w, v = linalg.eig(t.transpose())
    #for i,eigenval in enumerate(w):
        #s = numpy.real_if_close(v[:,i]) # handle small complex number errors
        #s /= s.sum() # normalize
        #if abs(eigenval - 1.) <= epsilon and (s >= 0.).sum() == len(s):
            #return s # must have unit eigenvalue and all non-neg components
    #raise ValueError('no stationary eigenvalue??')

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
    s = numpy.array(s) / n
    return s

def approximate_stationary(transitions):
    """
    Computes the stationary distribution using the principle eigenvector
    functionality of axelrod. The parameter `transitions` is the transition
    matrix of the Markov process for the four states CC, CD, DC, DD.
    """

    stationary, _ = principal_eigenvector(transitions, maximum_iterations=1000,
                                       max_error=1e-12)
    # Normalize to probability distribution
    stationary = stationary / sum(stationary)
    return stationary

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

def stationary_payoff(four_vector, opponent_four_vector, payoffs):
    """
    Computes the expected stationary payoff for two memory one players
    specified by their four-vectors
    """

    s = compute_stationary(four_vector, opponent_four_vector)
    return numpy.dot(s, payoffs)

def stationary_difference(four_vector, opponent_four_vector, payoffs):
    p1 = stationary_payoff(four_vector, opponent_four_vector, payoffs)
    p2 = stationary_payoff(opponent_four_vector, four_vector, payoffs)
    return p1 - p2

def compute_response_four_vector(opponent_four_vector, mode='t'):
    """
    Computes an optimal response strategy.
    """

    game = Game()
    (R, P, S, T) = game.RPST()
    payoffs = numpy.array([R, S, T, P])

    if mode == 't': # Maximize score
        to_optimize = functools.partial(stationary_payoff, payoffs=payoffs,
                                        opponent_four_vector=opponent_four_vector)
    if mode == 'd': # Maximize score difference
        to_optimize = functools.partial(stationary_difference, payoffs=payoffs,
                                        opponent_four_vector=opponent_four_vector)

    p, _, _ = optimize.fmin_tnc(lambda p: -1 * to_optimize(p), [0.5] * 4,
                                bounds=[(0, 1)] * 4, approx_grad=True,
                                messages=0, maxfun=1000)
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

    def __init__(self, initial_four_vector=(1, 0, 1, 0), initial=C,
                 initial_phase_length=10, mode='t'):
        Player.__init__(self)
        self._initial = initial
        self.set_four_vector(initial_four_vector)
        self.initial_phase_length = initial_phase_length
        self.play_counts = defaultdict(int)
        self._response_four_vector = None
        self.mode = mode

    def set_four_vector(self, vector, response=False):
        keys = [(C, C), (C, D), (D, C), (D, D)]
        values = map(float, vector)
        if response:
            self._response_four_vector = dict(zip(keys, values))
        else:
            self._four_vector = dict(zip(keys, values))

    def opponent_four_vector(self, const=0.):
        total_plays = float(len(self.history)) + const
        # Note (C, D) and (D, C) ordered swapped for opponent's plays
        four_vector = []
        for key in [(C, C), (D, C), (C, D), (D, D)]:
            four_vector.append((self.play_counts[key] + const) / total_plays)
        return four_vector

    def strategy(self, opponent):
        round_number = len(self.history)
        # Update play play counts
        if round_number:
            last_round = (self.history[-1], opponent.history[-1])
            self.play_counts[last_round] += 1
        if not round_number:
            return self._initial
        if round_number > self.initial_phase_length:
            # Compute the response strategy
            opponent_four_vector = self.opponent_four_vector()
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
