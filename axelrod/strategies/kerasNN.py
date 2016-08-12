from collections import defaultdict
from axelrod import Actions, Player

from keras.models import model_from_json

import numpy as np

C, D = Actions.C, Actions.D

"""
Todo:
* function to map histories to NN input
* decision function based on output
* run tournaments to collect more data and re-train
network
* Use game matrix in calculation
* RNN
* predict effect of defection
"""


mapping = {'C': 0, 'D': 1}

def map_history(h):
    return [mapping[x] for x in h]

def load_model(name="model"):
    json_file = open("/home/user/" + name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("/home/user/" + name + ".h5")
    return loaded_model

def compute_features(player, opponent):
    features = [len(player.history),
                player.cooperations, opponent.cooperations,
                player.play_counts[('C', 'C')],
                player.play_counts[('C', 'D')],
                player.play_counts[('D', 'C')],
                player.play_counts[('D', 'D')],
                ]
    features.extend(map_history(player.history[0:2]))
    features.extend(map_history(opponent.history[0:2]))
    features.extend(map_history(player.history[-2:]))
    features.extend(map_history(opponent.history[-2:]))
    return features

# def vectorize_interactions(h1, h2):
#     coops = np.cumsum(h1)
#     op_coops = np.cumsum(h2)
#     ccs = cumulative_context_counts(h1, h2)
#     for i in range(4, len(h1)):
#         row = [i, coops[i], op_coops[i],
#                h1[0], h1[1], h2[0], h2[1],
#                h1[i-2], h1[i-1], h2[i-2], h2[i-1]]
# #         cc = context_counts(h1, h2, i)
#         row.extend(ccs[i])
#         y = h2[i]
#         row.append(y)
#         yield row
#
# def zeros_and_ones(h):
#     return list(map(lambda x: mapping[x], h))
#
# def yield_data(filename):
#     with open(filename) as handle:
#         for line in handle:
#             s = line.strip().split(',')
#             h1, h2 = s[-2], s[-1]
#             h1 = zeros_and_ones(h1)
#             h2 = zeros_and_ones(h2)
#             yield from vectorize_interactions(h1, h2)

class KNN(Player):
    """A player who alternates between cooperating and defecting."""

    name = 'Keras NN'
    classifier = {
        'memory_depth': float('inf'),
        'stochastic': False,
        'makes_use_of': set(),
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def __init__(self):
        Player.__init__(self)
        self.model = load_model()
        self.play_counts = defaultdict(int)
        self.init_args = ( )

    def strategy(self, opponent):
        # Record context counts
        if len(self.history):
            last_round = (self.history[-1], opponent.history[-1])
            self.play_counts[last_round] += 1
        # TFT for first 4 rounds
        if len(self.history) == 0:
            return C
        if len(self.history) < 4:
            # React to the opponent's last move
            if opponent.history[-1] == D:
                return D
            return C
        features = compute_features(self, opponent)
        X = np.array([features])
        coop_prob = self.model.predict(X)
        if coop_prob < 0.6:
            return D
        return C

        def reset(self):
            Player.reset(self)
            self.play_counts = defaultdict(int)
