from collections import defaultdict
from axelrod import Actions, Player

# import theano
# theano.config.device = "cpu"
# theano.config.force_device = True

from keras.models import model_from_json

import numpy as np

C, D = Actions.C, Actions.D

"""
Todo:
* decision function based on output
* run tournaments to collect more data and re-train
network
* Use game matrix in calculation
* RNN
"""


mapping = {'C': 1, 'D': 0}

def map_history(h):
    return [mapping[x] for x in h]

def load_model(name="model"):
    json_file = open("/home/marc/repos/axelrod/Axelrod/model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("/home/marc/repos/axelrod/Axelrod/model.h5")
    return loaded_model

def compute_features(player, opponent):
    N = len(player.history)
    features = [N,
                player.cooperations, N - player.cooperations,
                opponent.cooperations, N - opponent.cooperations,
                player.play_counts[('C', 'C')],
                player.play_counts[('C', 'D')],
                player.play_counts[('D', 'C')],
                player.play_counts[('D', 'D')],
                ]
    if len(player.history) > 1:
        features.extend(map_history(player.history[0:2]))
        features.extend(map_history(opponent.history[0:2]))
        features.extend(map_history(player.history[-2:]))
        features.extend(map_history(opponent.history[-2:]))
    else:
        features.extend(map_history(player.history) + [0])
        features.extend(map_history(opponent.history) + [0])
        features.extend([0] + map_history(player.history))
        features.extend([0] + map_history(opponent.history[-2:]))
    return features


class KNN(Player):
    """A player who alternates between cooperating and defecting."""

    name = '------------------------'
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
        if len(self.history) > 0:
            last_round = (self.history[-1], opponent.history[-1])
            self.play_counts[last_round] += 1
        else:
            return C
        # # TFT for first 2 rounds
        # if len(self.history) == 0:
        #     return C
        # if len(self.history) == 1:
        #     if opponent.history[-1] == D:
        #         return D
        #     return C
        # if len(self.history) < 4:
        #     if len(self.history) == 0:
        #         return C
        #     # React to the opponent's last move
        #     if opponent.history[-1] == D:
        #         return D
        #     return C
        features = compute_features(self, opponent)
        X = np.array([features])
        coop_prob = self.model.predict_proba(X, verbose=False)[0]

        # if coop_prob > 0.90:
        #     # Can we get away with a defection?
        #     # Update features for a round of D, C
        #     features[0] += 1
        #     features[2] += 1
        #     features[3] += 1
        #     features[9:11] = map_history([self.history[-1], D])
        #     features[11:13] = map_history([opponent.history[-1], C])
        #     features[-2] += 1  # (D, C)
        #     X = np.array([features])
        #     retaliation_prob = self.model.predict_proba(X, verbose=False)
        #     if retaliation_prob < 0.10:
        #         return D
        #     # If not then retaliate
        #     else:
        #         return C
        # else:
        #     # If cooperation isn't likely and we can't defect without
        #     # retaliation, then defect
        #     return D

        if coop_prob > 0.:
            return C
        else:
            return D

        def reset(self):
            Player.reset(self)
            self.play_counts = defaultdict(int)
