from axelrod import Actions, Player

from .hunter import DefectorHunter, AlternatorHunter, RandomHunter, MathConstantHunter, CycleHunter, EventualCycleHunter
from .lookerup import EvolvedLookerUp
from .meta import MetaHunter
from .oncebitten import FoolMeOnce
from .titfortat import OmegaTFT


from axelrod.strategy_transformers import StrategyTransformerFactory, DeadlockBreakingTransformer, GrudgeTransformer

#team = [CycleHunter, RandomHunter, MathConstantHunter, EventualCycleHunter]
team = [RandomHunter, MathConstantHunter, EventualCycleHunter]


class HunterWrapper(object):
    """Enforces the TFT rule that the opponent pay back a defection with a
    cooperation for the player to stop defecting."""
    def __init__(self, team=None):
        self.meta_hunter = MetaHunter(team=team)

    def __call__(self, player, opponent, action):
        # Play the meta hunter
        self.meta_hunter.history = player.history
        hunter_action = self.meta_hunter.strategy(opponent)
        if hunter_action == 'D':
            return 'D'
        return action

HunterTransformer = StrategyTransformerFactory(
    HunterWrapper(team), name_prefix="Hunting")()

HuntingOmegaTFT = HunterTransformer(OmegaTFT)
HuntingOmegaTFT.classifier["memory_depth"] = float('inf')

HuntingFMO = HunterTransformer(FoolMeOnce)
HuntingFMO.classifier["memory_depth"] = float('inf')

HuntingELU = HunterTransformer(EvolvedLookerUp)
HuntingELU.classifier["memory_depth"] = float('inf')

DeadlockELU = DeadlockBreakingTransformer(EvolvedLookerUp)

GrudgeDeadlockELU = GrudgeTransformer(5)(DeadlockBreakingTransformer(EvolvedLookerUp))