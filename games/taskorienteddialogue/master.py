"""Game to test abilities of task-oriented-dialogue modelling.
Implementation of a GameMaster to control game mechanisms.
"""

from typing import Dict, List
from backends import Model
from clemgame.clemgame import GameMaster, GameBenchmark, GameScorer


# Extend from DialogueGameMaster here? Moderator between 2 players. If so, several functions to be implemented: https://github.com/clp-research/clembench/blob/main/docs/howto_add_games.md
class Dialogue(GameMaster):
    """Play a single instance of a Dialogue Game.

    Args:
        GameMaster (_type_): _description_
        experiment
        player_models
    """
    def __init__(self):
        pass

    def setup(self, **kwargs):
        """Sets the background information of the game.

        Returns:
            _type_: _description_
        """
        return super().setup(**kwargs)
    
    def play(self):
        """Executes game logic and performs turns of the game.
        """
        pass

    def compute_score(self, episode_interactions: Dict):
        """Computes the game's scores.

        Args:
            episode_interactions (Dict): _description_
        """
        pass


class DialogueGameBenchmark(GameBenchmark):
    """_summary_

    Args:
        GameBenchmark (_type_): _description_
    """
    def __init__(self, name: str):
        super().__init__(name)

    def get_description(self) -> str:
        return "This is a TOD-Game."

    # experiment from instances.json, player_models == dialogue pair
    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return Dialogue(experiment, player_models)

    def is_single_player(self) -> bool:
        return False
