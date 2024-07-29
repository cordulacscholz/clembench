"""Game to test abilities of task-oriented-dialogue modelling.
Implementation of a GameMaster to control game mechanisms.
"""

from typing import Dict, List
from backends import Model
from clemgame.clemgame import DialogueGameMaster, GameBenchmark, GameScorer


# Extend from DialogueGameMaster here? Moderator between 2 players. If so, several functions to be implemented: https://github.com/clp-research/clembench/blob/main/docs/howto_add_games.md
class DialogueQuest(DialogueGameMaster):
    """Play a single instance of a Dialogue Game.

    Args:
        GameMaster (_type_): _description_
        experiment
        player_models
    """
    def __init__(self):
        pass

    # functions:
    # _on_setup
    # _does_game_proceed
    # def _validate_player_response(self, player: Player, utterance: str) -> bool
    #     - `def _on_parse_response(self, player: Player, utterance: str) -> Tuple[str, bool]` to decide if a response utterance should be modified. If not simply return the utterance.
    #         When a modified utterance and a true value is returned, then a 'parse' event is logged.
    # - `def _after_add_player_response(self, player: Player, utterance: str)` to add the utterance to other player's history, if necessary.
    #         To do this use the method `add_user_message(other_player,utterance)`.
    # - the general game hooks `_on_before_game()` and `_on_before_game()`
    # - the general turn hooks `_on_before_turn(turn_idx)` and `_on_after_turn(turn_idx)`


    def compute_score(self, episode_interactions: Dict):
        """Computes the game's scores.

        Args:
            episode_interactions (Dict): _description_
        """
        pass


class DialogueQuestBenchmark(GameBenchmark):
    """_summary_

    Args:
        GameBenchmark (_type_): _description_
    """
    def __init__(self, name: str):
        super().__init__(name)

    def get_description(self) -> str:
        return "This is a TOD-Game."

    # experiment from instances.json, player_models == dialogue pair
    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> DialogueGameMaster:
        return DialogueQuest(experiment, player_models)

    def is_single_player(self) -> bool:
        return False
