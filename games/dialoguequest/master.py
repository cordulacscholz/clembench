"""Game to test abilities of task-oriented-dialogue modelling.
Implementation of a GameMaster to control game mechanisms.
"""

from typing import Dict, List
from backends import Model
from clemgame.clemgame import DialogueGameMaster, GameBenchmark, GameScorer, Player
from clemgame import get_logger
import constants


logger = get_logger(__name__)
GAME_NAME = constants.GAME_NAME


class Questioner(Player):
    """_summary_

    Args:
        Player (_type_): _description_
    """
    def __init__(self, model_name: str, player: str) -> None:
        super().__init__(model_name)
        self.player: str = player

        # list for storing dialogue history
        self.history: List = []

    # TODO: Define custom response - "Find all slots needed"
    def _custom_response(self, messages, turn_idx) -> str:
        return "question"


class Answerer(Player):
    """_summary_

    Args:
        Player (_type_): _description_
    """
    def __init__(self, model_name: str, player: str) -> None:
        super().__init__(model_name)
        self.player: str = player

        self.history: List = []

    # TODO: Define custom response for Answerer - "I suggest..."
    def _custom_response(self, messages, turn_idx) -> str:
        return "answer"


# Extend from DialogueGameMaster here? Moderator between 2 players. If so, several functions to be implemented: https://github.com/clp-research/clembench/blob/main/docs/howto_add_games.md
class DialogueQuest(DialogueGameMaster):
    """Play a single instance of a Dialogue Game.

    Args:
        GameMaster (_type_): _description_
        experiment
        player_models
    """
    # TODO: Check params!
    def __init__(self, experiment: Dict, player_models: List[Model]):
        super().__init__(GAME_NAME, experiment, player_models)
        self.max_turns: int = experiment["max_turns"]
        self.questioner_initial_prompt = self.experiment["prompt_a_initial_prompt"]
        self.answerer_initial_prompt = self.experiment["prompt_b_initial_prompt"]

    # functions:
    # - `def _on_setup(self, **kwargs)` which must be implemented. Use `add_player()` here to add the players.

    def _on_setup(self, **game_instance):
        logger.info("_on_setup")
        self.game_instance = game_instance

        self.questioner = Questioner(self.player_models[0], self.max_turns)
        self.answerer = Answerer(self.player_models[1])

        self.add_player(self.questioner)
        self.add_player(self.answerer)

    def _on_before_game(self):
        self.add_user_message(self.questioner, self.questioner_initial_prompt)
        self.add_user_message(self.answerer, self.answerer_initial_prompt)

    # TODO: Design + Implementation! Refine
    def _does_game_proceed(self) -> bool:
        """Proceed as long as there are still unfilled slots and max number of turns has not been reached.

        Returns:
            bool: True if proceed, False if not proceed
        """
        proceed = True if not self.current_turn >= self.max_turns else False
        return proceed
    
    # - `def _validate_player_response(self, player: Player, utterance: str) -> bool` to decide if an utterance should be added. This is also the place to check for game end conditions.
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
