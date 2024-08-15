"""Game to test abilities of task-oriented-dialogue modelling.
Implementation of a GameMaster to control game mechanisms.
"""

from typing import Dict, List, Tuple
from backends import Model
from clemgame.clemgame import DialogueGameMaster, GameBenchmark, GameScorer, Player
from clemgame import get_logger
from clemgame import file_utils
from games.dialoguequest.constants import (
    GAME_NAME, MAX_TURNS)


logger = get_logger(__name__)


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
        self.goal = {}

    # TODO: Define custom response - "Find all slots needed"
    def _custom_response(self, messages, turn_idx) -> str:
        utterance = f"{messages} TURN: {turn_idx}"
        print(utterance)
        return utterance


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
        utterance = f"{messages} TURN: {turn_idx}"
        print(utterance)
        return utterance


class DialogueQuest(DialogueGameMaster):
    """Play a single instance of a Dialogue Game.

    Args:
        GameMaster (_type_): _description_
        experiment
        player_models
    """
    def __init__(self, experiment: Dict, player_models: List[Model]):
        super().__init__(GAME_NAME, experiment, player_models)
        self.max_turns: int = MAX_TURNS

    def _on_setup(self, **game_instance):
        logger.info("_on_setup")

        self.game_instance = game_instance

        self.initial_prompt_a = game_instance["prompt_player_a"]
        print(self.initial_prompt_a)
        self.initial_prompt_b = game_instance["prompt_player_b"]
        print(self.initial_prompt_b)

        self.questioner = Questioner(self.player_models[0], "A")
        self.answerer = Answerer(self.player_models[1], "B")

        self.add_player(self.questioner)
        self.add_player(self.answerer)

        self.goal = game_instance["goal"]

    def _on_before_game(self):
        self.add_user_message(self.questioner, self.initial_prompt_a)
        self.add_user_message(self.answerer, self.initial_prompt_b)

    # TODO: Add working mechanism to check slots / internal object!
    # TODO: Add specialised error messages
    def _does_game_proceed(self) -> bool:
        """Proceed as long as there are still unfilled slots and max number of turns has not been reached.

        Returns:
            bool: True if proceed, False if not proceed
        """
        proceed = True if not self.current_turn >= self.max_turns else False
        return proceed

    # TODO: Implement + design validation / end of game!
    # not valid: empty
    # end of game: all slots filled
    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        # not empty?
        # json structure given at end of utterance?
        return True

    # TODO: Implement + design validation
    # json object?
    def _on_parse_response(self, player: Player, utterance: str) -> Tuple[str, bool]:
        return utterance, True

    # TODO: Design + modify logging events!
    def _after_add_player_response(self, player: Player, utterance: str):
        if player == Questioner:
            self.add_user_message(self.questioner, utterance)
        elif player == Answerer:
            self.add_user_message(self.answerer, utterance)

    # - the general game hooks `_on_before_game()` and `_on_before_game()`

    # TODO: Check these general turn defs
    def _on_before_turn(self, turn_idx: int):
        return super()._on_before_turn(turn_idx)

    def _on_after_turn(self, turn_idx: int):
        # return super()._on_after_turn(turn_idx)
        print(self.goal)
        #TODO: check internal_object

    # - the general turn hooks `_on_before_turn(turn_idx)` and `_on_after_turn(turn_idx)`

    def compute_score(self, episode_interactions: Dict):
        """Computes the game's scores.

        Args:
            episode_interactions (Dict): _description_
        """
        pass


class DialogueQuestScorer(GameScorer):
    def __init__(self, experiment: Dict, game_instance: Dict):
        super().__init__(GAME_NAME, experiment, game_instance)


class DialogueQuestBenchmark(GameBenchmark):
    """_summary_

    Args:
        GameBenchmark (_type_): _description_
    """
    def __init__(self):
        super().__init__(GAME_NAME)
        self.name = GAME_NAME

    def get_description(self) -> str:
        return "This is a Task Oriented Dialogue Game."

    # experiment from instances.json, player_models == dialogue pair
    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> DialogueGameMaster:
        return DialogueQuest(experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return DialogueQuestScorer(experiment, game_instance)

    def is_single_player(self) -> bool:
        return False


def main():
    # select one instance
    experiments = file_utils.load_json("in/instances.json", "dialoguequest")
    instance = experiments["experiments"][0]["game_instances"][0]
    master = DialogueQuest(instance, ["gpt-3.5-turbo", "gpt-3.5-turbo"])

    master.setup(**instance)
    master.play()


if __name__ == "__main__":
    main()
