"""Game to test abilities of task-oriented-dialogue modelling.
Implementation of a GameMaster to control game mechanisms.
"""

from typing import Dict, List, Tuple
from backends import Model
from clemgame.clemgame import DialogueGameMaster, GameBenchmark, GameScorer, Player
from clemgame import get_logger
# import constants
# from dialoguequest import constants


logger = get_logger(__name__)
GAME_NAME = "dialoguequest"
# GAME_NAME = constants.GAME_NAME


# def print_constants(value):
#     print(value)

# print_constants(GAME_NAME)


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
    # ??? key errors - not in instances file ???
    def __init__(self, experiment: Dict, player_models: List[Model]):
        super().__init__(GAME_NAME, experiment, player_models)
        # self.max_turns: int = experiment["max_turns"]
        self.max_turns: int = 10
        print(f"EXPERIMENT: {experiment}")
        # self.questioner_initial_prompt = self.experiment["prompt_player_a"]
        # self.answerer_initial_prompt = self.experiment["prompt_player_b"]

    # functions:
    # - `def _on_setup(self, **kwargs)` which must be implemented. Use `add_player()` here to add the players.

    def _on_setup(self, **game_instance):
        logger.info("_on_setup")
        self.game_instance = game_instance

        self.questioner = Questioner(self.player_models[0], "A")
        self.answerer = Answerer(self.player_models[1], "B")

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

    # TODO: Implement + design validation / end of game!
    # not valid: empty
    # end of game: all slots filled
    def _validate_player_response(self, player: Player, utterance: str) -> bool:
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
        return super()._on_after_turn(turn_idx)

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
    def __init__(self):
        super().__init__(GAME_NAME)
        self.name = GAME_NAME

    def get_description(self) -> str:
        return "This is a Task Oriented Dialogue Game."

    # experiment from instances.json, player_models == dialogue pair
    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> DialogueGameMaster:
        print(f"Hello! {self.name}")
        return DialogueQuest(experiment, player_models)

    def is_single_player(self) -> bool:
        return False

# def main():
#     # select one instance
#     experiments = file_utils.load_json("in/instances.json", "referencegame")
#     instance = experiments["experiments"][0]["game_instances"][0]
#     master = ReferenceGameMaster(instance, ["gpt-3.5-turbo", "gpt-3.5-turbo"])
#     master.setup(**instance)
#     master.play()


# if __name__ == '__main__':
#     main()