"""Game to test abilities of task-oriented-dialogue modelling.
Implementation of a GameMaster to control game mechanisms.
"""

from typing import Dict, List, Tuple
from backends import Model
from clemgame.clemgame import DialogueGameMaster, GameBenchmark, GameScorer, Player
from clemgame import get_logger
from clemgame import file_utils
import json
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


    # TODO: Define custom response - "Find all slots needed"
    def _custom_response(self, messages, turn_idx) -> str:
        placeholder_response = "I am looking for a restaurant."
        utterance = f"{messages} TURN: {turn_idx}"
        return placeholder_response
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

    def _custom_response(self, messages, turn_idx) -> str:
        placeholder_response = "Here is my restaurant suggestion. {'address': '33 Bridge Street', 'area': 'centre', 'food': 'european', 'id': '6780', 'introduction': '', 'location': [52.20951, 0.11669], 'name': 'galleria', 'phone': '01223362054', 'postcode': 'cb21uw', 'pricerange': 'moderate', 'signature': 'poached fillets of monkfish in lemongrass with sweet red chilli cream sauce and tiger prawns with leeks and mushrooms served with rice', 'type': 'restaurant'}"
        utterance = f"{messages} TURN: {turn_idx}"
        return placeholder_response
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
        self.initial_prompt_b = game_instance["prompt_player_b"]

        self.questioner = Questioner(self.player_models[0], "A")
        self.answerer = Answerer(self.player_models[1], "B")

        self.add_player(self.questioner)
        self.add_player(self.answerer)

        # Make this the incomplete item with only the desired slots filled - delete all other ones
        self.goal = game_instance["goal"]
        self.current_response = None

        self.invalid_response = False
        self.invalid_json = False
        self.all_slots_filled = False

    # !! Overwrite for testing purposes

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
        if self.invalid_response:
            self.log_to_self("invalid format", "abort game")
            return False
        if self.invalid_json:
            self.log_to_self("invalid json format", "abort game")
            return False
        if self.current_turn >= self.max_turns:
            self.log_to_self("max turns reached", str(self.max_turns))
            return False
        if self.all_slots_filled:
            self.log_to_self("all slots successfully filled", "end game")
        return True

    # TODO: Implement + design validation / end of game!
    # Error messages!
    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        """_summary_

        Args:
            player (Player): _description_
            utterance (str): _description_

        Returns:
            bool: _description_
        """
        # not empty?
        # json structure given at end of utterance?
        if not utterance:
            self.invalid_response = True
            return False
        if player == self.answerer:
            if not utterance.find('{'):
                self.invalid_json = True
                return False
        self.log_to_self("valid format", "continue")
        return True

    # TODO: Implement + design validation
    # json object?
    def _on_parse_response(self, player: Player, utterance: str) -> Tuple[str, bool]:
        if player == self.answerer:
            self.log_to_self("suggestion", self.extract_json_from_response(utterance))
            self.current_response = self.extract_json_from_response(utterance)
            # print(f"CURRENT RESPONSE: {self.current_response}")
        return utterance, True

    def _after_add_player_response(self, player: Player, utterance: str):
        """Adds response to history of other player.

        Args:
            player (Player): _description_
            utterance (str): _description_
        """
        if player == Questioner:
            self.add_user_message(self.answerer, utterance)
        elif player == Answerer:
            self.add_user_message(self.questioner, utterance)

    # TODO: Check these general turn defs
    def _on_before_turn(self, turn_idx: int):
        if turn_idx == 0:
            self.log_message_to(self.questioner, self.initial_prompt_a)

    def _on_after_turn(self, turn_idx: int):
        """Checks if the json object of the current response contains all the keys from the goal object; if so, the all_slots_filled flag is activated.

        Args:
            turn_idx (int): Number of current turn.
        """
        if all(key in self.current_response for key in self.goal):
            self.all_slots_filled = True
        print(f"GOAL: {self.goal}")
        print(type(self.goal))
        print(f"CURRENT RESP: {self.current_response}")
        print(type(self.current_response))
        print(f"SLOTS FILLED: {self.all_slots_filled}")

    def extract_json_from_response(self, utterance):
        """Extracts json code from a string.

        Args:
            utterance (str): String which contains potential json code.

        Returns:
            _type_: _description_
        """
        try:
            # Find the start of the JSON structure in the response
            json_start = utterance.find('{')
            # Parse the JSON
            # FIXME: Check for double quotes!!
            json_data = json.loads(utterance[json_start:].replace("\'", "\""))
            # print(f"JSON DATA: {json_data}")
            return json_data
        except json.JSONDecodeError:
            print(utterance[json_start:])
            print("Invalid JSON structure detected. Please try again.")
            return None

    def compute_scores(self, episode_interactions: Dict):
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
