"""Game to test abilities of task-oriented-dialogue modelling.
Implementation of a GameMaster to control game mechanisms.
"""

from typing import Dict, List, Tuple
from backends import Model
from clemgame.clemgame import DialogueGameMaster, GameBenchmark, GameScorer, Player
from clemgame import get_logger
from clemgame import file_utils
from clemgame import metrics as ms
import json
import numpy as np
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

    def _custom_response(self, messages, turn_idx) -> str:
        utterance = f"{messages} TURN: {turn_idx}"
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
        """_summary_

        Args:
            messages (str): _description_
            turn_idx (int): _description_

        Returns:
            str: message including number of turn
        """
        utterance = f"{messages} TURN: {turn_idx}"
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
        self.current_suggestion = None

        # flags for keeping track of the game status
        self.invalid_response = False
        self.invalid_json = False
        self.booking = False

    def _on_before_game(self):
        self.add_user_message(self.questioner, self.initial_prompt_a)
        self.add_user_message(self.answerer, self.initial_prompt_b)

    # TODO: How to proceed with incomplete json?
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
        # TODO: shorten key
        if self.booking:
            self.log_to_self("all slots successfully filled", "end game")
            return False
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
        # if player == self.answerer:
        #     if not utterance.find('{'):
        #         self.invalid_json = True
        #         return False
        self.log_to_self("valid format", "continue")
        return True

    # TODO: Implement + design validation
    # json object?
    def _on_parse_response(self, player: Player, utterance: str) -> Tuple[str, bool]:
        # if player == self.questioner:
        #     print(f"Player A: {utterance}")
        if player == self.answerer:
            # self.log_to_self("suggestion", self.extract_json_from_response(utterance))
            self.log_to_self("utterance", utterance)
            self.current_suggestion = self.extract_json_from_response(utterance)
            self.current_response = utterance
            # print(f"CURRENT RESPONSE: {self.current_response}")
            if "BOOKED" in utterance:
                self.booking = True
        return utterance, True

    def _after_add_player_response(self, player: Player, utterance: str):
        """Adds response to history of other player.

        Args:
            player (Player): _description_
            utterance (str): _description_
        """
        if player == self.questioner:
            # print(f"HISTORY QUESTIONER: {self.questioner.history}")
            self.add_user_message(self.answerer, utterance)
            # print(f"HISTORY QUESTIONER AFTER ADDITION: {self.questioner.history}")
        elif player == self.answerer:
            # print(f"HISTORY ANSWERER: {self.answerer.history}")
            self.add_user_message(self.questioner, utterance)
            # print(f"HISTORY ANSWERER AFTER ADDITION: {self.answerer.history}")

    def _on_after_turn(self, turn_idx: int):
        """Checks if the json object of the current response contains all the keys from the goal object; if so, the booking flag is activated.

        Args:
            turn_idx (int): Number of current turn.
        """
        # if all(key in self.current_response for key in self.goal):
        #     self.booking = True
        # print(f"GOAL: {self.goal}")
        # print(f"CURRENT RESP: {self.current_response}")
        # print(f"ALL SLOTS SUGGESTED: {self.booking}")

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
            # TODO: Maybe add a var for closing bracket?
            # Parse the JSON
            # FIXME: Check for double quotes!!
            json_data = json.loads(utterance[json_start:].replace("\'", "\""))
            # print(f"JSON DATA: {json_data}")
            return json_data
        except json.JSONDecodeError:
            print(utterance[json_start:])
            print("Invalid JSON structure detected. Please try again.")
            return utterance
            # return None

    # TODO: Decide on metrics to log!
    def _log_eval_assets(self) -> None:
        """Log everything needed for the evaluation."""
        pass
        # self.log_key(ms.METRIC_REQUEST_COUNT,
        #              self.request_counts)
        # self.log_key(ms.METRIC_REQUEST_COUNT_PARSED,
        #              self.parsed_request_counts)
        # self.log_key(ms.METRIC_REQUEST_COUNT_VIOLATED,
        #              self.violated_request_counts)
        # self.log_key('Filled Slots', self.filled_slots)
        # self.log_key('Aborted', self.aborted)
        # self.log_key('Played Probe Rounds', self.played_probing_rounds)


class DialogueQuestScorer(GameScorer):
    def __init__(self, experiment: Dict, game_instance: Dict):
        super().__init__(GAME_NAME, experiment, game_instance)

        # copied from Vorlage - MODIFY !!
    # Only general scores logged, add specific ones!
    # Check if in line with interactions.json
    def compute_scores(self, episode_interactions: Dict):
        """Compute episode-level and turn-level scores.

        Args:
            episode_interactions (Dict): _description_
        """

        # Episode level scores

        # Initialise counters for episode scores
        turn_scores = []
        invalid_response = False

        for turn_idx, turn in enumerate(episode_interactions["turns"]):
            turn_score = {"request_count": 1}

            for event in turn:
                action = event["action"]
                if action["type"] == "invalid format":
                    invalid_response = True

            if invalid_response:
                turn_score["violated_request_count"] = 1
                turn_score["parsed_request_count"] = 0
            else:
                turn_score["violated_request_count"] = 0
                turn_score["parsed_request_count"] = 1

            self.log_turn_score(turn_idx, ms.METRIC_REQUEST_COUNT_VIOLATED, turn_score["violated_request_count"])
            self.log_turn_score(turn_idx, ms.METRIC_REQUEST_COUNT_PARSED, turn_score["parsed_request_count"])
            self.log_turn_score(turn_idx, ms.METRIC_REQUEST_COUNT, turn_score["request_count"])

        violated_request_count = sum([turn["violated_request_count"] for turn in turn_scores])
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_VIOLATED, violated_request_count)

        parsed_request_count = sum([turn["parsed_request_count"] for turn in turn_scores])
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_PARSED, parsed_request_count)

        request_count = sum([turn["request_count"] for turn in turn_scores])
        self.log_episode_score(ms.METRIC_REQUEST_COUNT, request_count)

        self.log_episode_score(ms.METRIC_REQUEST_SUCCESS, parsed_request_count / request_count)
        # checking the last guess (could be None) is ok,
        # b.c. the game ends only successfully, when there is a correct guess

        # played_turns = episode_interactions['Played turns']
        # complete_turns = episode_interactions['Complete turns']
        # turn 0 was only the initial prompts, so we disregard it here

        # reqs = episode_interactions[ms.METRIC_REQUEST_COUNT][1:]
        # p_reqs = episode_interactions[ms.METRIC_REQUEST_COUNT_PARSED][1:]
        # v_reqs = episode_interactions[ms.METRIC_REQUEST_COUNT_VIOLATED][1:]
        # n_turns = len(reqs)

        # for turn in range(0, played_turns):
        #     self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT, reqs[turn])
        #     self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT_PARSED, p_reqs[turn])
        #     self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT_VIOLATED, v_reqs[turn])

        # aborted = int(episode_interactions[ms.METRIC_ABORTED])
        # lose = int(episode_interactions[ms.METRIC_LOSE]) if not aborted else 0
        # success = 1 - lose if not aborted else 0
        # bench_score = complete_turns / n_turns if not aborted else np.nan

        # self.log_episode_score(ms.METRIC_ABORTED, aborted)
        # self.log_episode_score(ms.METRIC_LOSE, lose)
        # self.log_episode_score(ms.METRIC_SUCCESS, success)
        # self.log_episode_score(ms.METRIC_REQUEST_COUNT, sum(reqs))
        # self.log_episode_score(ms.METRIC_REQUEST_COUNT_PARSED, sum(p_reqs))
        # self.log_episode_score(ms.METRIC_REQUEST_COUNT_VIOLATED, sum(v_reqs))
        # self.log_episode_score(ms.METRIC_REQUEST_SUCCESS, sum(p_reqs) / sum(reqs))
        # self.log_episode_score(ms.BENCH_SCORE, bench_score)


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


# def main():
#     # select one instance
#     experiments = file_utils.load_json("in/instances.json", "dialoguequest")
#     instance = experiments["experiments"][0]["game_instances"][0]
#     master = DialogueQuest(instance, ["gpt-3.5-turbo", "gpt-3.5-turbo"])

#     master.setup(**instance)
#     master.play()


# if __name__ == "__main__":
#     main()
