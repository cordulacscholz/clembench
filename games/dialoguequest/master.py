"""Game to test abilities of task-oriented-dialogue modelling.
Implementation of a GameMaster to control game mechanisms.
"""

from typing import Dict, List, Tuple
from backends import Model
from clemgame.clemgame import GameMaster, GameBenchmark, GameScorer, Player
from clemgame import get_logger
from clemgame import file_utils
from clemgame import metrics as ms
from games.dialoguequest.game import DialogueQuestGame, Questioner, Answerer
import json
import copy
import numpy as np
from games.dialoguequest.constants import (
    GAME_NAME, MAX_TURNS, WORDS_PATH)


logger = get_logger(__name__)
LANG = 'en'


class DialogueQuest(GameMaster):
    """Play a single instance of a DialogueQuest.

    Args:
        GameMaster (_type_): _description_
        experiment
        player_models
    """
    def __init__(self, experiment: Dict, player_models: List[Model]):
        super().__init__(GAME_NAME, experiment, player_models)
        self.max_turns: int = MAX_TURNS

        # Load language specific words
        words = self.load_json(WORDS_PATH.format(LANG))
        self.stop = words['STOP']

        # initialise attributes that will be used for the evaluation scores
        self.aborted: bool = False
        self.lose: bool = False
        self.complete_turns: int = 0
        self.all_slots_filled = False

    def setup(self, **game_instance):
        logger.info("setup")

        self.initial_prompt_a = game_instance["prompt_player_a"]
        self.initial_prompt_b = game_instance["prompt_player_b"]
        self.summarisation_prompt = game_instance["summarisation_prompt"]
        self.summarise_in_json_prompt = game_instance["summarise_in_json"]

        # TODO: Might be only needed in game
        self.player_a = Questioner(self.player_models[0], "a")
        self.player_b = Answerer(self.player_models[1], "b")

        self.game_instance = game_instance
        self.game = DialogueQuestGame(self.player_a, self.player_b, self.max_turns)

        # TODO: Call get_name() function for actual models
        self.log_players({
            'GM': 'Game master for DialogueQuest',
            'Player 1': f'Questioner: {self.game.questioner}',
            'Player 2': f'Answerer: {self.game.answerer}'
            # 'Player 2': f'Answerer: {self.game.answerer.get_name()}'
            })

        # initialise common metrics
        self.request_counts = [0] * (self.game.max_turns + 1)
        self.parsed_request_counts = [0] * (self.game.max_turns + 1)
        self.violated_request_counts = [0] * (self.game.max_turns + 1)
        self.average_char_count_a = [0] * (self.game.max_turns + 1)
        self.average_char_count_b = [0] * (self.game.max_turns + 1)

        # Put this into the class constructor ?
        self.goal = game_instance["goal"]
        self.slots_given = game_instance["slots_given"]
        self.data = game_instance["data"]
        # For the current goal, collect all slots which are mentioned by the Answerer. Start with empty goal object to be filled up.
        self.current_state = {key: None for key in self.goal.keys()}

        self.log_next_turn()

        # initiate game with the instructions prompt
        self.game.initiate(self.initial_prompt_a, self.initial_prompt_b)

        action = {'type': 'send message', 'content': self.initial_prompt_a}
        self.log_event(from_='GM', to='Player 1', action=action)
        action = {'type': 'send message', 'content': self.initial_prompt_b}
        self.log_event(from_='GM', to='Player 2', action=action)

    def play(self) -> None:
        """Play one episode of DialogueQuest
        """
        while self.game.proceeds() and not self.aborted:
            self.log_next_turn()
            turn_goes_on = self.turn()

            # Break if Player A utters keyword
            if not turn_goes_on:
                print("Ending because task fulfilled.")
                break

        action = {'type': 'end', 'content': 'Game finished.'}
        self.log_event(from_='GM', to='GM', action=action)
        self._log_eval_assets()

    def turn(self) -> bool:
        """Perform one conversational turn."""

        # print(f"QU history at {self.game.current_turn}: {self.game.questioner.history}")
        # print(f"A history at {self.game.current_turn}: {self.game.answerer.history}")

        logger.info('Game turn: %d', self.game.current_turn)
        print(f"CURRENT TURN: {self.game.current_turn}")

        # get request from questioner
        prompt, raw_answer, answer_a, from_ = self.game.get_utterance('a', self.game.current_turn)

        # add A's reply to B's history
        self.game._append_utterance(answer_a, 'b', 'user')

        # add API call to the records
        action = {'type': 'get message', 'content': answer_a}
        self.log_event(from_=from_, to='GM', action=action, call=(copy.deepcopy(prompt), raw_answer))

        # also add the reply to the transcript
        action = {'type': 'send message', 'content': answer_a}
        self.log_event(from_='GM', to='Player 2', action=action)

        # increase the number of API requests
        self.request_counts[self.game.current_turn] += 1

        # Record character length of answer
        self.average_char_count_a[self.game.current_turn] = len(answer_a)

        # Break if Player A utters keyword indicating completion
        if str(self.stop).lower() in answer_a.lower():
            action = {'type': 'fulfilled', 'content': 'End game.'}
            self.log_event(from_='GM', to='GM', action=action)
            self.game.current_turn += 1
            return False

        # get player B's reply and add it to its history
        prompt, raw_answer, answer_b, from_ = self.game.get_utterance('b', self.game.current_turn)

        # increase the number of API requests
        self.request_counts[self.game.current_turn] += 1

        # add API call to the records
        action = {'type': 'get message', 'content': answer_b}
        self.log_event(from_=from_, to='GM', action=action, call=(copy.deepcopy(prompt), raw_answer))

        # add B's reply to A's history
        self.game._append_utterance(answer_b, 'a', 'user')

        self.average_char_count_b[self.game.current_turn] = len(answer_b)
        
        # Grab the last content of the assistant for having it summed up in json structure
        last_assistant_utterance = None
        for message in reversed(self.game.answerer.history):
            if message['role'] == 'assistant':
                last_assistant_utterance = message['content']
                break

        # merge summarisation prompt with last utterance to be summed up
        merged_prompt = f"{self.summarise_in_json_prompt}\n{last_assistant_utterance}"

        answer_in_json = self.game.summarise_in_json(merged_prompt, self.game.answerer)
        action = {'type': 'send message', 'content': merged_prompt}
        self.log_event(from_='GM', to='Player 2', action=action)
        action = {'type': 'send message', 'content': answer_in_json}
        self.log_event(from_='Player 2', to='GM', action=action)

        # TODO: integrate jsonrepair
        # Try repair, work with repaired object

        # reprompt loop for model with n tries ?
        # Increase count (put into game class?)

        self._update_current_goal_object(answer_in_json)

        # also add the reply to the transcript
        action = {'type': 'send message', 'content': answer_b}
        self.log_event(from_='GM', to='Player 1', action=action)

        # Increase requests count
        self.request_counts[self.game.current_turn] += 1

        self.game.current_turn += 1

        return True

    def _update_current_goal_object(self, answer_in_json: Dict):
        """_summary_

        Args:
            answer_in_json (Dict): _description_

        Returns:
            _type_: _description_
        """
        if isinstance(answer_in_json, dict):
            for key in answer_in_json:
                if key in self.current_state:
                    self.current_state[key] = answer_in_json[key]
                    action = {'type': 'metadata', 'content': 'update game state'}
                    self.log_event(from_='GM', to='GM', action=action)

                if all(value is not None for value in self.current_state.values()):
                    self.slots_filled = True
                    action = {'type': 'metadata', 'content': 'slots filled'}
                    self.log_event(from_='GM', to='GM', action=action)
        else:
            print("not updated")
        print(f"CURRENT STATE: {self.current_state}")
        return self.current_state

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

    def extract_json_from_response(self, utterance):
        """Extracts json code from a string.

        Args:
            utterance (str): String which contains potential json code.

        Returns:
            _type_: _description_
        """
        try:
            # Find the start of the JSON structure in the response
            json_start = str(utterance).find('{')
            # TODO: Maybe add a var for closing bracket?
            # Parse the JSON
            # FIXME: Check for double quotes!!
            # json_data = json.loads(utterance[json_start:].replace("\'", "\""))
            # print(f"JSON DATA: {json_data}")
            return json_data
        except json.JSONDecodeError:
            print(utterance[json_start:])
            print("Invalid JSON structure detected.")
            return utterance
            # return None

    def _log_eval_assets(self) -> None:
        """Log everything needed for the evaluation."""
        self.log_key('realised_slots', self.current_state)
        self.log_key('slots_given', self.slots_given)
        self.log_key('data', self.data)
        self.log_key('n_turns', self.game.current_turn)
        self.log_key('Complete turns', self.game.current_turn)
        self.log_key(ms.METRIC_REQUEST_COUNT, self.request_counts)
        self.log_key(ms.METRIC_REQUEST_COUNT_PARSED, self.parsed_request_counts)
        self.log_key(ms.METRIC_REQUEST_COUNT_VIOLATED, self.violated_request_counts)
        # self.log_key('Filled Slots', self.filled_slots)
        self.log_key(ms.METRIC_ABORTED, self.aborted)
        self.log_key(ms.METRIC_LOSE, self.lose)
        self.log_key("average_char_count_a", self.average_char_count_a)
        self.log_key("average_char_count_b", self.average_char_count_b)


class DialogueQuestScorer(GameScorer):
    def __init__(self, experiment: Dict, game_instance: Dict):
        super().__init__(GAME_NAME, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict):
        """Compute episode-level and turn-level scores.

        Args:
            episode_interactions (Dict): _description_
        """

        played_turns = episode_interactions['Complete turns']
        n_turns = episode_interactions['Complete turns']

        realised_slots = episode_interactions['realised_slots']
        slots_given = episode_interactions['slots_given']
        data = episode_interactions['data']
        self.log_episode_score("SLOTS GIVEN TEST", slots_given)

        char_count_a = episode_interactions['average_char_count_a']
        char_count_b = episode_interactions['average_char_count_b']

        reqs = episode_interactions[ms.METRIC_REQUEST_COUNT][0:]
        p_reqs = episode_interactions[ms.METRIC_REQUEST_COUNT_PARSED][1:]
        v_reqs = episode_interactions[ms.METRIC_REQUEST_COUNT_VIOLATED][1:]
        # n_turns = len(reqs)

        for turn in range(0, played_turns):
            self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT, reqs[turn])
            self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT_PARSED, p_reqs[turn])
            self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT_VIOLATED, v_reqs[turn])
            self.log_turn_score(turn, "char count a", char_count_a[turn])
            self.log_turn_score(turn, "char count b", char_count_b[turn])

        # Episode level scores
        # aborted = int(episode_interactions[ms.METRIC_ABORTED])
        # lose = int(episode_interactions[ms.METRIC_LOSE]) if not aborted else 0
        # success = 1 - lose if not aborted else 0
        # bench_score = played_turns / n_turns if not aborted else np.nan

        # self.log_episode_score(ms.METRIC_ABORTED, aborted)
        # self.log_episode_score(ms.METRIC_LOSE, lose)
        # self.log_episode_score(ms.METRIC_SUCCESS, success)

        accuracy_slots_given = self.check_for_slots_given(realised_slots, slots_given)
        accuracy_data = self.check_for_database_slots(realised_slots, data)

        self.log_episode_score(ms.METRIC_REQUEST_COUNT, sum(reqs))
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_PARSED, sum(p_reqs))
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_VIOLATED, sum(v_reqs))
        self.log_episode_score(ms.METRIC_REQUEST_SUCCESS, sum(p_reqs) / sum(reqs))
        self.log_episode_score("Accuracy of slots given", accuracy_slots_given)
        self.log_episode_score("Accuracy of data", accuracy_data)
        # self.log_episode_score(ms.BENCH_SCORE, bench_score)
        # FIXME: Collect + apply sum of reqs by A
        self.log_episode_score("Average Char Count A", sum(char_count_a) / sum(reqs))

    def check_for_slots_given(self, realised_slots, slots_given):
        total_pairs = len(slots_given)
        # Count how many key-value pairs from dict_1 are found in dict_2
        matching_pairs = sum(1 for item in slots_given.items() if item in realised_slots.items())
        # Compute accuracy
        accuracy = matching_pairs / total_pairs if total_pairs > 0 else 0
        return accuracy

    def check_for_database_slots(self, realised_slots, data):
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
    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return DialogueQuest(experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return DialogueQuestScorer(experiment, game_instance)

    def is_single_player(self) -> bool:
        return False


# def main():
#     # select one instance
#     experiments = file_utils.load_json("in/instances.json", "dialoguequest")
#     instance = experiments["experiments"][0]["game_instances"][0]
#     master = DialogueQuest(instance, ["mock", "mock"])

#     master.setup(**instance)
#     master.play()


# if __name__ == "__main__":
#     main()
