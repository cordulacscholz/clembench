"""Game to test abilities of task-oriented-dialogue modelling.
Implementation of a GameMaster to control game mechanisms.
"""

from typing import Dict, List
from backends import Model
from clemgame.clemgame import GameMaster, GameBenchmark, GameScorer, Player
from clemgame import get_logger
from clemgame import metrics as ms
from clemgame import file_utils
from games.dialoguequest.game import DialogueQuestGame, Questioner, Answerer
import json
import copy
import pythonmonkey
import numpy as np
from games.dialoguequest.constants import (
    GAME_NAME, LANG, MAX_TURNS, WORDS_PATH)


logger = get_logger(__name__)


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
        self.max_reprompts = 3

        # Load language specific words
        words = self.load_json(WORDS_PATH.format(LANG))
        self.stop = words['STOP']

        # initialise attributes that will be used for the evaluation scores
        self.aborted: bool = False
        self.success: bool = False
        self.complete_turns: int = 0
        self.all_slots_filled = False

    def setup(self, **game_instance):
        logger.info("setup")

        self.initial_prompt_a = game_instance["prompt_player_a"]
        self.initial_prompt_b = game_instance["prompt_player_b"]
        self.summarise_in_json_prompt = game_instance["summarise_in_json"]
        self.reprompt = game_instance["reprompt"]

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
            # 'Player 2': f'Answerer: {self.player_b.get_name()}'
            })

        # initialise common turn metrics
        self.request_counts = [0] * (self.game.max_turns + 1)
        self.parsed_request_counts = [0] * (self.game.max_turns + 1)
        self.violated_request_counts = [0] * (self.game.max_turns + 1)
        self.char_count_a = [0] * (self.game.max_turns + 1)
        self.char_count_b = [0] * (self.game.max_turns + 1)

        # initialise episode scores
        self.conversational_turns_a = 0
        self.conversational_turns_b = 0

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
        while self._does_game_proceed() and not self.aborted:
            self.log_next_turn()
            turn_goes_on = self.turn()

            # Break if Player A utters keyword
            if not turn_goes_on:
                print("Ending because task fulfilled.")
                self.success = True
                break

        action = {'type': 'end', 'content': 'Game finished.'}
        self.log_event(from_='GM', to='GM', action=action)
        self._log_eval_assets()

    def turn(self) -> bool:
        """Perform one turn, consisting in request Player A, answer Player B, json summary Player B. Might break early in case Player A utters the stop signal.

        Returns:
            bool: True if turn goes on, False if Player A utters stop signal
        """

        # print(f"QU history at {self.game.current_turn}: {self.game.questioner.history}")
        # print(f"A history at {self.game.current_turn}: {self.game.answerer.history}")

        logger.info('Game turn: %d', self.game.current_turn)
        print(f"CURRENT TURN: {self.game.current_turn}")

        # get request from questioner
        # prompt, raw_answer, answer_a, from_ = self.game.get_utterance('a', self.game.current_turn)

        prompt, raw_answer, answer_a, from_ = self._get_valid_response('a', self.game.current_turn)

        if not answer_a:
            action = {'type': 'metadata', 'content': 'too many reprompts; abort'}
            self.log_event(from_='GM', to='GM', action=action, call=(copy.deepcopy(prompt), raw_answer))

        # add A's reply to B's history
        self.game._append_utterance(answer_a, 'b', 'user')

        # add API call to the records
        # action = {'type': 'get message', 'content': answer_a}
        # self.log_event(from_=from_, to='GM', action=action, call=(copy.deepcopy(prompt), raw_answer))

        # also add the reply to the transcript
        action = {'type': 'send message', 'content': answer_a}
        self.log_event(from_='GM', to='Player 2', action=action)

        # increase the number of API requests
        self.request_counts[self.game.current_turn] += 1

        # Add to conversational turns counter
        self.conversational_turns_a += 1

        # Validate response
        # self._validate_response(answer_a, from_)

        # Record character length of answer
        if answer_a:
            self.char_count_a[self.game.current_turn] = len(answer_a)
        else:
            None

        if answer_a:
            # Break if Player A utters keyword indicating completion
            if str(self.stop).lower() in answer_a.lower():
                action = {'type': 'fulfilled', 'content': 'End game.'}
                self.log_event(from_='GM', to='GM', action=action)
                # print(self.game.messages)
                # self._build_json()
                self.game.current_turn += 1
                return False

        # get player B's reply and add it to its history
        # prompt, raw_answer, answer_b, from_ = self.game.get_utterance('b', self.game.current_turn)

        answer_b = "test"

        valid_response = self._get_valid_response('b', self.game.current_turn)

        if valid_response:
            prompt, raw_answer, answer_b, from_ = valid_response
        else:
            print("ABORTING.")
            self._does_game_proceed()
            self.aborted = True

        if not answer_b:
            action = {'type': 'metadata', 'content': 'too many reprompts; abort'}
            self.log_event(from_='GM', to='GM', action=action, call=(copy.deepcopy(prompt), raw_answer))

        # increase the number of API requests
        self.request_counts[self.game.current_turn] += 1

        # Add to conversational turns counter
        self.conversational_turns_b += 1

        # add API call to the records
        # action = {'type': 'get message', 'content': answer_b}
        # self.log_event(from_=from_, to='GM', action=action, call=(copy.deepcopy(prompt), raw_answer))

        # add B's reply to A's history
        self.game._append_utterance(answer_b, 'a', 'user')

        if answer_b:
            self.char_count_b[self.game.current_turn] = len(answer_b)
        else:
            self.char_count_b[self.game.current_turn] = 0

        # Grab the last content of the assistant for having it summed up in json structure
        # last_assistant_utterance = None
        # for message in reversed(self.game.answerer.history):
        #     if message['role'] == 'assistant':
        #         last_assistant_utterance = message['content']
        #         break
        last_assistant_utterance = self.game.get_last_relevant_utterance('b', role='assistant')

        # merge summarisation prompt with last utterance to be summed up
        merged_json_prompt = f"{self.summarise_in_json_prompt}\n{last_assistant_utterance}"

        answer_in_json = self.game.summarise_in_json(merged_json_prompt, self.game.answerer)
        action = {'type': 'send message', 'content': merged_json_prompt}
        self.log_event(from_='GM', to='Player 2', action=action)
        action = {'type': 'send message', 'content': answer_in_json}
        self.log_event(from_='Player 2', to='GM', action=action)

        # reprompt loop for model with n tries ?
        # Increase count (put into game class?)

        # TODO: If _update_current_goal_object: continue,
        # else: reprompt 3x, if not successful, abort
        self._update_current_goal_object(answer_in_json)

        # attempt = 0
        # Reprompt loop for getting json format right
        # while attempt < self.max_reprompts:
        #     attempt += 1
        #     if self._update_current_goal_object(answer_in_json):  # Call your function
        #         log: f"Success on attempt {attempt}"
        #         break  # Exit the loop if the function returns True
        #     else:
        #         log: f"Attempt {attempt} failed"
        # else:
        #     print("Function failed after 3 attempts, exiting.")

        # also add the reply to the transcript
        action = {'type': 'metadata', 'content': f"json format Answer: {answer_in_json}"}
        self.log_event(from_='GM', to='GM', action=action)

        # also add the reply to the transcript
        action = {'type': 'metadata', 'content': f"Updated internal object: {self.current_state}"}
        self.log_event(from_='GM', to='GM', action=action)

        # also add the reply to the transcript
        action = {'type': 'send message', 'content': answer_b}
        self.log_event(from_='GM', to='Player 1', action=action)

        # Increase requests count
        self.request_counts[self.game.current_turn] += 1

        self.game.current_turn += 1

        return True

    def _does_game_proceed(self):
        # if self.invalid_response:
        #     self.log_to_self("invalid format", "abort game")
        #     return False
        if self.game.current_turn >= self.game.max_turns:
            action = {'type': 'metadata', 'content': f"max turns reached {self.game.max_turns}"}
            self.log_event(from_='GM', to='GM', action=action)
            return False
        if self.aborted:
            return False
        return True

    # FIXME: Get prompt right!
    def _get_valid_response(self, player, current_turn):
        """Prompts the player for a valid response, reprompting up to 3 times.

        Args:
            player (str): The player's name.

        Returns:
            str: The valid response.
        """
        attempts = 0
        merged_prompt = None
        while attempts < self.max_reprompts:
            if attempts == 0:
                prompt, raw_answer, answer, from_ = self.game.get_utterance(player, current_turn)
            else:
                # Last utterance is last 
                last_utterance = self.game.get_last_relevant_utterance(player)
                merged_prompt = f"{self.reprompt}\n{last_utterance}"
                prompt, raw_answer, answer, from_ = self.game.summarise_or_reprompt(self.summarise_in_json_prompt, last_utterance, player)
                action = {'type': 'send message', 'content': merged_prompt}
                self.log_event(from_='GM', to='Player 2', action=action)

            # increase the number of API requests
            self.request_counts[self.game.current_turn] += 1

            # add API call to the records
            action = {'type': 'get message', 'content': answer}
            self.log_event(from_=from_, to='GM', action=action, call=(copy.deepcopy(prompt), raw_answer))

            if self._validate_response(answer, from_):
                return prompt, raw_answer, answer, from_
            print(f"not valid, execute else {attempts}")
            action = {'type': 'invalid response, try again', 'content': "invalid"}
            self.log_event(from_='GM', to='GM', action=action)
            self.game._append_utterance(merged_prompt, player, 'user')
            attempts += 1
        return None, None, None, None

    def _validate_response(self, response, from_):
        print(f"VALIDATION... {response}")
        if not response:
            print(f"CASE0")
            action = {'type': 'metadata', 'content': f"Response {from_} empty."}
            self.log_event(from_='GM', to='GM', action=action)
            # increase the number of violated requests
            self.violated_request_counts[self.game.current_turn] += 1
            return False
        # Check whether last sentence of answer is incomplete, exception for stop string
        elif response.strip()[-1] not in [".", "?", "!"] and self.stop not in response:
            print(f"CASE1")
            action = {'type': 'metadata', 'content': f"Response {from_} incomplete."}
            self.log_event(from_='GM', to='GM', action=action)
            # increase the number of violated requests
            self.violated_request_counts[self.game.current_turn] += 1
            return False
        else:
            print(f"CASE2")
            action = {'type': 'metadata', 'content': f"Response {from_} successfully parsed."}
            self.log_event(from_='GM', to='GM', action=action)
            # increase the number of parsed requests
            self.parsed_request_counts[self.game.current_turn] += 1
            return True

    def _validate_json(self):
        pass

    def _reprompt_for_json(self):
        pass
        # merged_prompt = reprompt_text + summarise prompt + contents (param)
        # self.game.summarise_in_json

    # FIXME: See if needed, depends on structure of goal object storage
    def _build_json(self):
        merged_prompt = f"{self.summarise_in_json_prompt}\n{self.game.messages}"
        final_json = self.game.summarise_in_json(merged_prompt, self.game.answerer)
        print(f"BUILD JSON FINAL: {final_json}")

    # TODO: Return either game state if valid json detected, or False if invalid json
    def _update_current_goal_object(self, answer_in_json: str):
        """_summary_

        Args:
            answer_in_json (Dict): _description_

        Returns:
            _type_: _description_
        """
        # print(type(answer_in_json))
        # print(answer_in_json)
        answer_in_json = self._repair_json(answer_in_json)
        try:
            answer_in_json = json.loads(answer_in_json)
        # if isinstance(answer_in_json, dict):
            action = {'type': 'metadata', 'content': 'update game state'}
            self.log_event(from_='GM', to='GM', action=action)
            for key in answer_in_json:
                if key in self.current_state:
                    self.current_state[key] = answer_in_json[key]

                if all(value is not None for value in self.current_state.values()):
                    self.all_slots_filled = True
                    action = {'type': 'metadata', 'content': 'slots filled'}
                    self.log_event(from_='GM', to='GM', action=action)
        except json.JSONDecodeError:
        # else:
            action = {'type': 'metadata', 'content': "JSONDecodeError: not updated"}
            self.log_event(from_='GM', to='GM', action=action)
        return self.current_state

    # @staticmethod
    def _repair_json(self, json_object):
        # json repair example
        # print(f"JSON OBJ: {json_object}")
        # Strip the string of its double quotes to avoid issued with single quotes
        # json_object.replace("\"", "")
        jsonrepair = pythonmonkey.require('jsonrepair').jsonrepair

        repaired = jsonrepair(json_object)
        print(f"Repaired json: {repaired}")
        if json_object != repaired:
            action = {'type': 'metadata', 'content': f"JSON repaired: {json_object} modified into {repaired}"}
            self.log_event(from_='GM', to='GM', action=action)
        return repaired

    def _log_eval_assets(self) -> None:
        """Log everything needed for the evaluation.
        """
        self.log_key('realised_slots', self.current_state)
        self.log_key('slots_given', self.slots_given)
        self.log_key('data', self.data)
        self.log_key('n_turns', self.game.current_turn)
        self.log_key('Complete turns', self.game.current_turn)
        self.log_key(ms.METRIC_REQUEST_COUNT, self.request_counts)
        self.log_key(ms.METRIC_REQUEST_COUNT_PARSED, self.parsed_request_counts)
        self.log_key(ms.METRIC_REQUEST_COUNT_VIOLATED, self.violated_request_counts)
        self.log_key('All slots filled', self.all_slots_filled)
        self.log_key(ms.METRIC_ABORTED, self.aborted)
        self.log_key(ms.METRIC_SUCCESS, self.success)
        self.log_key('Conversational turns A', self.conversational_turns_a)
        self.log_key('Conversational turns B', self.conversational_turns_b)
        self.log_key('average_char_count_a', self.char_count_a)
        self.log_key('average_char_count_b', self.char_count_b)


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

        # dictionary / list - rename into suggestions?
        realised_slots = episode_interactions['realised_slots']
        slots_given = episode_interactions['slots_given']
        data = episode_interactions['data']
        all_slots_filled = episode_interactions['All slots filled']

        char_count_a = episode_interactions['average_char_count_a']
        char_count_b = episode_interactions['average_char_count_b']
        conversational_turns_a = episode_interactions['Conversational turns A']
        conversational_turns_b = episode_interactions['Conversational turns B']

        reqs = episode_interactions[ms.METRIC_REQUEST_COUNT][0:]
        p_reqs = episode_interactions[ms.METRIC_REQUEST_COUNT_PARSED][0:]
        v_reqs = episode_interactions[ms.METRIC_REQUEST_COUNT_VIOLATED][0:]
        # n_turns = len(reqs)

        success = int(episode_interactions[ms.METRIC_SUCCESS])
        lose =  1 - success

        for turn in range(0, played_turns):
            self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT, reqs[turn])
            self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT_PARSED, p_reqs[turn])
            self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT_VIOLATED, v_reqs[turn])
            self.log_turn_score(turn, "Character Count A", char_count_a[turn])
            self.log_turn_score(turn, "Character Count B", char_count_b[turn])

        # Episode level scores
        # aborted = int(episode_interactions[ms.METRIC_ABORTED])
        # lose = int(episode_interactions[ms.METRIC_LOSE]) if not aborted else 0
        # success = 1 - lose if not aborted else 0
        # bench_score = played_turns / n_turns if not aborted else np.nan

        # self.log_episode_score(ms.METRIC_ABORTED, aborted)

        accuracy_slots_given = self.check_for_slots_given(realised_slots, slots_given)
        accuracy_data = self.check_for_database_slots(realised_slots, data)

        self.log_episode_score(ms.METRIC_REQUEST_COUNT, sum(reqs))
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_PARSED, sum(p_reqs))
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_VIOLATED, sum(v_reqs))
        self.log_episode_score(ms.METRIC_REQUEST_SUCCESS, sum(p_reqs) / sum(reqs))
        self.log_episode_score(ms.METRIC_SUCCESS, success)
        self.log_episode_score(ms.METRIC_LOSE, lose)
        self.log_episode_score("All slots filled", all_slots_filled)
        self.log_episode_score("Accuracy of slots given", accuracy_slots_given)
        self.log_episode_score("Accuracy of data", accuracy_data)
        # Placeholder score
        self.log_episode_score(ms.BENCH_SCORE, accuracy_slots_given)
        self.log_episode_score("Conversational turns A", conversational_turns_a)
        self.log_episode_score("Conversational turns B", conversational_turns_b)
        self.log_episode_score("Average Char Count A", self.calculate_average_char_count(char_count_a, conversational_turns_a))
        self.log_episode_score("Average Char Count B", self.calculate_average_char_count(char_count_b, conversational_turns_b))

    def check_for_slots_given(self, realised_slots, slots_given):
        """Calculate how many requested slots are acutally fulfilled in final suggestion.

        Args:
            realised_slots (_type_): _description_
            slots_given (_type_): _description_

        Returns:
            float: Slot accuracy
        """
        total_pairs = len(slots_given)
        # Count how many key-value pairs from dict_1 are found in dict_2
        matching_pairs = sum(1 for item in slots_given.items() if item in realised_slots.items())
        # Compute accuracy
        accuracy = matching_pairs / total_pairs if total_pairs > 0 else 0
        return accuracy

    # Implement!
    def check_for_database_slots(self, realised_slots, data):
        return 1

    @staticmethod
    def calculate_average_char_count(char_count, total):
        """Calculate an average character count.

        Args:
            char_count (list): List of characters used per turn.
            total (int): Number of turns of a player

        Returns:
            float: Average number of characters per turn
        """
        return round((sum(char_count) / total), 2)


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
