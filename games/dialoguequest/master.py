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
from thefuzz import fuzz
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
        self.fulfilled = False

    def setup(self, **game_instance):
        logger.info("setup")

        self.initial_prompt_a = game_instance["prompt_player_a"]
        self.initial_prompt_b = game_instance["prompt_player_b"]
        self.summarise_in_json_prompt = game_instance["summarise_in_json"]
        self.reprompt = game_instance["reprompt"]

        self.player_a = Questioner(self.player_models[0], "a")
        self.player_b = Answerer(self.player_models[1], "b")

        self.game_instance = game_instance
        self.game = DialogueQuestGame(self.player_a, self.player_b, self.max_turns)

        self.log_players({
            'GM': 'Game master for DialogueQuest',
            'Player 1': f'Questioner: {self.game.questioner}',
            'Player 2': f'Answerer: {self.game.answerer}'
            })

        # initialise common turn metrics
        self.request_counts = [0] * (self.game.max_turns + 1)
        self.parsed_request_counts = [0] * (self.game.max_turns + 1)
        self.violated_request_counts = [0] * (self.game.max_turns + 1)
        self.char_count_a = [0] * (self.game.max_turns + 1)
        self.char_count_b = [0] * (self.game.max_turns + 1)
        self.word_count_a = [0] * (self.game.max_turns + 1)
        self.word_count_b = [0] * (self.game.max_turns + 1)

        # initialise episode scores
        self.conversational_turns_a = 0
        self.conversational_turns_b = 0

        self.goal = game_instance["goal"]
        self.slots_given = game_instance["slots_given"]
        self.data = game_instance["data"]
        self.current_state = []
        self.final_choice = None
        self.final_suggestion = None

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
            action = {'type': 'metadata', 'content': f'Current turn: {self.game.current_turn}'}
            self.log_event(from_='GM', to='GM', action=action)
            if not self.turn():
                break

        # latest utterance Player A: assistant
        if self.fulfilled:
            action = {'type': 'metadata', 'content': f'final choice: {self.final_choice}'}
            self.log_event(from_='GM', to='GM', action=action)
            self.final_suggestion = self._select_final_suggestion()
            if self.final_suggestion:
                self.success = True
            action = {'type': 'metadata', 'content': f"Log final suggestion:{self.final_suggestion}"}
            self.log_event(from_='GM', to='GM', action=action)

        action = {'type': 'end', 'content': 'Game finished.'}
        self.log_event(from_='GM', to='GM', action=action)
        self._log_eval_assets()

    def turn(self) -> bool:
        """Perform one turn, consisting in request Player A, answer Player B, json summary Player B.

        Returns:
            bool: True if turn goes on, False if Player A utters stop signal
        """

        logger.info('Game turn: %d', self.game.current_turn)
        print(f"CURRENT TURN: {self.game.current_turn}")

        valid_response_a = self._get_valid_response('a', self.game.current_turn)

        # Validate that the values of answer_a are filled
        if valid_response_a and any(valid_response_a):
            prompt, raw_answer, answer_a, from_ = valid_response_a
        else:
            print("ABORTING.")
            action = {'type': 'metadata', 'content': 'too many reprompts; abort'}
            self.log_event(from_='GM', to='GM', action=action)
            self.abort = True
            return False

        # add the reply to the transcript
        action = {'type': 'send message', 'content': answer_a}
        self.log_event(from_='GM', to='Player 2', action=action)

        # add A's reply to B's history
        self.game._append_utterance(answer_a, 'b', 'user')

        # Add to conversational turns counter
        self.conversational_turns_a += 1

        # Record character length of answer
        self.char_count_a[self.game.current_turn] = len(answer_a)
        self.word_count_a[self.game.current_turn] = len(answer_a.split())

        # Check and break if Player A has uttered the fulfillment keyword
        if str(self.stop).lower() in answer_a.lower():
            action = {'type': 'fulfilled', 'content': 'End game.'}
            self.log_event(from_='GM', to='GM', action=action)
            self.fulfilled = True
            self.final_choice = str(answer_a.lower())
            # return False

        valid_response_b = self._get_valid_response('b', self.game.current_turn)

        if valid_response_b and any(valid_response_b):
            prompt, raw_answer, answer_b, from_ = valid_response_b
        else:
            print("ABORTING.")
            action = {'type': 'metadata', 'content': 'too many reprompts; abort'}
            self.log_event(from_='GM', to='GM', action=action)
            self.aborted = True
            return False

        # Add to conversational turns counter
        self.conversational_turns_b += 1

        # add B's reply to A's history
        self.game._append_utterance(answer_b, 'a', 'user')

        self.char_count_b[self.game.current_turn] = len(answer_b)
        self.word_count_b[self.game.current_turn] = len(answer_b.split())

        # If fulfillment keyword had been uttered by Player A, stop here
        if self.fulfilled:
            return False

        last_assistant_utterance = self.game.get_latest_relevant_utterance('b', role='assistant')

        # merge summarisation prompt with last utterance to be summed up
        merged_json_prompt = f"{self.summarise_in_json_prompt}\n{last_assistant_utterance}"

        valid_response_json = self._get_valid_json_response(merged_json_prompt, 'b', self.game.current_turn)

        # Validate json structure; if failure, abort
        if valid_response_json and any(valid_response_json):
            prompt, raw_answer, answer_a, from_ = valid_response_json
        else:
            print("ABORTING.")
            action = {'type': 'metadata', 'content': 'too many reprompts; abort'}
            self.log_event(from_='GM', to='GM', action=action)
            self.abort = True
            return False

        # also add the reply to the transcript
        action = {'type': 'send message', 'content': answer_b}
        self.log_event(from_='GM', to='Player 1', action=action)

        self.game.current_turn += 1

        return True

    def _does_game_proceed(self):
        """Check if game proceeds.

        Returns:
            bool: Fale if max turns reached or self.aborted==True, else False
        """
        if self.game.current_turn >= self.game.max_turns:
            action = {'type': 'metadata', 'content': f"max turns reached {self.game.max_turns}"}
            self.log_event(from_='GM', to='GM', action=action)
            return False
        if self.aborted:
            action = {'type': 'metadata', 'content': f"Game aborted because of invalid response"}
            self.log_event(from_='GM', to='GM', action=action)
            return False
        return True

    def _get_valid_response(self, player, current_turn):
        """Prompts the player for a valid response, reprompting up to 3 times.

        Args:
            player (str): The player's name.

        Returns:
            _type_: prompt, raw_answer, answer, from_ if successful, else None, None, None, None
        """
        attempts = 0
        merged_prompt = None
        while attempts <= self.max_reprompts:
            if attempts == 0:
                prompt, raw_answer, answer, from_ = self.game.get_utterance(player, current_turn)
            else:
                latest_utterance = self.game.get_latest_relevant_utterance(player, role='user')
                # First time in reprompt loop, create the merged prompt. Else prompt is already merged.
                if attempts == 1:
                    merged_prompt = f"{self.reprompt}\n{latest_utterance}"
                else:
                    merged_prompt = latest_utterance
                # print(f"MERGED PROMPT {merged_prompt}")
                prompt, raw_answer, answer, from_ = self.game.summarise_or_reprompt(merged_prompt, player)
                action = {'type': 'send message', 'content': merged_prompt}
                self.log_event(from_='GM', to=from_, action=action)

            # increase the number of API requests
            self.request_counts[self.game.current_turn] += 1

            # add API call to the records
            action = {'type': 'get message', 'content': answer}
            self.log_event(from_=from_, to='GM', action=action, call=(copy.deepcopy(prompt), raw_answer))

            if self._validate_text(answer, from_):
                return prompt, raw_answer, answer, from_
            print(f"not valid, execute else {attempts}")
            action = {'type': 'invalid response, try again', 'content': "invalid"}
            self.log_event(from_='GM', to='GM', action=action)
            attempts += 1
        return None, None, None, None

    def _get_valid_json_response(self, merged_prompt: str, player: str, current_turn: int):
        """Get a valid summary in json from a player. Reprompt self.max_reprompts times if answer not valid.

        Args:
            merged_prompt (str): Reprompt instruction + last relevant utterance
            player (str): Player prompted
            current_turn (int): current turn

        Returns:
            _type_: prompt, raw_answer, answer, from_ if successful, else None, None, None, None
        """
        attempts = 0
        while attempts <= self.max_reprompts:
            prompt, raw_answer, answer, from_ = self.game.summarise_or_reprompt(merged_prompt, player)
            action = {'type': 'send message', 'content': merged_prompt}
            self.log_event(from_='GM', to=from_, action=action)

            # increase the number of API requests
            self.request_counts[self.game.current_turn] += 1

            # add API call to the records
            action = {'type': 'get message', 'content': answer}
            self.log_event(from_=from_, to='GM', action=action, call=(copy.deepcopy(prompt), raw_answer))

            validated_json = self._validate_json(answer)

            if validated_json is not None:
                self._update_current_state(validated_json)
                action = {'type': 'Game state', 'content': f"updated game state: {self.current_state}"}
                self.log_event(from_='GM', to='GM', action=action)
                return prompt, raw_answer, answer, from_
            print(f"not valid, execute else {attempts}")
            action = {'type': 'invalid response, try again', 'content': "invalid"}
            self.log_event(from_='GM', to='GM', action=action)
            attempts += 1
        return None, None, None, None

    def _validate_json(self, answer_in_json: str):
        """Validates if string can be parsed as json.

        Args:
            answer_in_json (str): Relevant answer to be validated.

        Returns:
            dict: json structure of answer given
        """
        answer_in_json = self._repair_json(answer_in_json)
        try:
            parsed_json = json.loads(answer_in_json)
            action = {'type': 'metadata', 'content': "json successfully parsed"}
            self.log_event(from_='GM', to='GM', action=action)
            # increase the number of violated requests
            self.parsed_request_counts[self.game.current_turn] += 1
            return parsed_json
        except json.JSONDecodeError:
            # call reprompt loop
            action = {'type': 'metadata', 'content': "JSONDecodeError: not updated"}
            self.log_event(from_='GM', to='GM', action=action)
            # increase the number of violated requests
            self.violated_request_counts[self.game.current_turn] += 1

    def _validate_text(self, response, from_):
        """Validate text answer. Valid if answer not None and answer contains a closing punctuation mark as the last character.

        Args:
            response (str): answer to be validated
            from_ (str): player name

        Returns:
            bool: True if answer can be parsed, False if not.
        """
        if not response:
            action = {'type': 'metadata', 'content': f"Response {from_} empty."}
            self.log_event(from_='GM', to='GM', action=action)
            # increase the number of violated requests
            self.violated_request_counts[self.game.current_turn] += 1
            return False
        # Check whether last sentence of answer is incomplete, exception for stop string
        elif response.strip()[-1] not in [".", "?", "!"] and self.stop not in response:
            action = {'type': 'metadata', 'content': f"Response {from_} incomplete."}
            self.log_event(from_='GM', to='GM', action=action)
            # increase the number of violated requests
            self.violated_request_counts[self.game.current_turn] += 1
            return False
        else:
            action = {'type': 'metadata', 'content': f"Response {from_} successfully parsed."}
            self.log_event(from_='GM', to='GM', action=action)
            # increase the number of parsed requests
            self.parsed_request_counts[self.game.current_turn] += 1
            return True

    def _update_current_state(self, answer_in_json: list) -> None:
        """Updates the internal game state. Check if id or name in structure. If so: If current_state is empty, add element. If current_state already contains items, add new info to those.

        Args:
            answer_in_json (list): updated game state.
        """
        for item_answer_given in answer_in_json:
            if 'id' not in item_answer_given and 'name' not in item_answer_given:
                continue

            # update current_game_state
            action = {'type': 'metadata', 'content': 'update game state'}
            self.log_event(from_='GM', to='GM', action=action)

            key_to_check = 'id' if 'id' in item_answer_given and item_answer_given['id'] is not None else 'name'
            input_key = item_answer_given[key_to_check]

            if not self.current_state:
                print("appending!")
                self.current_state.append(item_answer_given)
                continue

            found_match = False

            # If key exists in current_state, update the corresponding item
            for item in self.current_state:
                # FIXME: Never overwrite with None values!
                if item.get(key_to_check) == input_key:
                    for key, value in item_answer_given.items():
                        # Update the corresponding item in current_state
                        if value is not None:
                            item[key] = value 
                            found_match = True
                    break  # Stop processing after updating

            if not found_match:
                self.current_state.append(item_answer_given)

    def _select_final_suggestion(self):
        """Select one final suggestion by logged final_choice.

        Returns:
            dict: Final suggestion agreed upon.
        """
        relevant_keys = ['id', 'name']
        for item in self.current_state:
            for key in relevant_keys:
                if key in item and item[key] is not None:
                    if str(item[key]).lower() in str(self.final_choice).lower():
                        return item

    def _repair_json(self, json_object: str):
        """Try to repair json if generated faulty.

        Args:
            json_object (str): Answer in json structure given.

        Returns:
            dict: json object (if parsable), else string
        """
        jsonrepair = pythonmonkey.require('jsonrepair').jsonrepair

        # Convert object to a JSON string if it's not already
        if not isinstance(json_object, str):
            try:
                json_object = json.dumps(json_object)
            except (TypeError, ValueError) as e:
                print(f"Error serializing object to JSON: {e}")
                return json_object

        try:
            # Attempt to repair the JSON string
            repaired = jsonrepair(json_object)

            # Log if repairs were made
            if json_object != repaired:
                action = {'type': 'metadata', 'content': f"JSON repaired: {repaired}"}
                self.log_event(from_='GM', to='GM', action=action)
            return repaired

        except Exception as e:
            # Handle cases where jsonrepair cannot fix the JSON
            print(f"Error repairing JSON: {e}")
            return json_object

    def _log_eval_assets(self) -> None:
        """Log everything needed for the evaluation.
        """
        self.log_key('slots_given', self.slots_given)
        self.log_key('Final suggestion', self.final_suggestion)
        self.log_key('data', self.data)
        self.log_key('n_turns', self.game.current_turn+1)
        self.log_key('Complete turns', self.game.current_turn+1)
        self.log_key(ms.METRIC_REQUEST_COUNT, self.request_counts)
        self.log_key(ms.METRIC_REQUEST_COUNT_PARSED, self.parsed_request_counts)
        self.log_key(ms.METRIC_REQUEST_COUNT_VIOLATED, self.violated_request_counts)
        self.log_key(ms.METRIC_ABORTED, self.aborted)
        self.log_key(ms.METRIC_SUCCESS, self.success)
        self.log_key(ms.METRIC_LOSE, (1 if not self.success and not self.aborted else 0))
        self.log_key('Conversational turns A', self.conversational_turns_a)
        self.log_key('Conversational turns B', self.conversational_turns_b)
        self.log_key('Char count A', self.char_count_a)
        self.log_key('Word count A', self.word_count_a)
        self.log_key('Char count B', self.char_count_b)
        self.log_key('Word count B', self.word_count_b)


class DialogueQuestScorer(GameScorer):
    def __init__(self, experiment: Dict, game_instance: Dict):
        super().__init__(GAME_NAME, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict):
        """Compute episode-level and turn-level scores.

        Args:
            episode_interactions (Dict): _description_
        """

        played_turns = episode_interactions['Complete turns']
        n_turns = episode_interactions['n_turns']

        final_suggestion = episode_interactions['Final suggestion']
        slots_given = episode_interactions['slots_given']
        data = episode_interactions['data']

        char_count_a = episode_interactions['Char count A']
        char_count_b = episode_interactions['Char count B']
        word_count_a = episode_interactions['Word count A']
        word_count_b = episode_interactions['Word count B']
        conversational_turns_a = episode_interactions['Conversational turns A']
        conversational_turns_b = episode_interactions['Conversational turns B']

        reqs = episode_interactions[ms.METRIC_REQUEST_COUNT]
        p_reqs = episode_interactions[ms.METRIC_REQUEST_COUNT_PARSED]
        v_reqs = episode_interactions[ms.METRIC_REQUEST_COUNT_VIOLATED]

        success = int(episode_interactions[ms.METRIC_SUCCESS])
        lose = int(episode_interactions[ms.METRIC_LOSE])

        for turn in range(0, played_turns):
            self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT, reqs[turn])
            self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT_PARSED, p_reqs[turn])
            self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT_VIOLATED, v_reqs[turn])
            self.log_turn_score(turn, "Character Count A", char_count_a[turn])
            self.log_turn_score(turn, "Character Count B", char_count_b[turn])
            self.log_turn_score(turn, "Word Count A", word_count_a[turn])
            self.log_turn_score(turn, "Word Count B", word_count_b[turn])

        # Episode level scores
        aborted = int(episode_interactions[ms.METRIC_ABORTED])
        accuracy_slots_given, accuracy_with_data, penalty = self._check_for_database_slots(final_suggestion, slots_given, data)

        self.log_episode_score("n Turns", n_turns)
        self.log_episode_score(ms.METRIC_REQUEST_COUNT, sum(reqs))
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_PARSED, sum(p_reqs))
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_VIOLATED, sum(v_reqs))
        self.log_episode_score(ms.METRIC_REQUEST_SUCCESS, sum(p_reqs) / sum(reqs))
        self.log_episode_score(ms.METRIC_SUCCESS, success)
        self.log_episode_score(ms.METRIC_LOSE, lose)
        self.log_episode_score(ms.METRIC_ABORTED, aborted)
        self.log_episode_score("Accuracy of slots given", accuracy_slots_given)
        self.log_episode_score("Accuracy of data", accuracy_with_data)
        self.log_episode_score(ms.BENCH_SCORE, self._calculate_bench_score(accuracy_slots_given, accuracy_with_data, penalty) if not aborted else np.nan)
        self.log_episode_score("Conversational turns A", conversational_turns_a)
        self.log_episode_score("Conversational turns B", conversational_turns_b)
        self.log_episode_score("Average Char Count A", self.calculate_average_count(char_count_a, conversational_turns_a))
        self.log_episode_score("Average Char Count B", self.calculate_average_count(char_count_b, conversational_turns_b))
        self.log_episode_score("Average Word Count A", self.calculate_average_count(word_count_a, conversational_turns_a))
        self.log_episode_score("Average Word Count B", self.calculate_average_count(word_count_b, conversational_turns_b))

    # FIXME: Deal with empty slots (key given, but not filled)
    def _check_for_database_slots(self, final_suggestion: dict, given_slots: dict, data: list):
        """Checks final suggestion with given slots and data for comparison.
        Args:
            final_suggestion (dict): Final object generated and suggested in dialogue
            data (list): Original database items for comparison

        Returns:
            float, float, int: Accuracy with requested slots, accuracy with database item, pentaly for invented slots
        """
        penalty = 0
        acc_slots = 0
        acc_data = 0
        if final_suggestion:
            selected_db_item = None
            key_to_check = 'id' if 'id' in final_suggestion else 'name'
            input_key = final_suggestion[key_to_check]

            # Search for corresponding item in data
            for item in data:
                if item[key_to_check] == input_key:
                    selected_db_item = item
                    break
            if not selected_db_item:
                acc_data = 0
                penalty = 0
            else:
                total_pairs = len(final_suggestion)
                correct_vals = 0
                # Check data_accuracy for final_suggestion against selected_db_item
                for k, v in final_suggestion.items():
                    if k in selected_db_item:
                        if self._match_fuzzily(v, selected_db_item[k]):
                            correct_vals += 1
                    else:
                        penalty += 1
                acc_data = correct_vals / total_pairs if total_pairs > 0 else 0

                # Check accuracy of given_slots
                checked_slots = set()  # To ensure no slot is counted twice

                for slot, value in given_slots.items():
                    # First, check in final_suggestion
                    if slot in final_suggestion and self._align_string(value) == self._align_string(final_suggestion[slot]):
                        acc_slots += 1
                        checked_slots.add(slot)
                    # If not in final_suggestion, check in selected_db_item
                    elif slot not in checked_slots and slot in selected_db_item and self._match_fuzzily(value, selected_db_item[slot]):
                        acc_slots += 1
                        checked_slots.add(slot)

        return acc_slots, acc_data, penalty

    def _match_fuzzily(self, item_a, item_b):
        """Checks if two items have at least 90% similarity. If items are of diffent types, break.

        Args:
            item_a (_type_): First item to compare, could be string, list or dict
            item_b (_type_): Second item to compare, could be string, list or dict

        Returns:
            bool: True if match, else False
        """
        threshold = 90
        match = False
        # Both items are strings
        if isinstance(item_a, str) and isinstance(item_b, str):
            if fuzz.partial_ratio(self._align_string(item_a), self._align_string(item_b)) >= threshold:
                match = True

        # Both items are lists
        elif isinstance(item_a, list) and isinstance(item_b, list):
            if len(item_a) == 0 or len(item_b) == 0:
                return False
            matched_count = 0
            min_length = min(len(item_a), len(item_b))
            for x, y in zip(item_a, item_b):
                if fuzz.ratio(self._align_string(str(x)), self._align_string(str(y))) >= threshold:
                    matched_count += 1
            if matched_count / min_length >= (threshold / 100):
                match = True

        # Both items are dicts: fuzzy match key/value pairs
        elif isinstance(item_a, dict) and isinstance(item_b, dict):
            if len(item_a) == 0 or len(item_b) == 0:
                return False
            matched_count = 0
            total_comparable_items = 0
            for k in item_a:
                if k in item_b:  # Only compare if the keys exist in both
                    total_comparable_items += 1
                    if fuzz.ratio(self._align_string(str(item_a[k])), self._align_string(str(item_b[k]))) >= threshold:
                        matched_count += 1
            if total_comparable_items > 0 and matched_count / total_comparable_items >= (threshold / 100):
                match = True

        # Mismatched types - no comparison
        else:
            match = False
        return match

    @staticmethod
    def _align_string(some_string: str):
        """Strips and modifies string for comparison.

        Args:
            some_string (str): Input string

        Returns:
            str: Modified string
        """
        return some_string.strip().lower()

    @staticmethod
    def calculate_average_count(char_count, total):
        """Calculate an average character count.

        Args:
            char_count (list): List of characters used per turn.
            total (int): Number of turns of a player

        Returns:
            float: Average number of characters per turn
        """
        return round((sum(char_count) / total), 2) if total != 0 else 0

    # FIXME: Cap metric at 0
    @staticmethod
    def _calculate_bench_score(request_accuracy, database_accuracy, penalty):
        return ((request_accuracy * 0.6) + (database_accuracy * 0.4) - (penalty * 0.1)) * 100
        # max(0, (request_accuracy * 0.6) + (database_accuracy * 0.4) - (penalty * 0.1))


class DialogueQuestBenchmark(GameBenchmark):
    """Organizes the run of a particular collection of game instances.

    Args:
        GameBenchmark (GameBenchmark): GameBenchmark
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
