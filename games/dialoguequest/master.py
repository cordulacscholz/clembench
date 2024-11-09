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
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
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
        self.avg_sentence_count_a = [0] * (self.game.max_turns + 1)
        self.avg_sentence_count_b = [0] * (self.game.max_turns + 1)

        # initialise episode scores
        self.conversational_turns_a = 0
        self.conversational_turns_b = 0

        self.goal = game_instance["goal"]
        self.slot_constraints = game_instance["slots_given"]
        self.slot_requests = game_instance["slots_to_fill"]
        self.user_goal = self._create_user_goal(self.slot_constraints, self.slot_requests)
        self.data = game_instance["data"]
        self.current_state = []
        self.final_choice = None
        self.final_suggestion = None

        self.log_next_turn()

        # initiate game with the instructions prompt
        self.game.initiate(self.initial_prompt_a, self.initial_prompt_b)

        action = {'type': 'send message', 'content': self.initial_prompt_a}
        self.log_event(from_='GM', to='Player 1', action=action)

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
            # print(f"VALID A: {valid_response_a}")
        else:
            print("ABORTING.")
            action = {'type': 'metadata', 'content': 'too many reprompts; abort'}
            self.log_event(from_='GM', to='GM', action=action)
            self.abort = True
            return False

        # Add B's initial prompt to the transcript
        if self.game.current_turn == 0:
            action = {'type': 'send message', 'content': self.initial_prompt_b}
            self.log_event(from_='GM', to='Player 2', action=action)

        # add A's reply to B's history
        self.game._append_utterance(answer_a, 'b', 'user')

        # Add to conversational turns counter
        self.conversational_turns_a += 1

        # Record character length of answer
        self.char_count_a[self.game.current_turn] = len(answer_a)
        self.word_count_a[self.game.current_turn] = len(answer_a.split())
        self.avg_sentence_count_a[self.game.current_turn] = self._calculate_avg_sentence_length(answer_a)

        # Check and break if Player A has uttered the fulfilment keyword
        if str(self.stop).lower() in answer_a.lower():
            action = {'type': 'fulfilled', 'content': 'Ending game...'}
            self.log_event(from_='GM', to='GM', action=action)
            self.fulfilled = True
            self.final_choice = str(answer_a.lower())
            return False

        # add A's reply to the transcript
        action = {'type': 'send message', 'content': answer_a}
        self.log_event(from_='GM', to='Player 2', action=action)

        valid_response_b = self._get_valid_response('b', self.game.current_turn)

        if valid_response_b and any(valid_response_b):
            prompt, raw_answer, answer_b, from_ = valid_response_b
            # print(f"VALID A: {valid_response_b}")
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
        self.avg_sentence_count_b[self.game.current_turn] = self._calculate_avg_sentence_length(answer_b)

        last_assistant_utterance = self.game.get_latest_relevant_utterance('b', role='assistant')
        last_user_utterance = self.game.get_latest_relevant_utterance('b', role='user')

        # merge summarisation prompt with last utterance to be summed up
        merged_json_prompt = f"{self.summarise_in_json_prompt}\n{last_user_utterance}\n\n{last_assistant_utterance}"

        valid_response_json = self._get_valid_json_response(merged_json_prompt, 'b', self.game.current_turn)

        # Validate json structure; if failure, abort
        if valid_response_json and any(valid_response_json):
            prompt, raw_answer, answer_a, from_ = valid_response_json
            # print(f"VALID JSON: {valid_response_json}")
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

    @staticmethod
    def _create_user_goal(constraints: dict, requests: list):
        user_goal = {}
        for key, value in constraints.items():
            user_goal[key] = value

        for request in requests:
            user_goal[request] = "required"
        return user_goal

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
        print(f"JSON ANSWER: {answer_in_json}")
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
        elif response.strip()[-1] not in [".", "?", "!", ")"] and self.stop not in response:
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
                if item.get(key_to_check) is not None:
                    if item.get(key_to_check).lower() == input_key.lower():
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

    @staticmethod
    def _calculate_avg_sentence_length(text: str):
        """Gets the average sentence length (including punctuation).

        Args:
            text (str): Input string

        Returns:
            float: Average length of sentences in input text
        """
        sentences = sent_tokenize(text)
        sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
        if sentence_lengths:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
        else:
            avg_length = 0
        return avg_length

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
        self.log_key('constraints', self.slot_constraints)
        self.log_key('User goal', self.user_goal)
        self.log_key('Final suggestion', self.final_suggestion)
        self.log_key('data', self.data)
        self.log_key('n_turns', self.game.current_turn+1)
        self.log_key('Complete turns', self.game.current_turn+1)
        self.log_key(ms.METRIC_REQUEST_COUNT, self.request_counts)
        self.log_key(ms.METRIC_REQUEST_COUNT_PARSED, self.parsed_request_counts)
        self.log_key(ms.METRIC_REQUEST_COUNT_VIOLATED, self.violated_request_counts)
        self.log_key(ms.METRIC_ABORTED, self.aborted)
        # self.log_key(ms.METRIC_SUCCESS, self.success)
        # self.log_key(ms.METRIC_LOSE, (1 if not self.success and not self.aborted else 0))
        self.log_key('Conversational turns A', self.conversational_turns_a)
        self.log_key('Conversational turns B', self.conversational_turns_b)
        self.log_key('Char count A', self.char_count_a)
        self.log_key('Word count A', self.word_count_a)
        self.log_key('Char count B', self.char_count_b)
        self.log_key('Word count B', self.word_count_b)
        self.log_key('Average sentence count A', self.avg_sentence_count_a)
        self.log_key('Average sentence count B', self.avg_sentence_count_b)


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
        user_goal = episode_interactions['User goal']
        data = episode_interactions['data']

        char_count_a = episode_interactions['Char count A']
        char_count_b = episode_interactions['Char count B']
        word_count_a = episode_interactions['Word count A']
        word_count_b = episode_interactions['Word count B']
        conversational_turns_a = episode_interactions['Conversational turns A']
        conversational_turns_b = episode_interactions['Conversational turns B']
        avg_sentence_count_a = episode_interactions['Average sentence count A']
        avg_sentence_count_b = episode_interactions['Average sentence count B']

        reqs = episode_interactions[ms.METRIC_REQUEST_COUNT]
        p_reqs = episode_interactions[ms.METRIC_REQUEST_COUNT_PARSED]
        v_reqs = episode_interactions[ms.METRIC_REQUEST_COUNT_VIOLATED]

        # success = int(episode_interactions[ms.METRIC_SUCCESS])
        # lose = int(episode_interactions[ms.METRIC_LOSE])

        for turn in range(0, played_turns):
            self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT, reqs[turn])
            self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT_PARSED, p_reqs[turn])
            self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT_VIOLATED, v_reqs[turn])
            self.log_turn_score(turn, "Character Count A", char_count_a[turn])
            self.log_turn_score(turn, "Character Count B", char_count_b[turn])
            self.log_turn_score(turn, "Word Count A", word_count_a[turn])
            self.log_turn_score(turn, "Word Count B", word_count_b[turn])
            self.log_turn_score(turn, "Average sentence count A", avg_sentence_count_a[turn])
            self.log_turn_score(turn, "Average sentence count B", avg_sentence_count_b[turn])

        # Episode level scores
        aborted = int(episode_interactions[ms.METRIC_ABORTED])
        tsr, accuracy_with_data, penalty = self._check_for_database_slots(final_suggestion, user_goal, data)
        success = 1 if tsr >= 0.9 else 0
        lose = 1 if not success and not aborted else 0

        self.log_episode_score("n Turns", n_turns)
        self.log_episode_score(ms.METRIC_REQUEST_COUNT, sum(reqs))
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_PARSED, sum(p_reqs))
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_VIOLATED, sum(v_reqs))
        self.log_episode_score(ms.METRIC_REQUEST_SUCCESS, sum(p_reqs) / sum(reqs))
        self.log_episode_score(ms.METRIC_SUCCESS, success)
        self.log_episode_score(ms.METRIC_LOSE, lose)
        self.log_episode_score(ms.METRIC_ABORTED, aborted)
        self.log_episode_score("Task Success Rate", tsr)
        self.log_episode_score("Accuracy of data", accuracy_with_data)
        self.log_episode_score("Penalty for invented slots", penalty)
        self.log_episode_score(ms.BENCH_SCORE, self._calculate_bench_score(tsr, accuracy_with_data, penalty) if not aborted else np.nan)
        self.log_episode_score("Conversational turns A", conversational_turns_a)
        self.log_episode_score("Conversational turns B", conversational_turns_b)
        self.log_episode_score("Average Char Count A", self.calculate_average_count(char_count_a, conversational_turns_a))
        self.log_episode_score("Average Char Count B", self.calculate_average_count(char_count_b, conversational_turns_b))
        self.log_episode_score("Average Word Count A", self.calculate_average_count(word_count_a, conversational_turns_a))
        self.log_episode_score("Average Word Count B", self.calculate_average_count(word_count_b, conversational_turns_b))
        self.log_episode_score("Average Sentence Count A", self.calculate_average_count(avg_sentence_count_a, conversational_turns_a))
        self.log_episode_score("Average Sentence Count B", self.calculate_average_count(avg_sentence_count_b, conversational_turns_b))

    def _calculate_task_success_rate(self, given_slots: dict, final_suggestion: dict, db_item: dict):
        """Calculates an accuracy of correctly generated goals. For constraints, checks if key/value pair is correct in final_suggestion and, as a fallback, in database item (since the slot might just have been implied). For requests, checks if there is a k/v pair in final_suggestion.

        Args:
            given_slots (dict): Goals which should have been reached
            final_suggestion (dict): Solution given
            db_item (dict): Database item corresponding to solution given

        Returns:
            float: Task Success Rate
        """
        total_goals = len(given_slots)
        successful_matches = 0

        for slot, expected_value in given_slots.items():
            # If the value is 'required', we only check for the presence of the slot in final_suggestion or db_item
            if expected_value == 'required':
                if slot in final_suggestion:
                    successful_matches += 1
            # Otherwise, compare the values directly using fuzzy matching
            else:
                # Check in final suggestion first
                if slot in final_suggestion and self._match_fuzzily(final_suggestion[slot], expected_value):
                    successful_matches += 1
                # Check in database item as a fallback
                elif db_item and slot in db_item and self._match_fuzzily(db_item[slot], expected_value):
                    successful_matches += 1
        return successful_matches / total_goals if total_goals > 0 else 0

    def _find_database_entry(self, final_suggestion: dict, data: list):
        """Finds the corresponding entry in the database based on id or name.

        Args:
            final_suggestion (dict): Solution given
            data (list): List of database items

        Returns:
            dict or None: Corresponding database item if found, else None
        """
        key_to_check = 'id' if 'id' in final_suggestion else 'name'
        input_key = final_suggestion[key_to_check]

        for item in data:
            if self._match_fuzzily(item.get(key_to_check), input_key):
                return item
        return None

    def _check_for_database_slots(self, final_suggestion: dict, given_slots: dict, data: list):
        """Check final suggestion against given slots and data for comparison.

        Args:
            final_suggestion (dict): Final object generated and suggested in dialogue.
            given_slots (dict): User goals or required slots.
            data (list): Original database items for comparison.

        Returns:
            float, float, int: Task success rate, accuracy with database, and penalty for invented slots.
        """
        penalty = 0
        acc_data = 0

        if not final_suggestion:
            return 0.0, 0.0, 0

        # Retrieve database entry using id or name as fallback
        db_item = self._find_database_entry(final_suggestion, data)

        # Calculate Task Success Rate using given slots and final suggestion
        tsr = self._calculate_task_success_rate(given_slots, final_suggestion, db_item)

        # Calculate Database Accuracy and Penalty for Invented Slots
        if db_item:
            total_pairs = len(final_suggestion)
            correct_vals = 0

            for k, v in final_suggestion.items():
                if k in db_item:
                    # Use fuzzy match if available or direct equality
                    if self._match_fuzzily(v, db_item[k]):
                        correct_vals += 1
                else:
                    # Slot is invented, add to penalty
                    penalty += 1

            acc_data = correct_vals / total_pairs if total_pairs > 0 else 0

        return tsr, acc_data, penalty

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

    @staticmethod
    def _calculate_bench_score(tsr, database_accuracy, penalty):
        return max(0, (tsr * 0.7) + (database_accuracy * 0.3) - (penalty * 0.1)) * 100


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
