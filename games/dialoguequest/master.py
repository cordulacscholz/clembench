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
    GAME_NAME, MAX_TURNS)


logger = get_logger(__name__)


# class Questioner(Player):
#     """_summary_

#     Args:
#         Player (_type_): _description_
#     """
#     def __init__(self, model_name: str, player: str) -> None:
#         super().__init__(model_name)
#         self.player: str = player

#         # list for storing dialogue history
#         self.history: List = []

#     def _custom_response(self, messages, turn_idx) -> str:
#         utterance = f"{messages} TURN: {turn_idx}"
#         return utterance


# class Answerer(Player):
#     """_summary_

#     Args:
#         Player (_type_): _description_
#     """
#     def __init__(self, model_name: str, player: str) -> None:
#         super().__init__(model_name)
#         self.player: str = player

#         self.history: List = []

#     def _custom_response(self, messages, turn_idx) -> str:
#         """_summary_

#         Args:
#             messages (str): _description_
#             turn_idx (int): _description_

#         Returns:
#             str: message including number of turn
#         """
#         utterance = f"{messages} TURN: {turn_idx}"
#         return utterance


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

        self.questioner = Questioner(self.player_models[0], "A")
        self.answerer = Answerer(self.player_models[1], "B")

        self.game_instance = game_instance
        self.game = DialogueQuestGame(self.questioner, self.answerer, self.max_turns)

        self.log_players({
            'GM': 'Game master for DialogueQuest',
            'Player 1': f'Questioner: {self.questioner}',
            'Player 2': f'Answerer: {self.answerer}'
            })

        # initialise game variables
        self.current_turn: int = 0
        self.log_key('n_turns', self.current_turn)

        # ? Investigate !
        # initialise common metrics
        self.request_counts = [0] * (self.max_turns + 1)
        self.parsed_request_counts = [0] * (self.max_turns + 1)
        self.violated_request_counts = [0] * (self.max_turns + 1)

        # add initial prompts to each player's messages
        # self.initiate(self.initial_prompt_a, self.initial_prompt_b)

        # Put this into the class constructor ?
        self.goal = game_instance["goal"]
        # For the current goal, collect all slots which are mentioned by the Answerer. Start with empty goal object to be filled up.
        self.current_state = {key: None for key in self.goal.keys()}

    def play(self) -> None:
        self.log_next_turn()
        # initiate game with the instructions prompt
        self.game.initiate(self.initial_prompt_a, self.initial_prompt_b)

        action = {'type': 'send message', 'content': self.initial_prompt_a}
        self.log_event(from_='GM', to='Player 1', action=action)
        action = {'type': 'send message', 'content': self.initial_prompt_b}
        self.log_event(from_='GM', to='Player 2', action=action)

        # self.log_event(from_='GM', to='Player 1', action=action)

        while self.game.proceeds() and not self.aborted:
            self.log_next_turn()
            self.turn()

            # if not turn_successful:
            #     action = {'type': 'invalid format', 'content': 'Abort: invalid format in slot filling.'}
            #     self.log_event(from_='GM', to='GM', action=action)
            #     self.aborted = True
            #     break

        self.log_key('realised_slots', self.current_state)
        action = {'type': 'end', 'content': 'Game finished.'}
        self.log_event(from_='GM', to='GM', action=action)
        self._log_eval_assets()

    def turn(self) -> bool:
        """Perform one conversational turn."""

        logger.info('Game turn: %d', self.game.current_turn)
        print(f"CURRENT TURN: {self.game.current_turn}")

        # get request from questioner
        prompt, raw_answer, answer_a, from_ = self.game.get_utterance('a', self.game.current_turn)

        # add API call to the records
        action = {'type': 'get message', 'content': answer_a}
        self.log_event(from_=from_, to='GM', action=action, call=(copy.deepcopy(prompt), raw_answer))

        print(prompt)

        # increase the number of API requests
        self.request_counts[self.current_turn] += 1

        # get player B's reply and add it to its history
        prompt, raw_answer, answer_b, from_ = self.game.get_utterance('b', self.game.current_turn)

        # add A's reply to B's history
        self.game._append_utterance(answer_a, 'b', 'user')
        # also add the reply to the transcript
        action = {'type': 'send message', 'content': answer_b}
        self.log_event(from_='GM', to='Player 1', action=action)

        # add B's reply to A's history
        self.game._append_utterance(answer_b, 'a', 'user')
        # also add the reply to the transcript
        action = {'type': 'send message', 'content': answer_a}
        self.log_event(from_='GM', to='Player 2', action=action)

        # request = self.game.questioner_turn(self.questioner, self.current_turn)
        # print(f"REQUEST: {request}")

        # pass on request to answerer
        # action = {'type': 'get message', 'content': request}
        # self.log_event(from_='Player 2', to='GM', action=action)

        # action = {'type': 'send message', 'content': request}
        # self.log_event(from_='GM', to='Player 1', action=action)

        # get answer from answerer
        # prompt, raw_answer, answer = self.game.answerer_turn()
        # print(f"ANSWER: {answer}")
        # action = {'type': 'get message', 'content': answer}
        # call = (prompt, raw_answer)
        # # print(f"CALL: ")
        # self.log_event(from_='Player 1', to='GM', action=action, call=call)
        # at this point, turn count has just been increased, so it matches the
        # current probing turn

        # Reprompt here to summarise json
        # current answer from Player B -> Player B again (issue with roles??)
        # Pseude turns as in PrivatedShared!

        answer_in_json = self.game.summarise_in_json(self.summarise_in_json_prompt, self.answerer.history[-1], self.answerer)
        action = {'type': 'send message', 'content': answer_in_json}
        # self.log_event(from_='GM', to='Player 2', action=action, call=call)
        print(answer_in_json)

        # Update the current state of the goal dictionary
        self._update_current_goal_object(answer_in_json)
        print(self.current_state)
        # TODO: verify json structure before passing to function
        # json repair tool
        # reprompt loop for model with n tries ?

        # Increase turn count
        self.request_counts[self.game.current_turn] += 1

        return True

    # def initiate(self, prompt_player_a: str, prompt_player_b: str) -> None:
    #     """Initialise the dialogue history (firstlast specific)."""
    #     # always call log_next_turn what a turn starts
    #     self.log_next_turn()

    #     # append the initial message of each player to their history
    #     # the value user means the message is from an interlocutor of the model
    #     self.questioner.history.append({'role': 'user', 'content': prompt_player_a})
    #     self.answerer.history.append({'role': 'user', 'content': prompt_player_b})

    #     # also log the messages as events for the transcriptions
    #     action = {'type': 'send message', 'content': prompt_player_a}
    #     self.log_event(from_='GM', to='Player 1', action=action)
    #     action = {'type': 'send message', 'content': prompt_player_b}
    #     self.log_event(from_='GM', to='Player 2', action=action)

    def _update_current_goal_object(self, answer_in_json: Dict):
        for key in answer_in_json:
            if key in self.current_state:
                self.current_state[key] = answer_in_json[key]
                action = {'type': 'metadata', 'content': 'update game state'}
                self.log_event(from_='GM', to='GM', action=action)

            if all(value is not None for value in self.current_state.values()):
                self.slots_filled = True

    # Example for logging
    # def _log_probing_outcome(self, probe: Dict, successful: bool, tries: int):
    #     if not successful:
    #         content = NOT_SUCCESS.format(probe['target'])
    #     else:
    #         content = SUCCESS.format(probe['target'], tries)
    #     # answer valid?
    #     action = {'type': 'metadata', 'content': content}
    #     self.log_event(from_='GM', to='GM', action=action)
    #     logger.info(content)
    #     # answer correct?
    #     result = '' if probe['value'] == probe['gt'] else 'in'
    #     action = {'type': 'check', 'content': RESULT.format(result)}
    #     self.log_event(from_='GM', to='GM', action=action)

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
        self.log_key('Aborted', self.aborted)
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
            slots_filled = False

            for event in turn:
                action = event["action"]
                if action["type"] == "invalid format":
                    invalid_response = True
                if action["type"] == "all slots successfully filled":
                    slots_filled = True

            if invalid_response:
                turn_score["violated_request_count"] = 1
                turn_score["parsed_request_count"] = 0
            else:
                turn_score["violated_request_count"] = 0
                turn_score["parsed_request_count"] = 1

            self.log_turn_score(turn_idx, 'Accuracy', 1 if slots_filled else 0)
            self.log_turn_score(turn_idx, ms.METRIC_REQUEST_COUNT_VIOLATED, turn_score["violated_request_count"])
            self.log_turn_score(turn_idx, ms.METRIC_REQUEST_COUNT_PARSED, turn_score["parsed_request_count"])
            self.log_turn_score(turn_idx, ms.METRIC_REQUEST_COUNT, turn_score["request_count"])

            turn_scores.append(turn_score)

        violated_request_count = sum([turn["violated_request_count"] for turn in turn_scores])
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_VIOLATED, violated_request_count)

        parsed_request_count = sum([turn["parsed_request_count"] for turn in turn_scores])
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_PARSED, parsed_request_count)

        request_count = sum([turn["request_count"] for turn in turn_scores])
        self.log_episode_score(ms.METRIC_REQUEST_COUNT, request_count)

        self.log_episode_score(ms.METRIC_REQUEST_SUCCESS, parsed_request_count / request_count)


        # Common metrics
        if invalid_response:  # whether a violation of the game rules happened (response not parsable)
            self.log_episode_score(ms.METRIC_ABORTED, 1)
            self.log_episode_score(ms.METRIC_SUCCESS, 0)
            self.log_episode_score(ms.METRIC_LOSE, 0)
            # Game-specific metrics
            self.log_episode_score(ms.BENCH_SCORE, np.nan)  # metric not applicable
        else:
            self.log_episode_score(ms.METRIC_ABORTED, 0)
            if slots_filled:
                self.log_episode_score(ms.METRIC_SUCCESS, 1)
                self.log_episode_score(ms.METRIC_LOSE, 0)
                self.log_episode_score(ms.BENCH_SCORE, 100 / len(turn_scores))  # how early the guesser found the word
            else:
                self.log_episode_score(ms.METRIC_SUCCESS, 0)
                self.log_episode_score(ms.METRIC_LOSE, 1)
                self.log_episode_score(ms.BENCH_SCORE, 0)  # word not found
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
    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return DialogueQuest(experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return DialogueQuestScorer(experiment, game_instance)

    def is_single_player(self) -> bool:
        return False


def main():
    # select one instance
    experiments = file_utils.load_json("in/instances.json", "dialoguequest")
    instance = experiments["experiments"][0]["game_instances"][0]
    master = DialogueQuest(instance, ["mock", "mock"])

    master.setup(**instance)
    master.play()


if __name__ == "__main__":
    main()
