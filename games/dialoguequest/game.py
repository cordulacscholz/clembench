import random
from typing import Dict, List, Tuple, Any

from clemgame.clemgame import Player
from backends import Model, CustomResponseModel
import copy


# Initialize players here

# class Instruction:

#     def __init__(self):
#         self.user_messages = []
#         self.system_messages = []

#     def add_user_message(self, message):
#         self.user_messages.append(message)

#     def add_system_message(self, message):
#         self.system_messages.append(message)

#     def convert_to_query_messages(self):
#         messages = []
#         messages.append({"role": "system", "content": ""})
#         for i in range(0, len(self.user_messages)):
#             messages.append({"role": "user", "content": self.user_messages[i]})

#             if i < len(self.system_messages):
#                 messages.append({"role": "assistant", "content": self.system_messages[i]})

#         return messages

    # def serialize(self):
    #     output = []

    #     for i in range(0, len(self.user_messages)):
    #         t = {"user": self.user_messages[i]}

    #         if i < len(self.system_messages):
    #             t["assistant"] = self.system_messages[i]
    #         output.append(t)
    #     return output

    # def get_last_user_message(self):
    #     return self.user_messages[-1]

    # def get_last_system_message(self):
    #     return self.system_messages[-1]


class Questioner(Player):
    """_summary_

    Args:
        Player (_type_): _description_
    """
    def __init__(self, model_name: str, player: str) -> None:
        super().__init__(CustomResponseModel())
        # super().__init__(model_name)
        # self.player: str = player

        # initialise list for storing dialogue history
        self.history: List = []

    def _custom_response(self, messages, turn_idx) -> str:
        utterance = f"Hello, I'm looking for something. TURN: {turn_idx}"
        return utterance


class Answerer(Player):
    """_summary_

    Args:
        Player (_type_): _description_
    """
    def __init__(self, model_name: str, player: str) -> None:
        super().__init__(CustomResponseModel())
        # super().__init__(model_name)
        # self.player: str = player

        # initialise list for storing dialogue history
        self.history: List = []

    def _custom_response(self, messages, turn_idx) -> str:
        utterance = f"No problem. Here's a suggestion: TURN: {turn_idx}"
        json_example = {'address': 'Hamilton Lodge'}
        return utterance


# Modify !!
class DialogueQuestGame:
    def __init__(self,
                 model_0: Model,
                 model_1: Model,
                 max_turns: int
                 ):
        self.max_turns: int = max_turns
        # self.questioner: Questioner = Questioner(model_0, "A")
        self.questioner: Questioner = model_0
        self.answerer: Answerer = model_1
        self.messages: List = []
        self.current_turn: int = 0

    def proceeds(self) -> bool:
        """Check if the game can continue: max number of turns not reached"""
        return self.current_turn < self.max_turns

    def initiate(self, initial_prompt_a: str, initial_prompt_b: str) -> None:
        """Add initial prompts to the dialogue history."""
        self.messages.append({'role': 'user', 'content': initial_prompt_a})
        self.messages.append({'role': 'assistant', 'content': initial_prompt_b})

    def get_utterance(self, player: str, current_turn) -> str:
        """Get utterance from a player and log it (firstlast specific)."""
        assert player in ('a', 'b')
        if player == 'a':
            # make an API call (or get a programmatic response) from player a
            prompt, raw_answer, answer = self.questioner(self.questioner.history, current_turn)
            from_ = "Player 1"
            # add reply to its own memory
            self._append_utterance(answer, 'a', 'assistant')
        else:
            # make an API call (or get a programmatic response) from player b
            prompt, raw_answer, answer = self.answerer(self.answerer.history,self.current_turn)
            from_ = "Player 2"
            # add reply to its own memory
            self._append_utterance(answer, 'b', 'assistant')
        return prompt, raw_answer, answer, from_

    def _append_utterance(self, utterance: str, player: str, role: str) -> None:
        """Add an utterance to the history of a player (firstlast specific)."""
        assert player in ('a', 'b')
        if player == 'a':
            self.questioner.history.append({'role': role, 'content': utterance})
        else:
            self.answerer.history.append({'role': role, 'content': utterance})

    def questioner_turn(self) -> str:
        """Append tagged next question to dialogue history and return it."""
        _, _, request = self.questioner(self.messages, self.current_turn)
        self.messages.append({'role': 'user', 'content': request})
        return request

    def answerer_turn(self) -> Tuple[Any, Any, str]:
        """
        Get response via API call, append it to dialogue history and return 
        manipulated prompt and response.
        """
        prompt, raw_answer, answer = self.answerer(self.messages,
                                                   self.current_turn)
        # make a copy to log a static state
        prompt = copy.deepcopy(prompt)
        self.messages.append({'role': 'assistant', 'content': answer})
        # increase the turn counter
        self.current_turn += 1
        return prompt, raw_answer, answer

    def summarise_in_json(self, prompt_text, answer, player):
        merged_prompt = f"{prompt_text}\n{answer}"
        _, _, request = self.answerer(merged_prompt, self.current_turn)
        self.messages.append({'role': 'system', 'content': request})
        self.current_turn += 1
        return request
        # answer_in_json = answer
        # return answer_in_json
