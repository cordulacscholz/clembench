from typing import List

from clemgame.clemgame import Player
from clemgame import get_logger
from backends import Model, CustomResponseModel


logger = get_logger(__name__)


# Initialise player classes
class Questioner(Player):
    """_summary_

    Args:
        Player (Player): _description_
    """
    def __init__(self, model_name: str, player: str) -> None:
        # Programmatic player mode for testing
        # super().__init__(CustomResponseModel())
        super().__init__(model_name)
        self.player: str = player

        # initialise list for storing dialogue history
        self.history: List = []

    def _custom_response(self, messages, turn_idx) -> str:
        """Predefined response if player is not an LLM (for testing purposes)

        Args:
            messages (list): _description_
            turn_idx (int): Current turn

        Returns:
            str: Utterance
        """
        if turn_idx <= 1:
            utterance = f"Hello, I'm looking for something. TURN{turn_idx}"
        else:
            utterance = "Ok, game FULFILLED"
        return utterance


class Answerer(Player):
    """_summary_

    Args:
        Player (_type_): _description_
    """
    def __init__(self, model_name: str, player: str) -> None:
        # Programmatic player mode for testing
        # super().__init__(CustomResponseModel())
        super().__init__(model_name)
        self.player: str = player

        # initialise list for storing dialogue history
        self.history: List = []

    def _custom_response(self, messages, turn_idx) -> str:
        """Predefined response if player is not an LLM (for testing purposes)

        Args:
            messages (list): _description_
            turn_idx (int): Current turn

        Returns:
            str: Utterance
        """
        if turn_idx <= 3:
            utterance = f"No problem. Here's a suggestion: TURN{turn_idx}"
        else:
            utterance = {'address': 'Hamilton Lodge'}
        json_example = {'address': 'Hamilton Lodge'}
        # return json_example
        return utterance


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
        self.game_proceeds = True

    def proceeds(self) -> bool:
        """Check if the game can continue: max number of turns not reached

        Returns:
            bool: False if max number of turns reached, else True
        """
        if not self.game_proceeds:
            return False
        if self.current_turn >= self.max_turns:
            logger.info(f"Maximum turns: {self.max_turns} reached")
            return False
        else:
            return True

    def initiate(self, prompt_player_a: str, prompt_player_b: str) -> None:
        """Initialise the dialogue history.

        Args:
            prompt_player_a (str): Initial prompt Player A
            prompt_player_b (str): Initial prompt Player B
        """
        # append the initial message of each player to their history
        self.questioner.history.append({'role': 'user', 'content': prompt_player_a})
        self.answerer.history.append({'role': 'user', 'content': prompt_player_b})
        # Mock turn to ensure alternation of roles
        self.answerer.history.append({'role': 'assistant', 'content': "OK"})

    def get_utterance(self, player: str, current_turn: int) -> str:
        """Get utterance from a model and log it.

        Args:
            player (str): Name of player
            current_turn (int): Current turn of the game

        Returns:
            str: _description_
        """
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
        """Add an utterance to the history of a player."""
        assert player in ('a', 'b')
        if player == 'a':
            self.questioner.history.append({'role': role, 'content': utterance})
        else:
            self.answerer.history.append({'role': role, 'content': utterance})

    def summarise_in_json(self, merged_prompt, player):
        self.answerer.history.append({'role': 'user', 'content': merged_prompt})

        # get request from questioner
        prompt, raw_answer, request, from_ = self.get_utterance('b', self.current_turn)

        # _, _, request = self.answerer(merged_prompt, self.current_turn)

        # self.messages.append({'role': 'system', 'content': request})
        # self.current_turn += 1
        # from_ = "Player 1"
        # add reply to its own memory
        # Mock turn for ensuring alteration
        self._append_utterance("Ok", 'a', 'assistant')
        self._append_utterance("Ok", 'b', 'user')
        self._append_utterance(request, 'b', 'assistant')
        self._append_utterance(request, 'a', 'user')
        return request
        # answer_in_json = answer
        # return answer_in_json
