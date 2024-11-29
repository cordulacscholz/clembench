from typing import List

from clemgame.clemgame import Player
from clemgame import get_logger
from backends import Model, CustomResponseModel


logger = get_logger(__name__)


# Initialise player classes
class Questioner(Player):
    """Questioner who acts as the User.

    Args:
        Player (Player): instance of the Player class
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
            utterance = f"Hello, I'm looking for something. TURN{turn_idx}."
        else:
            utterance = "Ok, game FULFILLED. choosing the Igel's inn. number 3456789"
        return utterance


class Answerer(Player):
    """Answerer who acts as the Assistant.

    Args:
        Player (Player): instance of the Player class
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
        if turn_idx < 1:
            utterance = "[{'id': '8475875', 'address': 'Hamilton Lodge]."
        elif turn_idx == 1:
            # utterance = {'address': 'Hamilton Lodge'}
            utterance = "[{'name': 'Igel's Inn', 'people': '200'}, {'ONE': 'ONE', 'TWO': 'TWO'}, {'id': '3456789', 'address': 'Nordbahnstraße 3'}, {'address': null, 'id': '8475875', 'name': 'test', 'additional_key': 'hello!]."
            # utterance = "[{'name': 'Igel's Inn', 'people': '200'}, {'ONE': 'ONE', 'TWO': 'TWO'}, {'id': '3456789', 'address': 'Nordbahnstraße 3'}, {'address': 'Test Drive 4', 'id': '8475875', 'name': 'test', 'additional_key': 'hello!]."
        else:
            utterance = "Hi!"
        return utterance


class DialogueQuestGame:
    """Instance of the DialogueQuest Game.
    """
    def __init__(self,
                 model_0: Model,
                 model_1: Model,
                 max_turns: int
                 ):
        self.max_turns: int = max_turns
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
        # mock turn to ensure alternation of roles
        self.answerer.history.append({'role': 'assistant', 'content': "OK"})

    def get_utterance(self, player: str, current_turn: int) -> str:
        """Get utterance from a model and log it.

        Args:
            player (str): Name of player
            current_turn (int): Current turn of the game

        Returns:
            _type_: prompt: Prompt, raw_answer: dict of answer, answer: answer string, from_: Player
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
        """Add an utterance to the history of a player.
        """
        assert player in ('a', 'b')
        if player == 'a':
            self.questioner.history.append({'role': role, 'content': utterance})
            # self.messages.append(utterance)
        else:
            self.answerer.history.append({'role': role, 'content': utterance})
            # self.messages.append(utterance)

    def get_latest_relevant_utterance(self, player, role='user'):
        """Grabs the last content of a specified player.

        Args:
            player (str): Name of player
            role (str, optional): Role of player, either user or assistant. Defaults to 'user'.

        Returns:
            str: Relevant message selected.
        """
        assert player in ('a', 'b')
        if player == 'a':
            history = self.questioner.history
        else:
            history = self.answerer.history
        latest_relevant_message = None
        for message in reversed(history):
            if message['role'] == role:
                latest_relevant_message = message['content']
                break
        return latest_relevant_message

    def summarise_or_reprompt(self, prompt, player):
        """Appends utterance to history and requests a new answer.

        Args:
            prompt (str): Prompt text for summary or reprompt
            player (str): Name of player

        Returns:
            _type_: prompt, raw_answer, answer, from_
        """
        assert player in ('a', 'b')

        # Add reprompt text to player's history
        self._append_utterance(prompt, player, 'user')
        prompt, raw_answer, answer, from_ = self.get_utterance(player, self.current_turn)

        return prompt, raw_answer, answer, from_
