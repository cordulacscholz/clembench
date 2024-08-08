"""Generate instances as json files for DialogueQuest.
"""

import random
import string

from clemgame.clemgame import GameInstanceGenerator
import constants


GAME_NAME = constants.GAME_NAME
N_INSTANCES = constants.N_INSTANCES
SEED = constants.SEED


class DialogueQuestInstanceGenerator(GameInstanceGenerator):
    """generate() method creates JSON file with key 'experiments', list of experiments.

    Args:
        GameInstanceGenerator (_type_): _description_
    """
    def __init__(self):
        super().__init__("dialoguequest")

    # TODO: Specify on_generate
    # Variations on topic, variations with same topic...
    def on_generate(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        # topic
        prompt_a = self.load_template('resources/initial_prompts/dummy_prompt_a_hotel')
        prompt_b = self.load_template('resources/initial_prompts/dummy_prompt_b_hotel')

        # TODO: Set up relevant params
        for n in range(0, 5):
            experiment = self.add_experiment(n)

            for game_id in range(N_INSTANCES):
                instance = self.add_game_instance(experiment, game_id)
                instance['prompt_player_a'] = prompt_a
                instance['prompt_player_b'] = prompt_b
                instance['max_turns'] = 10

    # Fill prompt template
    # TODO: Modify!
    def create_prompt(self,
                      topic: str,
                      prompt: str,
                      letter: str,
                      n_turns: int) -> str:
        """Replace a prompt template with slot values."""
        text = string.Template(prompt).substitute(topic=topic, letter=letter,
                                                  nturns=n_turns)
        return text


if __name__ == '__main__':
    random.seed(SEED)
    DialogueQuestInstanceGenerator().generate()
