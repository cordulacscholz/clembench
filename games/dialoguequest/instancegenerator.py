"""Generate instances as json files for DialogueQuest.
"""

import os
import random
import json
import string

from clemgame.clemgame import GameInstanceGenerator
from games.dialoguequest.constants import (
    GAME_NAME, N_INSTANCES, MAX_TURNS, SEED, TOPICS, WORDS_PATH)

LANG = 'en'
JSON_PREFIX = 'JSON'


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
        prompt_a = self.load_template('resources/initial_prompts/prompt_a')
        prompt_b = self.load_template('resources/initial_prompts/prompt_b')

        # TODO: Set up relevant params
        for n in range(0, 5):
            experiment = self.add_experiment(n)

            for game_id in range(N_INSTANCES):
                topic = self._select_topic()
                goal_object = self.sample_random_json_object(topic)
                example_object = self.sample_random_json_object(topic)
                # Ensure that goal and example object are not the same
                while example_object == goal_object:
                    example_object = self.sample_random_json_object(topic)
                categorical_slots, non_categorical_slots = self.get_cat_and_non_cat_keys(topic)
                # Select NUMBER of cat slots for SLOTS_GIVEN
                internal_object, slots_given, slots_to_fill = self.select_slots(goal_object, categorical_slots, non_categorical_slots)
                # Select NUMBER of non_cat / unsued cat slots for SLOTS_TO_FILL
                instance = self.add_game_instance(experiment, game_id)
                instance['prompt_player_a'] = self.create_prompt_a(prompt_a, topic, slots_given, slots_to_fill, example_object)
                instance['prompt_player_b'] = self.create_prompt_b(prompt_b, topic, example_object)
                instance['max_turns'] = MAX_TURNS
                instance['goal'] = goal_object

    def _select_topic(self):
        """Selects a topic out of possible lists of topics
        """
        topic = random.choice(TOPICS)
        return topic

    def create_prompt_a(self, prompt: str, topic: str, slots_given, slots_to_fill, example_object) -> str:
        """Fill in the initial prompt variables."""
        text = prompt.replace('$TOPIC$', topic)
        text = text.replace('$SLOTS_GIVEN$', str(slots_given))
        text = text.replace('$SLOTS_TO_FILL$', str(slots_to_fill))
        text = text.replace('$EXAMPLE$', str(example_object))
        print(text)
        return text

    def create_prompt_b(self, prompt: str, topic: str, example_object):
        data = self.load_database_file(topic)
        text = prompt.replace('$DATA$', str(data))
        text = text.replace('$JSON_PREFIX', JSON_PREFIX)
        text = text.replace('$EXAMPLE$', str(example_object))
        return text

    def select_slots(self, goal_object, categorical_slots, non_categorical_slots):
        goal_object_empty = {key: None for key in goal_object}
        # TODO: See what could be a good way to choose/modify this
        number_of_slots = 2
        slots_given = random.sample(categorical_slots, number_of_slots)
        slots_to_fill = random.sample(non_categorical_slots, number_of_slots)
        # TODO: Pick the according slots from the goal object
        # TODO: Filter out "no" values
        return goal_object_empty, slots_given, slots_to_fill

    def load_database_file(self, topic):
        file_path = os.path.join(os.path.dirname(__file__), 'resources/database_files', f'{topic}.json')
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                return (data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"An error occurred: {e}")
            return None

    def sample_random_json_object(self, topic):
        """Sample one example json object from data file.

        Args:
            topic (_type_): _description_
        """
        # file_path = os.path.join(os.path.dirname(__file__), 'resources/database_files', f'{topic}.json')
        # try:
        #     with open(file_path, 'r') as file:
        #         data = json.load(file)
        #         selected_object = random.choice(data)
        #         return selected_object
        # except (FileNotFoundError, json.JSONDecodeError) as e:
        #     print(f"An error occurred: {e}")
        #     return None
        data = self.load_database_file(topic)
        selected_object = random.choice(data)
        return selected_object

    def get_cat_and_non_cat_keys(self, topic):
        file_path = os.path.join(os.path.dirname(__file__), 'resources/database_files', 'schema.json')
        with open(file_path, 'r') as file:
            data = json.load(file)
            for service in data:
                if service['service_name'] == topic:
                    categorical_slots = [slot['name'] for slot in service['slots'] if slot['is_categorical']]
                    non_categorical_slots = [slot['name'] for slot in service['slots'] if not slot['is_categorical']]
                    return categorical_slots, non_categorical_slots


if __name__ == '__main__':
    random.seed(SEED)
    DialogueQuestInstanceGenerator().generate()
