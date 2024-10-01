"""Generate instances as json files for DialogueQuest.
"""

import os
import random
import json
import string
import math
from copy import deepcopy

from clemgame.clemgame import GameInstanceGenerator
from clemgame import get_logger
from games.dialoguequest.constants import (
    GAME_NAME, N_INSTANCES, N_EXPERIMENTS, MAX_TURNS, SEED, TOPICS, WORDS_PATH)

LANG = 'en'
JSON_PREFIX = 'JSON'

logger = get_logger(__name__)


class DialogueQuestInstanceGenerator(GameInstanceGenerator):
    """generate() method creates JSON file with key 'experiments', list of experiments.

    Args:
        GameInstanceGenerator (_type_): _description_
    """
    def __init__(self):
        super().__init__("dialoguequest")
        self.language = LANG
        words = self.load_json(WORDS_PATH.format(LANG))
        self.stop = words['STOP']

    # Variations on topic, variations with same topic...
    def on_generate(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        # topic
        print("Generating instance...")
        prompt_a = self.load_template('resources/initial_prompts/prompt_a')
        prompt_b = self.load_template('resources/initial_prompts/prompt_b')
        summarisation_prompt = self.load_template('resources/initial_prompts/summarisation_prompt')
        summarise_in_json = self.load_template('resources/initial_prompts/summarise_in_json')

        for n in range(0, N_EXPERIMENTS):
            experiment = self.add_experiment(n)

            for game_id in range(N_INSTANCES):
                topic = self._select_topic()
                article = self._select_article(topic)
                goal_object = self._sample_random_json_object(topic)
                example_object = self._sample_random_json_object(topic)

                # For testing purposes, select only 3 items from whole database
                sample_data = self._sample_multi_random_json_object(topic)

                # Insert goal object at random index in list of sample data
                # TODO: Integrate switch for 'level' selection (goal object included, goal object not necessarily included)
                selected_data = deepcopy(sample_data)
                random_index = random.randint(0, len(selected_data))
                selected_data.insert(random_index, goal_object)

                # Ensure that goal and example object are not the same
                while example_object == goal_object:
                    example_object = self._sample_random_json_object(topic)
                categorical_slots, non_categorical_slots = self._get_cat_and_non_cat_keys(topic)
                # Select NUMBER of cat slots for SLOTS_GIVEN
                slots_given, slots_to_fill = self._select_slots(goal_object, categorical_slots, non_categorical_slots)

                # Add all relevant parameters to instance file
                instance = self.add_game_instance(experiment, game_id)
                instance['summarisation_prompt'] = summarisation_prompt
                instance['summarise_in_json'] = summarise_in_json
                instance['prompt_player_a'] = self._create_prompt_a(prompt_a, topic, article, slots_given, slots_to_fill, self.stop, example_object)
                instance['prompt_player_b'] = self._create_prompt_b(prompt_b, topic, example_object, selected_data)
                instance['max_turns'] = MAX_TURNS
                instance['slots_given'] = slots_given
                instance['slots_to_fill'] = slots_to_fill
                instance['goal'] = goal_object
                instance['data'] = selected_data

    def _select_topic(self):
        """Selects a topic out of possible lists of topics
        """
        topic = random.choice(TOPICS)
        return topic

    def _create_prompt_a(self, prompt: str, topic: str, article: str, slots_given, slots_to_fill, stop: str, example_object) -> str:
        """Fill in the initial prompt variables."""
        text = prompt.replace('$TOPIC$', topic)
        text = text.replace('$ARTICLE$', article)
        text = text.replace('$SLOTS_GIVEN$', str(slots_given))
        text = text.replace('$SLOTS_TO_FILL$', str(slots_to_fill))
        text = text.replace('$STOP$', str(stop))
        text = text.replace('$EXAMPLE$', str(example_object))
        return text

    def _create_prompt_b(self, prompt: str, topic: str, example_object, selected_data):
        data = selected_data
        text = prompt.replace('$DATA$', str(data))
        text = text.replace('$JSON_PREFIX$', JSON_PREFIX)
        text = text.replace('$EXAMPLE$', str(example_object))
        return text

    # TODO: Adjust for AR, RU, ZH (suffixes)
    def _select_article(self, topic):
        """Adjustment for article for language specific prompt depending on noun(EN, DE).

        Args:
            topic (str): Topic of game instance

        Returns:
            string: Article to be used in prompt
        """
        if self.language.strip().lower() == "en":
            article = "an" if topic.lower()[0] in ["a", "e", "i", "o", "u"] else "a"
            print(article)
        elif self.language == "de":
            if topic in ['attraction']:
                article = "eine"
            elif topic in ['train']:
                article = "einen"
            else:
                article = "ein"
        else:
            article = ""
        return article

    # Deal with "ref" keys
    # Filter out "no" vals
    # Refine key selection
    def _select_slots(self, goal_object, categorical_slots, non_categorical_slots):
        # TODO: See what could be a good way to choose/modify this
        # Check if slots make sense to be selected
        # Remove key-values pairs with val=='no', as this does not work for a goal ("need hotel with no internet")
        filtered_goal_object = {key: goal_object[key] for key in categorical_slots if key in goal_object and goal_object[key] != "no"}
        # Filter number_of_slots by difficulty or some similar category?
        number_of_slots = math.floor(len(filtered_goal_object)/2)
        random_keys_given = random.sample(list(filtered_goal_object.keys()), number_of_slots)
        slots_given = {key: filtered_goal_object[key] for key in random_keys_given}

        slots_to_fill = random.sample(non_categorical_slots, number_of_slots)
        return slots_given, slots_to_fill

    def _load_database_file(self, topic):
        file_path = os.path.join(os.path.dirname(__file__), 'resources/database_files', f'{topic}.json')
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                return (data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"An error occurred: {e}")
            return None

    def _sample_random_json_object(self, topic):
        """Sample one example json object from data file.

        Args:
            topic (_type_): _description_
        """
        data = self._load_database_file(topic)
        selected_object = random.choice(data)
        return selected_object

    def _sample_multi_random_json_object(self, topic):
        data = self._load_database_file(topic)
        selected_objects = random.sample(data, 3)
        return selected_objects

    def _get_cat_and_non_cat_keys(self, topic):
        file_path = os.path.join(os.path.dirname(__file__), 'resources/database_files', 'schema.json')
        with open(file_path, 'r') as file:
            data = json.load(file)
            for service in data:
                if service['service_name'] == topic:
                    categorical_slots = [slot['name'].split('-')[1] for slot in service['slots'] if slot['is_categorical']]
                    non_categorical_slots = [slot['name'].split('-')[1] for slot in service['slots'] if not slot['is_categorical']]
                    return categorical_slots, non_categorical_slots


if __name__ == '__main__':
    random.seed(SEED)
    DialogueQuestInstanceGenerator().generate()
