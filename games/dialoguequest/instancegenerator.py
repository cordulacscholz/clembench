"""Generate instances as json files for DialogueQuest.
"""

import os
import random
import json
import math
from copy import deepcopy

from clemgame.clemgame import GameInstanceGenerator
from clemgame import get_logger
from games.dialoguequest.constants import (
    GAME_NAME, LANG, N_INSTANCES, N_EXPERIMENTS, MAX_TURNS, N_DATABASE_ITEMS, SEED, TOPICS, WORDS_PATH)


logger = get_logger(__name__)


class DialogueQuestInstanceGenerator(GameInstanceGenerator):
    """generate() method creates JSON file with key 'experiments', list of experiments.

    Args:
        GameInstanceGenerator (GameInstanceGenerator): _description_
    """
    def __init__(self):
        super().__init__("dialoguequest")
        self.language = LANG
        words = self.load_json(WORDS_PATH.format(LANG))
        self.stop = words['STOP']

    def on_generate(self):
        """Generates instances of DialogueQuest

        Returns:
            dict: Instances file for DialogueQuest
        """
        print("Generating instance...")
        prompt_a = self.load_template('resources/initial_prompts/prompt_a')
        prompt_b = self.load_template('resources/initial_prompts/prompt_b')
        summarise_in_json = self.load_template('resources/initial_prompts/summarise_in_json')
        reprompt = self.load_template('resources/initial_prompts/reprompt')

        reprompt_options = ["reprompt", "no_reprompt"]

        for reprompt_option in reprompt_options:
        # for topic in TOPICS:
        # for n in range(0, N_EXPERIMENTS):
            # experiment = self.add_experiment(n)
            # experiment = self.add_experiment(topic)
            self.reprompt_option = True if reprompt_option == "reprompt" else False
            experiment = self.add_experiment(reprompt_option)

            for game_id in range(N_INSTANCES):
                topic = self._select_topic()
                article = self._select_article(topic)
                example_object = self._sample_random_json_objects(topic, 1)

                # Select restricted number of database items available to the Answerer (to avoid exceeding the model's token limit)
                sample_data = self._sample_random_json_objects(topic, N_DATABASE_ITEMS)

                # Select goal object
                goal_object = random.choice(sample_data)
                selected_data = deepcopy(sample_data)

                # Ensure that goal and example object are not the same
                while example_object == goal_object:
                    example_object = self._sample_random_json_objects(topic, 1)
                # Sort out categorical and non-categorial slots according to topic
                categorical_slots, non_categorical_slots = self._get_cat_and_non_cat_keys(topic)
                # Select NUMBER of cat slots for SLOTS_GIVEN
                slots_given, slots_to_fill = self._select_slots(goal_object, categorical_slots, non_categorical_slots)

                # Add all relevant parameters to instance file
                instance = self.add_game_instance(experiment, game_id)
                instance['prompt_player_a'] = self._create_prompt_a(prompt_a, topic, article, slots_given, slots_to_fill, self.stop, example_object)
                instance['prompt_player_b'] = self._create_prompt_b(prompt_b, example_object, selected_data)
                instance['summarise_in_json'] = self._create_summarisation_prompt(summarise_in_json, example_object)
                instance['reprompt'] = reprompt
                instance['max_turns'] = MAX_TURNS
                instance['slots_given'] = slots_given
                instance['slots_to_fill'] = slots_to_fill
                instance['goal'] = goal_object
                instance['data'] = selected_data
                instance['reprompt_option'] = self.reprompt_option

    @staticmethod
    def _select_topic():
        """Selects a topic out of possible lists of topics
        """
        topic = random.choice(TOPICS)
        return topic

    @staticmethod
    def _create_prompt_a(prompt: str, topic: str, article: str, slots_given, slots_to_fill, stop: str, example_object) -> str:
        """Fill slot template for Player A with selected values

        Args:
            prompt (str): Prompt template to be filled
            topic (str): Topic slected for instance
            article (str): Article for topic (EN, DE specific)
            slots_given (_type_): _description_
            slots_to_fill (_type_): _description_
            stop (str): Signal for indicating fulfilment
            example_object (dict): _description_

        Returns:
            str: Complete prompt for direct use
        """
        text = prompt.replace('$TOPIC$', topic)
        text = text.replace('$ARTICLE$', article)
        text = text.replace('$SLOTS_GIVEN$', str(slots_given))
        text = text.replace('$SLOTS_TO_FILL$', str(slots_to_fill))
        text = text.replace('$STOP$', str(stop))
        text = text.replace('$EXAMPLE$', str(example_object))
        return text

    @staticmethod
    def _create_prompt_b(prompt: str, example_object, selected_data):
        """Fill slot template for Player B with selected values

        Args:
            prompt (str): Prompt template to be filled
            topic (str): Topic slected for instance
            example_object (dict): Random example object for orientation in structure
            selected_data (dict): Collection of selected instances to be chosen from

        Returns:
            str: Complete prompt for direct use
        """
        data = selected_data
        text = prompt.replace('$DATA$', str(data))
        text = text.replace('$EXAMPLE$', str(example_object))
        return text

    @staticmethod
    def _create_summarisation_prompt(prompt: str, example_object: dict):
        text = prompt.replace('$EXAMPLE$', str(example_object))
        return text

    # TODO: Adjust for AR, RU, ZH
    def _select_article(self, topic):
        """Adjustment for article for language specific prompt depending on noun(EN, DE).

        Args:
            topic (str): Topic of game instance

        Returns:
            string: Article to be used in prompt
        """
        if self.language.strip().lower() == "en":
            article = "an" if topic.lower()[0] in ["a", "e", "i", "o", "u"] else "a"
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

    def _select_slots(self, goal_object: dict, categorical_slots: list, non_categorical_slots: list):
        """Selects slots which are given (categorical slots) and slots which are to be filled (non-categorical slots) from goal object

        Args:
            goal_object (dict): Item used as goal
            categorical_slots (list): List of categorical slots
            non_categorical_slots (list): List of non-categorical slots

        Returns:
            dict, dict: Slots given, Slots to fill
        """
        # Choose constrains (k/v pairs)

        # Get all keys which are categorical (~informable)
        # Remove key-values pairs with val=='no', as this does not work for a goal ("need hotel with no internet")
        requestable_slots_goal = {key: goal_object[key] for key in categorical_slots if key in goal_object and goal_object[key] != "no"}

        print(f"GOAL: {requestable_slots_goal}")

        # Randomly choose slots the constraints should be taken from
        # MultiWOZ constraint average: 2-3
        n_slots_to_choose = self._choose_number(len(requestable_slots_goal), preferred_range=(2, 3))
        print(n_slots_to_choose)

        # Randomly select slots as constraints + add their values
        random_keys_given = random.sample(list(requestable_slots_goal.keys()), n_slots_to_choose)
        print(f"RANDOMGIVEN: {random_keys_given}")
        slots_given = {key: requestable_slots_goal[key] for key in random_keys_given}
        print(f"GIVEN: {slots_given}")

        # Get all keys which are non-categorical (~requestable)
        # MultiWOZ constraint average: 3-4
        available_slots_to_request = [slot for slot in non_categorical_slots if slot in goal_object]
        print(f"REQUEST: {available_slots_to_request}")

        n_slots_to_request = self._choose_number(len(available_slots_to_request), preferred_range=(2, 3))
        print(n_slots_to_request)

        slots_to_fill = random.sample(available_slots_to_request, n_slots_to_request)
        print(f"TOFILL: {slots_to_fill}")
        return slots_given, slots_to_fill

    @staticmethod
    def _load_database_file(topic):
        """Loads a file from database

        Args:
            topic (str): Topic chosen for instance

        Returns:
            dict: Data selected, or None if an error occurs
        """
        file_path = os.path.join(os.path.dirname(__file__), 'resources/database_files', f'{topic}.json')
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                return (data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"An error occurred: {e}")
            return None

    def _sample_random_json_objects(self, topic: str, n: int):
        """Randomly selects items from list of json objects from database file for slelected topic

        Args:
            topic (str): Topic selected for instance
            n (int): Number of items to be chosen

        Returns:
            dict: Items selected
        """
        data = self._load_database_file(topic)
        if n == 1:
            selected = random.choice(data)
        else:
            selected = random.sample(data, n)
        return selected

    @staticmethod
    def _choose_number(list_length, preferred_range=(2, 3)):
        """Chooses a number based on the list length and preferred range.

        Args:
            list_length: The length of the list.
            preferred_range: A tuple of the preferred minimum and maximum numbers.

        Returns:
            The chosen number.
        """

        min_preferred, max_preferred = preferred_range
        if list_length >= max_preferred:
            return random.choice(range(min_preferred, max_preferred + 1))
        elif list_length >= min_preferred:
            return list_length
        else:
            return 1

    @staticmethod
    def _get_cat_and_non_cat_keys(topic: str):
        """Selects categorical and non-categorical slots from the database file given for selected topic

        Args:
            topic (str): Topic selected for instance

        Returns:
            list, list: Categorical slots, Non-categorical slots
        """
        file_path = os.path.join(os.path.dirname(__file__), 'resources/database_files', 'schema.json')
        with open(file_path, 'r') as file:
            data = json.load(file)
            for service in data:
                if service['service_name'] == topic:
                    categorical_slots = [slot['name'].split('-')[1] for slot in service['slots'] if slot['is_categorical']]
                    non_categorical_slots = [slot['name'].split('-')[1] for slot in service['slots'] if not slot['is_categorical']]
                    return categorical_slots, non_categorical_slots


if __name__ == '__main__':
    SEED = random.seed()
    DialogueQuestInstanceGenerator().generate()
