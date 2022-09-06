from .rules_induction import RulesInductorSklearn, RulesInductorChefboost
from owlready2 import *
import os
import json
import pandas as pd
from time import time, process_time
import numpy as np
from chefboost import Chefboost as chef


ONTO = "jp_masters_project.owl"
DEFAULT_LEARNING_ALGORITHM = "sklearn"
DEFAULT_REASONER = "pellet"
KNOWLEDGE_FOLDER = "knowledge_folder"
RESULTS_FOLDER = "results"
DEFAULT_DATASET = "actions_taken.csv"
DEFAULT_RESULTS_FILE = "analysis_results.json"


class ReasoningAndLearningSystemPrototype:
    """ The representation of the system's main core """
    def __init__(self, ask_rate=0.1):
        self.onto = None  # the ontology representing the main knowledge
        self.rules_inductor = None  # chosen inductor responsible for machine learning process
        self.sync_reasoner = None  # function, activating the chosen reasoner
        self.ask_rate = ask_rate  # probability of activating the 'ask the user' procedure
        self.rng = None  # random number generator for the system
        self.backup_memory = dict()  # backup memory with saved classification results
        # set of statistics measured during the analysis
        self.statistics = {
            "satisfaction_growth": [0],
            "reasoning_count_growth": [0],
            "learning_count_growth": [0],
            "asking_count_growth": [0],
            "exec_time": [0],
            "rules_num": [0],
            "average_rule_body_length": [0]
        }

    @staticmethod
    def action_roleplay(action_name):
        """ 'Decorative' function, returning the string output in accordance to the classification result """
        # additional dictionary for one of the options
        medium_dictionary = {"Playing_some_games": "play a game",
                             "Reading_a_book": "read a book",
                             "Watching_a_movie": "watch a movie"}
        # individual representations of certain actions
        if action_name == "Verbal_greeting":
            return "'Hello to you human. It is good to see you.'"
        elif action_name == "Telling_a_joke":
            return "'Why did the chicken cross the road?'"
        elif action_name == "Getting_closer":
            return "*Comes closer to the user*"
        elif action_name == "Hand_wave":
            return "*Waves hand at the user to draw their attention*"
        elif action_name == "Staying_quiet":
            return "*Decides not to do anything*"
        elif action_name == "Thumb_up":
            return "*Shows the 'thumbs up' gesture*"
        elif action_name == "User_comforting":
            return "'Please, do not worry, human. Everything is going to be alright.'"
        elif action_name == "Question_about_feeling":
            return "'How do you feel today, human? Is everything alright?'"
        elif action_name == "Friend_or_family_talk_recommendation":
            return "'Maybe it would be a good idea to call your friend or family member?'"
        elif action_name == "Doing_some_exercises":
            return "'It is time for some exercises. Here is the plan for today...'"
        elif action_name == "Going_for_a_walk":
            return "'I recommend taking a walk and getting some fresh air.'"
        elif action_name == "Ordering_some_food":
            return "'Initiating the food order sequence...'"
        # the representation for the actions about choosing a certain medium (book, movie, game)
        if action_name in ["Playing_some_games", "Reading_a_book", "Watching_a_movie"]:
            return f"'Maybe you'd like to {medium_dictionary[action_name]}? The pick for today is...'"
        # the representation for the actions about choosing a music playlist
        if "music" in action_name:
            act_name = " ".join(action_name.split('_'))
            return f"'Opening the '{act_name}' playlist. Now playing...'"

    def load_components(self, onto_name, rules_inductor_type=DEFAULT_LEARNING_ALGORITHM, reasoner_type=DEFAULT_REASONER,
                        random_state=None):
        """ This method is used to load all the necessary components for the system """
        print(f"Welcome to the Reasoning And Learning System's Prototype!")
        # setting seed for 'local' random number generator
        self.rng = np.random.default_rng(random_state)
        # loading base ontology
        self.onto = get_ontology("file://" + os.path.join(KNOWLEDGE_FOLDER, onto_name)).load()
        # setting the python learning method for the system
        if rules_inductor_type == "sklearn":
            self.rules_inductor = RulesInductorSklearn(self.onto)
        elif rules_inductor_type == "chefboost":
            self.rules_inductor = RulesInductorChefboost(self.onto)
        else:
            raise ValueError(f"Unrecognized Rules Inductor's name: '{rules_inductor_type}'; "
                             f"Accepted values: ['sklearn', 'chefboost']")
        # setting the reasoner for the system
        if reasoner_type == "pellet":
            self.sync_reasoner = sync_reasoner_pellet
        elif reasoner_type == "hermit":
            self.sync_reasoner = sync_reasoner_hermit
        else:
            raise ValueError(f"Unrecognized reasoner's name: '{reasoner_type}'; "
                             f"Accepted values: ['pellet', 'hermit']")
        # saving the information about previous situations and their actions to the backup memory
        for s in self.onto.Situation.instances():
            self.backup_memory[s.name] = s.takenAction[0]
        # system prints out the names of its major components
        print(f"Base Ontology: {self.onto.base_iri};\n"
              f"Reasoning engine: {reasoner_type.capitalize()};\n"
              f"Learning method for python: {rules_inductor_type.capitalize()}.")
        input("Press 'Enter' to proceed with the analysis...")
        print()

    def system_shutdown(self):
        """ System's behaviour during its shutdown """
        # printing out the final values of the statistics
        print(f"Statistics for the analysis:\n"
              f"Users' satisfaction: {self.statistics['satisfaction_growth'][-1] / (len(self.statistics['satisfaction_growth']) - 1)}\n"
              f"Times reasoning process was executed: {self.statistics['reasoning_count_growth'][-1]}\n"
              f"Times learning process was executed: {self.statistics['learning_count_growth'][-1]}\n"
              f"Times system asked users to pick an option themselves: {self.statistics['asking_count_growth'][-1]}\n"
              f"Average execution time: {sum(self.statistics['exec_time']) / (len(self.statistics['exec_time']) - 1)}\n"
              f"Number of rules after the last learning process: {self.statistics['rules_num'][-1]}\n"
              f"Average length of rules' bodies after the last learning process: {self.statistics['average_rule_body_length'][-1]}")
        # saving results (measured statistics) to .json file
        stat_file_dir = os.path.join(RESULTS_FOLDER, DEFAULT_RESULTS_FILE)
        with open(stat_file_dir, 'w') as f:
            json.dump(self.statistics, f, indent=4)
        print("Shutting down the Reasoning And Learning System's Prototype! Thank you for your cooperation!...")

    def save_onto(self, save_name):
        """ Saving the modified knowledge (ontology) to the given file """
        save_path = os.path.join(KNOWLEDGE_FOLDER, save_name)
        self.onto.save(file=save_path)

    def execute_reasoning(self):
        """ Reasoning procedure """
        print("Reasoninng procedure initiated...")
        with self.onto:
            self.sync_reasoner(infer_property_values=True)
        print("Reasoning process complete...")
        print()

    def learn_new_rules(self):
        """ Learning procedure """
        print("Learning procedure initiated...")
        # resetting the previous rules
        if len(self.rules_inductor.inferred_rules_list) > 0:
            print("Resetting the rules set...")
            self.rules_inductor.reset()
        # setting 'main' results for the situations in case some of them got more than one action due to the reasoner
        for s in self.onto.Situation.instances():
            if len(s.takenAction) > 1:
                s.takenAction = [self.backup_memory[s.name]]
        print("Parsing the history knowledge to dataframe...")
        self.rules_inductor.get_dataset()
        print("Training the new model...")
        self.rules_inductor.train_model()
        print("Establishing a new set of inference rules...")
        self.rules_inductor.parse_to_swrl()
        # saving statistics for the current learning process
        rules_num, avg_rule_body_len = self.rules_inductor.get_rules_info()
        self.statistics["rules_num"].append(rules_num)
        self.statistics["average_rule_body_length"].append(avg_rule_body_len)
        print("Learning process complete...")
        print()

# An implementation below was created solely for the special experiments and it is not considered as an actual part of the system
#=======================================================================================================================
    # def learn_new_rules(self):
    #     """ Alternative take on the learning procedure """
    #     print("Learning procedure initiated...")
    #     # setting 'main' results for the situations in case some of them got more than one action due to the reasoner
    #     for s in self.onto.Situation.instances():
    #         if len(s.takenAction) > 1:
    #             s.takenAction = [self.backup_memory[s.name]]
    #     print("Parsing the history knowledge to dataframe...")
    #     self.rules_inductor.get_dataset()
    #     print("Training the new model...")
    #     self.rules_inductor.train_model()
    #     # saving statistics for the current learning process
    #     with open(os.path.join(KNOWLEDGE_FOLDER, self.rules_inductor.wanted_rules_file + ".json")) as f:
    #         rules_json = json.load(f)
    #     max_depth = 0
    #     leaves_num = 0
    #     for r in rules_json:
    #         if r["return_statement"] == 1 or r["feature_name"] == ""::
    #             max_depth = max(max_depth, r["current_level"])
    #             leaves_num += 1
    #     self.statistics["rules_num"].append(leaves_num)
    #     self.statistics["average_rule_body_length"].append(max_depth)
    #     print("Learning process complete...")
    #     print()
#=======================================================================================================================

    def ask_user_directly(self):
        """ 'Ask the user' procedure """
        print("Activating the 'Ask for an answer from the user' procedure...\nPossible options:")
        # getting the sorted list of possible interaction options
        action_options = sorted(self.onto.Interaction_with_user.instances(), key=lambda k: k.name)
        action_options = list(dict.fromkeys(action_options))
        # listing out the options
        for i, a in enumerate(action_options):
            print(f"{i} - {a.name}")
        # awaiting an user to make a choice
        while True:
            choice_id = input("Choose one of the above that suits you (put an appropriate ID): ")
            # making sure that input is valid
            try:
                user_choice = action_options[int(choice_id)]
                break
            except (ValueError, IndexError):
                print("WARNING: Invalid input detected! Please try again!")
        print(f"Answer affirmative...")
        print()
        return user_choice

# An implementation below was created solely for the special experiments and it is not considered as an actual part of the system
#=======================================================================================================================
    # def ask_user_directly(self, wanted_interaction):
    #     """ 'Ask the user' procedure - alternative take """
    #     print("Activating the 'Ask for an answer from the user' procedure...")
    #     user_choice = self.onto.Staying_quiet  # default value, just in case
    #     # looking for an interaction adequate to the current observation
    #     for i in self.onto.Interaction_with_user.instances():
    #         if i.name == wanted_interaction:
    #             user_choice = i
    #             break
    #     return user_choice
#=======================================================================================================================

    def classify_new_situation(self, sit_dict):
        """ Actual classification process, taking one observation at a time """
        # preparing statistics for the current observation's classification
        reward = 1.0
        reasoning_count = 0
        learning_count = 0
        asking_count = 0
        exec_time = 0.0
        # getting new id for the new observation
        last_sit_id = max(self.onto.Situation.instances(), key=lambda k: int(k.name[1:])).name
        new_sit_id = "s" + str(int(last_sit_id[1:]) + 1)
        new_sit = self.onto.Situation(new_sit_id, hadUser=sit_dict['User'], hadMood=sit_dict['General_mood'],
                                      wasWeather=sit_dict['Weather'], wasTime=sit_dict['General_time'])
        print(f"Received a new observation '{new_sit.name}':\nUser - {new_sit.hadUser.name}, "
              f"Mood - {new_sit.hadMood.name}, Weather - {new_sit.wasWeather.name}, Time - {new_sit.wasTime.name}")
        input("Press 'Enter' to initiate the classification process...")
        # classification process
        classif_time = time()
        self.execute_reasoning()  # first reasoning
        reasoning_count += 1
        # side variables controlling the procedures
        learning_done = False
        was_asked = False
        # if reasoning fails
        while len(new_sit.takenAction) == 0:
            print("System could not assign any action for the current situation...")
            # 'ask the user' procedure ("last resort")
            if learning_done or self.rng.random() <= self.ask_rate:
                classif_time = time() - classif_time
                exec_time += classif_time
                chosen_action = self.ask_user_directly()
                was_asked = True
                asking_count += 1
                reward -= 0.66  # punishment for the need to ask an user
                classif_time = time()
                new_sit.takenAction.append(chosen_action)
            # learning procedure (most desirable)
            else:
                self.learn_new_rules()
                learning_count += 1
                learning_done = True
                self.execute_reasoning()  # second reasoning
                reasoning_count += 1
        # checking the results of the classification (multiple results possible)
        new_sit.takenAction = sorted(new_sit.takenAction, key=lambda k: k.name)
        # removing potential duplicates
        new_sit.takenAction = list(dict.fromkeys(new_sit.takenAction))
        actions_num = len(new_sit.takenAction)
        classif_time = time() - classif_time
        exec_time += classif_time
        # checking the correctness of the results
        while True:
            # picking one of the gained results
            pick = self.rng.choice(new_sit.takenAction)
            print("Prompt:", self.action_roleplay(pick.name))
            # asking for affirmation (input needs to be valid)
            while True:
                try:
                    satisfaction = input("Do you accept this result? Yes (Y) / No (N): ").upper()
                    if satisfaction not in ["Y", "N"]:
                        raise ValueError("WARNING: Invalid input detected! Please try again!")
                    break
                except ValueError as e:
                    print(e)
            if satisfaction == "N":
                print("Affirmative. Removing the result from the found possibilities...")
                if not was_asked:
                    reward -= 1 / actions_num  # punishment for the wrong result (if it was via learning or reasoning)
                new_sit.takenAction.remove(pick)
            else:
                break
            # asking the user for a definitive answer if none of the results was satisfactory
            if len(new_sit.takenAction) == 0:
                print("All the previous possibilities have been removed!")
                chosen_action = self.ask_user_directly()
                was_asked = True
                asking_count += 1
                new_sit.takenAction.append(chosen_action)
        print(f"Affirmative. Saving action '{pick.name}' for the situation '{new_sit.name}'...")
        new_sit.takenAction = [pick]
        self.backup_memory[new_sit.name] = pick
        # regular reset of the inference rules after the 10th situation checked
        if len(self.onto.Situation.instances()) % 10 == 0:
            refresh_time = time()
            print("Refreshing the inference rules after the history expansion...")
            self.learn_new_rules()
            learning_count += 1
            refresh_time = time() - refresh_time
            exec_time += refresh_time
        # saving statistics to the systems variable/dictionary
        for k, v in zip(["satisfaction_growth", "reasoning_count_growth", "learning_count_growth", "asking_count_growth", "exec_time"],
                        [reward, reasoning_count, learning_count, asking_count, exec_time]):
            if k != "exec_time":
                v = v + self.statistics[k][-1]
            self.statistics[k].append(v)
        print()

# An implementation below was created solely for the special experiments and it is not considered as an actual part of the system
#=======================================================================================================================
    # def classify_new_situation(self, sit_dict, wanted_interaction):
    #     """ Alternative take on classification process, taking one observation at a time """
    #     # preparing statistics for the current observation's classification
    #     reward = 1.0
    #     asking_count = 0
    #     learning_count = 0
    #     exec_time = 0.0
    #     # getting new id for the new observation
    #     last_sit_id = max(self.onto.Situation.instances(), key=lambda k: int(k.name[1:])).name
    #     new_sit_id = "s" + str(int(last_sit_id[1:]) + 1)
    #     new_sit = self.onto.Situation(new_sit_id, hadUser=sit_dict['User'], hadMood=sit_dict['General_mood'],
    #                                   wasWeather=sit_dict['Weather'], wasTime=sit_dict['General_time'])
    #     print(f"Received a new observation '{new_sit.name}':\nUser - {new_sit.hadUser.name}, "
    #           f"Mood - {new_sit.hadMood.name}, Weather - {new_sit.wasWeather.name}, Time - {new_sit.wasTime.name}")
    #     #input("Press 'Enter' to initiate the classification process...")
    #     # classification process
    #     classif_time = time()
    #     if self.rules_inductor.model is None:
    #         self.learn_new_rules()
    #     # getting the observation from parsed knowledge
    #     self.rules_inductor.get_dataset()
    #     dataset = pd.read_csv(os.path.join(KNOWLEDGE_FOLDER, "actions_taken.csv"), sep=';')
    #     obs = dataset.iloc[-1:]
    #     obs.drop(columns=['Id'], inplace=True)
    #     obs.dropna(axis=1, inplace=True)
    #     # initiating predicting process
    #     was_asked = False
    #     try:
    #         if isinstance(self.rules_inductor, RulesInductorSklearn):
    #             prediction = self.rules_inductor.model.predict(obs)[0]
    #         else:
    #             moduleName = os.path.join(KNOWLEDGE_FOLDER, "chefboost_rules")
    #             tree = chef.restoreTree(moduleName)
    #             prediction = tree.findDecision(obs.values.tolist()[0])
    #         # 'ask_user_directly' method is used solely to parse prediction to OWL
    #         chosen_action = self.ask_user_directly(prediction)
    #     # if prediction fails
    #     except ValueError:
    #         print("Learning process failed!")
    #         chosen_action = self.ask_user_directly(wanted_interaction)
    #         was_asked = True
    #         asking_count += 1
    #         reward -= 0.66  # punishment for the need to ask an user
    #     else:
    #         learning_count += 1
    #     new_sit.takenAction.append(chosen_action)
    #     # checking the result of the classification
    #     actions_num = len(new_sit.takenAction)
    #     classif_time = time() - classif_time
    #     exec_time += classif_time
    #     # checking the correctness of the results
    #     while True:
    #         # picking one of the gained results
    #         pick = self.rng.choice(new_sit.takenAction)
    #         print("Prompt:", self.action_roleplay(pick.name))
    #         pick_syn = [s.name for s in pick.INDIRECT_equivalent_to]
    #         if pick.name == wanted_interaction or wanted_interaction in pick_syn:
    #             satisfaction = "Y"
    #         else:
    #             satisfaction = "N"
    #         if satisfaction == "N":
    #             print("Affirmative. Removing the result from the found possibilities...")
    #             if not was_asked:
    #                 reward -= 1 / actions_num  # punishment for the wrong result (if it was via learning or reasoning)
    #             new_sit.takenAction.remove(pick)
    #         else:
    #             break
    #         # asking the user for a definitive answer if none of the results was satisfactory
    #         if len(new_sit.takenAction) == 0:
    #             print("All the previous possibilities have been removed!")
    #             chosen_action = self.ask_user_directly(wanted_interaction)
    #             was_asked = True
    #             asking_count += 1
    #             new_sit.takenAction.append(chosen_action)
    #     print(f"Affirmative. Saving action '{pick.name}' for the situation '{new_sit.name}'...")
    #     new_sit.takenAction = [pick]
    #     self.backup_memory[new_sit.name] = pick
    #     # regular reset of the inference rules after the 10th situation checked
    #     if len(self.onto.Situation.instances()) % 5 == 0:
    #         refresh_time = time()
    #         print("Refreshing the inference rules after the history expansion...")
    #         self.learn_new_rules()
    #         refresh_time = time() - refresh_time
    #         exec_time += refresh_time
    #     # saving statistics to the systems variable/dictionary
    #     for k, v in zip(["satisfaction_growth", "learning_count_growth", "asking_count_growth", "exec_time"],
    #                     [reward, learning_count, asking_count, exec_time]):
    #         if k != "exec_time":
    #             v = v + self.statistics[k][-1]
    #         self.statistics[k].append(v)
    #     print()
#=======================================================================================================================
