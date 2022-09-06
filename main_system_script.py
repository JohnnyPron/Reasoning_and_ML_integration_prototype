from integration.system_core import ReasoningAndLearningSystemPrototype
from other_functions.situations_generator import random_situations_generator
from other_functions.results_visualizer import *
import pandas as pd
import os


ONTO_NAME = "jp_masters_project.owl"
ONTO_SAVE = "jp_masters_final.owl"
KNOWLEDGE_FOLDER = "knowledge_folder"
RULES_INDUCTOR_TYPE = "chefboost"
REASONER_TYPE = "pellet"
SYSTEM_RANDOM_STATE = 100
GENERATOR_RANDOM_STATE = 100
SITUATIONS_NUMBER = 50


if __name__ == "__main__":
    # initiating the system
    ral_sys = ReasoningAndLearningSystemPrototype()
    # loading the components
    ral_sys.load_components(ONTO_NAME, rules_inductor_type=RULES_INDUCTOR_TYPE, reasoner_type=REASONER_TYPE,
                            random_state=SYSTEM_RANDOM_STATE)
    # generating a data sample
    random_samples = random_situations_generator(ral_sys.onto, random_state=GENERATOR_RANDOM_STATE)
    # getting wanted interactions from the 'test dataset'
    # test_set = pd.read_csv(os.path.join(KNOWLEDGE_FOLDER, "actions_test.csv"), sep=';')
    # test_set = test_set[20:]["takenAction"]
    # i = 20  # observation index
    # classification of the generated observations
    for _ in range(SITUATIONS_NUMBER):
        new_sit = next(random_samples)
        # wanted_interaction = test_set[i]
        ral_sys.classify_new_situation(new_sit)
        # i += 1
    # system shutdown (saving the statistics to .json)
    ral_sys.system_shutdown()
    # visualising the results
    visualise_stats("satisfaction_growth", "satisfaction_growth.png",
                    "Users' absolute satisfaction growth", "Users' satisfaction")
    visualise_stats("exec_time", "execution_time.png",
                    "Classification's execution time change", "Execution time")
    visualise_protocols_count_growth()
    visualise_rules_info_growth()
    # saving ontology to another file
    ral_sys.save_onto(ONTO_SAVE)
