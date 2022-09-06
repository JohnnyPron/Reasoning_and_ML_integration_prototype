from integration.rules_induction import RulesInductorChefboost, RulesInductorSklearn
from owlready2 import *
import os
import pandas as pd
from time import time
import json
from chefboost import Chefboost as chef
from sklearn.model_selection import train_test_split


ONTO_NAME = "jp_masters_project.owl"
KNOWLEDGE_FOLDER = "knowledge_folder"
RULES_INDUCTOR_TYPE = "chefboost"


if __name__ == "__main__":
    total_time = time()
    # loading an ontology (it is needed as an argument)
    onto = get_ontology("file://" + os.path.join(KNOWLEDGE_FOLDER, ONTO_NAME)).load()
    # loading final dataset and splitting it
    dataset = pd.read_csv(os.path.join(KNOWLEDGE_FOLDER, "actions_test.csv"), sep=';')
    dataset.drop(columns=['Id'], inplace=True)
    X = dataset.drop(columns=["takenAction"])
    y = dataset["takenAction"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
    X_train["takenAction"] = y_train
    # choosing the rule inductor
    if RULES_INDUCTOR_TYPE == "sklearn":
        rules_inductor = RulesInductorSklearn(onto)
    else:
        rules_inductor = RulesInductorChefboost(onto)
    # setting dataset to the rule inductor
    rules_inductor.dataset = X_train
    # training the model
    rules_inductor.train_model()
    # model evaluation
    if RULES_INDUCTOR_TYPE == "sklearn":
        evaluation = rules_inductor.model.score(X_test, y_test)
    else:
        X_test["takenAction"] = y_test
        evaluation = chef.evaluate(rules_inductor.model, X_test, target_label="takenAction", task="test")
    total_time = time() - total_time
    # getting information about tree's depth and number of leaves
    with open(os.path.join(KNOWLEDGE_FOLDER, rules_inductor.wanted_rules_file + ".json")) as f:
        rules_json = json.load(f)
    max_depth = 0
    leaves_num = 0
    for r in rules_json:
        if r["return_statement"] == 1 or r["feature_name"] == "":
            max_depth = max(max_depth, r["current_level"])
            leaves_num += 1
    # printing iut statistics
    print("Mean accuracy:", evaluation)
    print("Execution time:", total_time)
    print("Max tree depth:", max_depth)
    print("Number of leaves", leaves_num)
