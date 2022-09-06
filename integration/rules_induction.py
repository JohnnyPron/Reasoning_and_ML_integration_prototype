from .data_parser import OntologyDataParser
import json
import re
import shutil
import os
from owlready2 import *
from chefboost import Chefboost as chef
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, export_text


KNOWLEDGE_FOLDER = "knowledge_folder"
DEFAULT_DATASET = "actions_taken.csv"


class RulesInductor:
    """ Parent class for the child inductors; the purpose of the 'Rules Inductor' is to train machine learning
     models based on the historical data from the ontology and parse them into inference rules for the ontology """
    wanted_rules_file = None  # base name of the file in which model's conditional expressions will be saved

    def __init__(self, onto):
        self.onto = onto  # the ontology with important data
        self.dataset = None  # dataframe, parsed from the ontology
        self.model = None  # trained machine learning model
        self.inferred_rules_list = list()  # list of inferred rule, parsed from conditional expressions

    def reset(self):
        """ This function's goal is to reset/clear all the class' parameters, except for the ontology """
        self.dataset = None
        self.model = None
        # removing rules from the ontology one at a time
        with self.onto:
            for r in self.inferred_rules_list:
                destroy_entity(r)
        self.inferred_rules_list = []

    def get_rules_info(self):
        """ An additional function which is supposed to return analysis statistics about inferred rules """
        rules_num = len(self.inferred_rules_list)  # number of the inferred rules
        # counting the average length of the rules' bodies
        body_length_sum = 0
        for r in self.inferred_rules_list:
            body_length_sum += len(r.body)
        return rules_num, body_length_sum / rules_num

    def get_dataset(self, dataset_csv=DEFAULT_DATASET):
        """ Using 'Data Parser' to get the dataset form the ontology """
        onto_parser = OntologyDataParser(self.onto)
        onto_parser.parse_situations_to_df()
        # saving dataset into csv file just to be safe
        onto_parser.save_dataframe(dataset_csv)
        self.dataset = onto_parser.get_dataframe()
        # removing an 'Id' column and potential records with missing values
        self.dataset.drop(columns=['Id'], inplace=True)
        self.dataset.dropna(inplace=True)

    def train_model(self):
        """ Base for the training process """
        # get synonyms of the instances
        synonyms = dict()
        for i in self.onto.Interaction_with_user.instances():
            if len(i.INDIRECT_equivalent_to) > 0:
                # parsing synonyms to string
                parsed_equivalents = [e.name for e in i.INDIRECT_equivalent_to]
                # checking whether or not this set of synonyms was already covered
                present = False
                for p in parsed_equivalents:
                    if p in synonyms:
                        present = True
                        break
                # saving synonyms to the dictionary
                if not present:
                    synonyms[i.name] = parsed_equivalents
        # parsing class labels in dataset
        for k, v in synonyms.items():
            self.dataset["takenAction"].replace(v, k, inplace=True)

    def parse_to_swrl(self):
        """ This function is supposed to parse conditional expressions from learned models to inference rules """
        # gender map for the inference rules (fitting to the ontology)
        gender_map = {"male": "false", "female": "true"}
        # reversre gender map for the inference rules
        reverse_gender_map = {"false": "male", "true": "female"}

        def eliminate_negations(current_body, head_value):
            """ Side function, used to transform negated predicates to positive equivalents """
            new_body = []  # replacement for the current body
            row_cond_dict = dict()  # conditions for row selection in dataset
            negations_dict = dict()  # dictionary for the negations (what the subsequent negations should be transformed into)
            # getting records from dataset, concerning the current head_value in check
            possible_records = self.dataset.loc[self.dataset['takenAction'] == head_value]
            # going over every predicate and preparing for the dataset check
            for b in current_body:
                # skipping the first element
                if b == "Situation(?s)":
                    new_body.append("Situation(?s)")
                    continue
                # splitting the "special" element to atoms
                if "User(?u), hadUser(?s, ?u), " in b:
                    for i in ["User(?u)", "hadUser(?s, ?u)"]:
                        new_body.append(i)
                # "hasAge" is treated separately since it doesn't always go together with above predicates
                if "hasAge(?u, ?a), " in b:
                    new_body.append("hasAge(?u, ?a)")
                # getting the "true form" of the current predicate (ignoring the "special" atoms)
                current_pred = re.search("~?\w+\(\?[asu], \w+\)", b)[0]
                new_body.append(current_pred)
                # looking for a possible negation to change
                if '~' in current_pred:
                    negations_dict[current_pred] = None
                # splitting predicate into feature name and value
                feat_name = re.search("~?\w+(?=\()", current_pred)[0]
                feat_val = re.search("\w+(?=\))", current_pred)[0]
                # parsing 'ontology's gender value' to 'dataframe's gender value'
                if feat_name == "hasGender":
                    feat_val = reverse_gender_map[feat_val]
                # establishing a new condition for the row search
                if feat_name in row_cond_dict:
                    row_cond_dict[feat_name].append(feat_val)
                else:
                    row_cond_dict[feat_name] = [feat_val]
            # looking for the records that meet found conditions
            for k, v in row_cond_dict.items():
                if '~' in k:
                    # k[1:], because we need to ignore tilda for the search
                    possible_records = possible_records.loc[~possible_records[k[1:]].isin(v)]
                else:
                    # special search for the 'hasAge' feature
                    if k == "lessThan":
                        v = list(map(int, v))
                        possible_records = possible_records.loc[possible_records["hasAge"] < min(v)]
                    elif k == "greaterThan":
                        v = list(map(int, v))
                        possible_records = possible_records.loc[possible_records["hasAge"] > max(v)]
                    else:
                        possible_records = possible_records.loc[possible_records[k].isin(v)]
            # eliminating duplicates
            possible_records.drop_duplicates(inplace=True)
            # creating rules fitting for the found records
            for index, row in possible_records.iterrows():
                # getting desired values from sampled record
                for n in negations_dict:
                    negated_feat = re.search("\w+(?=\()", n)[0]
                    desired_value = row[negated_feat]
                    # the condition below shouldn't be a necessity since gender value should have been properly
                    # parsed earlier but it's placed here just to be safe
                    if negated_feat == "hasGender":
                        desired_value = gender_map[desired_value]
                    desired_predicate = f"{negated_feat}(?s, {desired_value})"
                    # changes for the users' properties
                    if negated_feat in ["hasPersonality", "hasGender"]:
                        desired_predicate = desired_predicate.replace('?s', '?u')
                    negations_dict[n] = desired_predicate
                # making replacements in the new body
                new_body_copy = new_body.copy()
                for j, b in enumerate(new_body_copy):
                    if b in negations_dict:
                        new_body_copy[j] = negations_dict[b]
                # eliminating potential duplicates
                new_body_copy = list(dict.fromkeys(new_body_copy))
                # yielding created substitutes for the negated rules (returning a generator)
                yield ", ".join(new_body_copy)

        # get json file with the rules' elements
        json_file_dir = os.path.join(KNOWLEDGE_FOLDER, self.wanted_rules_file) + ".json"
        with open(json_file_dir, 'r') as f:
            facts_list = json.load(f)
        # list used in verification of potential rules' repeats
        temp_rules_list = list()
        # base for the body of an inference rule (starting with 'Situation(?s)')
        rule_body_list = ["Situation(?s)"]
        # iterating over  subsequent dictionaries in json file
        for i in facts_list:
            level = i["current_level"]
            # feature value is usually stored in the "rule" statement as the last element
            if i["feature_name"] != "hasAge":
                feature_value = re.search("\w+", i["rule"].split()[-1])[0]
            # special treatment for the "hasAge" property
            # we take float numbers into consideration to properly extract the needed information
            else:
                feature_value = int(float(re.findall("\d+\.?\d*", i["rule"])[-1]))
                if "<=" in i["rule"]:
                    feature_value += 1
            # if the element refers to the rule's body
            if i["return_statement"] == 0:
                feature_name = i["feature_name"]
                # mapping gender values to boolean values
                if feature_name == "hasGender":
                    feature_value = gender_map[feature_value]
                # setting a fact for the inference rule
                if feature_name != "hasAge":
                    fact = f"{feature_name}(?s, {feature_value})"
                    # taking negation into consideration (sklearn)
                    if i["rule"].split()[1] == "not":
                        fact = "~" + fact
                # special treatment for the "hasAge" property, again
                else:
                    additional_facts = f"{feature_name}(?u, ?a), "
                    if '<' in i["rule"]:
                        fact = f"lessThan(?a, {feature_value})"
                    else:
                        fact = f"greaterThan(?a, {feature_value})"
                    if additional_facts not in ", ".join(rule_body_list[:level]):
                        fact = additional_facts + fact
                # additional changes to the fact if the property is referring to the user specifically
                if feature_name in ["hasPersonality", "hasGender", "hasAge"]:
                    additional_facts = "User(?u), hadUser(?s, ?u), "
                    fact = fact.replace('?s', '?u')
                    if additional_facts not in ", ".join(rule_body_list[:level]):
                        fact = additional_facts + fact
                # modifying the rule's body after getting to the 'lower' level
                if level < len(rule_body_list):
                    rule_body_list = rule_body_list[:level]
                # adding another fact to the body
                rule_body_list.append(fact)
            # if we get to the head of the rule
            else:
                rule_head = f" -> takenAction(?s, {feature_value})"
                rule_body = ", ".join(rule_body_list)
                # replacing negations with proper values based on the information from dataset (sklearn)
                # due to the possibility of creating multiple rules after the elimination of negations,
                # all the current rules are being put in in the appropriate list
                if '~' in rule_body:
                    possible_bodies_list = eliminate_negations(rule_body_list, feature_value)
                else:
                    possible_bodies_list = [rule_body]
                for rb in possible_bodies_list:
                    # if rule's body contains information about User's properties and the specific User, all the
                    # predicates referring to the User's properties are skipped (rule puts an emphasis on the specific
                    # User);
                    if re.search("hadUser\(\?s, \?u\)", rb) is not None and \
                       re.search("hadUser\(\?s, \w+\)", rb) is not None:
                        for j in [", User\(\?u\)", ", hadUser\(\?s, \?u\)", ", hasPersonality\(\?u, \w+\)",
                                  ", hasGender\(\?u, \w+\)", ", hasAge\(\?u, \?a\)", ", lessThan\(\?a, \d+\)",
                                  ", greaterThan\(\?a, \d+\)"]:
                            rb = re.sub(j, "", rb)
                    rule_swrl = rb + rule_head
                    # setting a new rule to the ontology
                    if rule_swrl not in temp_rules_list:
                        temp_rules_list.append(rule_swrl)
                        with self.onto:
                            new_rule = Imp()
                            new_rule.set_as_rule(rule_swrl)
                            self.inferred_rules_list.append(new_rule)
        # printing inferred rules for the user to see
        print()
        for r in self.inferred_rules_list:
            print(str(r))


class RulesInductorChefboost(RulesInductor):
    """ The child inductor that is supposed to use chefboost module for machine learning """
    # directory in which chefboost saves the learning results (rules) by default
    chefboost_output_dir = os.path.join("outputs", "rules", "rules")
    # the base name for the files (.py and .json) that are supposed to contain learnt rules
    wanted_rules_file = "chefboost_rules"

    def _alter_json_rules(self, json_file_dir):
        """ Side function which makes some modifications to the json file with learnt rules """
        # specifying important information about rules that will be needed in the parsing process
        wanted_keys = ["current_level", "return_statement", "feature_name", "rule"]
        # base for the new json file
        new_json_list = []
        # opening the default file
        with open(json_file_dir, 'r') as f:
            json_list = json.load(f)
        # going over each rule element in the default file
        for j in json_list:
            # base for a new rule element
            new_json_dict = dict()
            # getting needed information from established keys
            for k in wanted_keys:
                new_json_dict[k] = j[k]
            # modification for the 'feature_name' kay if the element represents the return statement
            if new_json_dict["return_statement"] == 1:
                new_json_dict["feature_name"] = "takenAction"
            # omitting the "else" statements that do not fit any specific values
            if new_json_dict["feature_name"] != "":
                new_json_list.append(new_json_dict)
        # replacing "default" file with the new one
        os.remove(json_file_dir)
        with open(json_file_dir, 'w') as f:
            json.dump(new_json_list, f, indent=4)

    def train_model(self):
        """ Training process using chefboost """
        super().train_model()
        # setting the algorithm for the model
        config = {'algorithm': 'C4.5'}
        # training the model
        self.model = chef.fit(self.dataset, config=config, target_label='takenAction')
        # moving rules files (.py and .json) from default directory to the knowledge folder
        wanted_file_dir = os.path.join(KNOWLEDGE_FOLDER, self.wanted_rules_file)
        for sufix in [".py", ".json"]:
            full_file_dir = wanted_file_dir + sufix
            # deleting previous files form the knowledge folder
            if os.path.exists(full_file_dir):
                os.remove(full_file_dir)
            os.rename(self.chefboost_output_dir + sufix, full_file_dir)
        # removing the default directory
        shutil.rmtree('outputs')
        # making some modifications to the .json file
        json_file_dir = os.path.join(KNOWLEDGE_FOLDER, self.wanted_rules_file) + ".json"
        self._alter_json_rules(json_file_dir)


class RulesInductorSklearn(RulesInductor):
    """ The child inductor that is supposed to use scikit-learn module for machine learning """
    # the base name for the files (.txt and .json) that are supposed to contain learnt rules
    wanted_rules_file = "sklearn_rules"

    def _parse_rules_to_json(self, text_repr, json_file_dir):
        """ Private method that will help in parsing the tree's output into readable json file;
         Items needed for the rule dictionary:
         current_level, rule, feature_name, return_statement"""
        # splitting text from .txt file into separate string lines
        rules_list = text_repr.splitlines()
        # base for the .json file
        rules_json = []
        # going over each rule element in the .txt file
        for r in rules_list:
            # base for a new rule element
            new_rule_dict = dict()
            # setting the 'current level' for the current rule element
            new_rule_dict["current_level"] = len(re.findall("\|", r))
            new_rule_dict["return_statement"] = 0  # default value
            # getting the conditional expression from the element
            rule_atoms = re.search("\w.+$", r)[0].split()
            # settings for the return statements
            if rule_atoms[0] == "class:":
                new_rule_dict["return_statement"] = 1
                new_rule_dict["feature_name"] = "takenAction"
                new_rule_dict["rule"] = ' '.join(rule_atoms)
            # settings for the 'hasAge' predicate (keeping the mathematical comparison)
            elif rule_atoms[0] == "hasAge":
                new_rule_dict["feature_name"] = "hasAge"
                new_rule_dict["rule"] = "if " + ''.join(rule_atoms)
            # settings for everything else (object properties that are not 'takenAction')
            else:
                feature_name, feature_value = rule_atoms[0].split('_', 1)
                new_rule_dict["feature_name"] = feature_name
                # if the variable from OneHotEncoding is equal to '0'
                if rule_atoms[1] == "<=":
                    # replacing 'negated' genders with their opposites
                    if feature_name == "hasGender":
                        if feature_value == "male":
                            feature_value = "female"
                        else:
                            feature_value = "male"
                    # establishing the 'negated' conditional expression
                    else:
                        feature_name = "not " + feature_name
                new_rule_dict["rule"] = f"if {feature_name} == {feature_value}"
            rules_json.append(new_rule_dict)
        # saving .json file
        with open(json_file_dir + ".json", 'w') as f:
            json.dump(rules_json, f, indent=4)

    def train_model(self):
        """ Training process using scikit-learn """
        super().train_model()
        # splitting training dataset to X and y
        X = self.dataset.drop(columns=["takenAction"])
        y = self.dataset["takenAction"]
        # implementing an One Hot Encoder
        ohe = OneHotEncoder()
        # implementing the column transformer that will transform all categorical variables with the use of OHE
        col_trans = make_column_transformer(
            (ohe, ["hadUser", "hasPersonality", "hasGender", "hadMood", "wasWeather", "wasTime"]),
            remainder="passthrough"  # making sure that 'hasAge' won't be dropped
        )
        # setting the base classification tree with predetermined random state
        clf = DecisionTreeClassifier(random_state=0)
        # creating the model taht combines the column transformer with the classifier
        self.model = make_pipeline(col_trans, clf)
        # training the classifier
        self.model.fit(X, y)
        # getting feature names
        feat_names = self.model["columntransformer"].transformers_[0][1].get_feature_names_out()
        feat_names = list(feat_names) + ["hasAge"]
        # getting the conditional expressions form the learnt tree
        text_representation = export_text(clf, feature_names=list(feat_names), max_depth=100)
        # saving if...else rules to .txt file
        save_file_dir = os.path.join(KNOWLEDGE_FOLDER, self.wanted_rules_file)
        for sufix in [".txt", ".json"]:
            if os.path.exists(save_file_dir + sufix):
                os.remove(save_file_dir + sufix)
        with open(save_file_dir + '.txt', 'w', encoding='utf-8') as f:
            f.write(text_representation)
        # parsing if...else rules to json file
        self._parse_rules_to_json(text_representation, save_file_dir)
