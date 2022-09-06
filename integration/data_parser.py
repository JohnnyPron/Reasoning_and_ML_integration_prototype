from owlready2 import *
import os
import pandas as pd


KNOWLEDGE_FOLDER = "knowledge_folder"
DEFAULT_DATASET = "actions_taken.csv"


class OntologyDataParser:
    """ The purpose of this class is to parse the important information from ontology to more easily interpretable
    dataframe and later - the .csv file """
    def __init__(self, onto):
        self.onto = onto  # the ontology with important data
        self.dataframe = None  # dataframe, parsed from the ontology

    def get_ontology(self):
        """ get loaded ontology """
        return self.onto

    def get_dataframe(self):
        """ get dataframe, created from the parsed information """
        return self.dataframe

    def parse_situations_to_df(self):
        """ This method is supposed to get the information about systems' interactions with users that were previously
         registered in the main ontology and parse them to the interpretable dataframe """
        def check_properties(current_ind, current_obs):
            """ Method that is responsible for getting the values from the individual's consecutive properties """
            for p in current_ind.get_properties():
                for value in p[current_ind]:
                    # if current value is an integer or boolean, the raw value is added to the dataframe
                    if isinstance(value, int):
                        current_obs[p.python_name] = value
                        continue
                    # else - the value needs to be "parsed to string"
                    else:
                        current_obs[p.python_name] = value.name
                    # getting values from the User's individual specifically (recursion)
                    if self.onto.User in value.is_a:
                        check_properties(value, current_obs)

        # getting the instances of the previous situations, when the system interacted with different users
        sit_list = self.onto.Situation.instances()
        # preparing the base for the future dataframe
        actions_list = []
        # getting the information about circumstances from the properties of each situation
        for s in sit_list:
            # preparing a new row with an ID
            new_observ = {"Id": s.name}
            # getting needed information about the current situation
            check_properties(s, new_observ)
            # adding the "row" to the base
            actions_list.append(new_observ)
        # creating the dataframe
        self.dataframe = pd.DataFrame.from_records(actions_list)
        # parsing boolean values for users' genders
        self.dataframe["hasGender"].replace({False: "male", True: "female"}, inplace=True)
        # rearranging the column order for better readability and to keep consistency
        self.dataframe = self.dataframe[["Id", "hadUser", "hasPersonality", "hasGender", "hasAge", "hadMood",
                                         "wasWeather", "wasTime", "takenAction"]]

    def save_dataframe(self, csv_name=DEFAULT_DATASET):
        """ Saving dataframe to the csv file """
        df_path = os.path.join(KNOWLEDGE_FOLDER, csv_name)
        self.dataframe.to_csv(df_path, sep=';', index=False)
