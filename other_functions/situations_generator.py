from owlready2 import *
import os
import numpy as np


def random_situations_generator(onto, random_state=None):
    """ An infinite random situations generator for the project's ontology """
    rng = np.random.default_rng(random_state)  # setting the random number generator
    # infinite generator
    while True:
        situation = dict()
        for v in [onto.User, onto.General_mood, onto.Weather, onto.General_time]:
            situation[v.name] = rng.choice(v.instances())
        yield situation


if __name__ == "__main__":
    my_onto_path = os.path.join("..", "knowledge_folder", "jp_masters_project.owl")
    onto = get_ontology("file://" + my_onto_path).load()
    sit_gen = random_situations_generator(onto, 100)
    for i in range(50):
        print(next(sit_gen))
