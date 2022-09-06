import json
import os
import numpy as np
import matplotlib.pyplot as plt


RESULTS_FOLDER = "results"
DEFAULT_RESULTS_FILE = "analysis_results.json"


def visualise_stats(stat_name, save_file, my_title, my_ylabel):
    with open(os.path.join(RESULTS_FOLDER, DEFAULT_RESULTS_FILE), 'r') as f:
        results = json.load(f)
    x = list(range(0, len(results[stat_name])))
    plt.plot(x, results[stat_name])
    plt.title(my_title)
    plt.xlabel("Classified observations")
    plt.xticks(np.arange(0, len(x) + 1, 5))
    plt.ylabel(my_ylabel)
    plt.savefig(os.path.join(RESULTS_FOLDER, save_file))
    plt.clf()


def visualise_protocols_count_growth():
    with open(os.path.join(RESULTS_FOLDER, DEFAULT_RESULTS_FILE), 'r') as f:
        results = json.load(f)
    for r in ["reasoning_count_growth", "learning_count_growth", "asking_count_growth"]:
        x = list(range(0, len(results[r])))
        plt.plot(x, results[r])
    plt.title("System protocols' count growth")
    plt.xlabel("Classified observations")
    plt.ylabel("Protocol activation count")
    plt.xticks(np.arange(0, len(x) + 1, 5))
    plt.legend(["reasoning", "learning", "asking user"])
    plt.savefig(os.path.join(RESULTS_FOLDER, "protocols_count.png"))
    plt.clf()


def visualise_rules_info_growth():
    with open(os.path.join(RESULTS_FOLDER, DEFAULT_RESULTS_FILE), 'r') as f:
        results = json.load(f)
    for i, r in enumerate(zip(["rules_num", "average_rule_body_length"],
                              ["Number of learnt rules growth", "Avg. rule body length growth"],
                              ["Learnt rules count", "Average rule length"])):
        x = list(range(0, len(results[r[0]])))
        plt.subplot(1, 2, i + 1)
        plt.plot(x, results[r[0]])
        plt.title(r[1])
        plt.ylabel(r[2])
        plt.xlabel("Instances of learning's activation")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, "rules_info_count.png"))
    plt.clf()
