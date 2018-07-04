import json
import pandas as pd

def to_csv():
    with open('results/retrofitting_results.json', 'r') as f:
        results_dict = json.load(f)

    d = {}
    for key in results_dict.keys():
        d[key] = {}
        # d[key]['lexicon'] = results_dict[key]['lexicon']
        if 'evaluation' in results_dict[key].keys():
            for key2 in results_dict[key]['evaluation'].keys():
                d[key][key2] = results_dict[key]['evaluation'][key2]
        if 'accuracies' in results_dict[key].keys():
            for key2 in results_dict[key]['accuracies'].keys():
                d[key][key2] = results_dict[key]['accuracies'][key2]
        if 'losses' in results_dict[key].keys():
            for key2 in results_dict[key]['losses'].keys():
                d[key][key2] = results_dict[key]['losses'][key2]

    df = pd.DataFrame(d)
    df = df.round(3)
    print(df)
    pd.DataFrame.to_csv(df, "results/retrofitting_results.csv")

if __name__ == '__main__':
    to_csv()