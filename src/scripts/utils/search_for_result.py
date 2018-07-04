import os
import json

def search_results(rootdir, text = "random_uniform"):

    for subdir in os.walk(rootdir):
        if 'config.json' in subdir[-1]:
            json1_file = open(os.path.join(subdir[0], 'config.json'))
            json1_str = json1_file.read()
            json1_data = json.loads(json1_str)
            if json1_data['embedding']['name'] == text:
                print(subdir[0])

if __name__ == "__main__":
    search_results("./results")