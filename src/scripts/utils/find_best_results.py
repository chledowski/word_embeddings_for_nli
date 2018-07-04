import pandas as pd
import os

def find_best_results(path, prepath='.'):
    df = pd.read_csv(os.path.join(prepath,path,'history.csv'))
    max_id = df['val_acc'].idxmax()
    print(df.loc[max_id,["acc", "val_acc", "test_acc"]])
    print("Epochs: ", len(df.index))

if __name__ == "__main__":
    find_best_results('2018_03_15_15211196254_cbow_cc42')