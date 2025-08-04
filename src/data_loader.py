import pandas as pd

def load_train_data(path='data/train.csv'):
    df = pd.read_csv(path)
    return df
