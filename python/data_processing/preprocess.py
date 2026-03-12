
import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df = df.dropna()
    return df

if __name__ == "__main__":
    data = load_data("../../data/raw/agriculture_dataset.csv")
    data = preprocess(data)
    data.to_csv("../../data/processed/clean_dataset.csv", index=False)
