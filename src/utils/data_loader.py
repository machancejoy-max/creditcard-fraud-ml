import pandas as pd

def load_data(path: str = "data/creditcard.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df
