import pandas as pd
from sklearn.model_selection import train_test_split

raw_df = pd.read_csv("data/raw_data/raw.csv")
raw_df.drop(columns=["date"], inplace=True)

X_set = raw_df.drop(columns=["silica_concentrate"])
y_set = raw_df["silica_concentrate"]

X_train, X_test, y_train, y_test = train_test_split(X_set, y_set)

X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)