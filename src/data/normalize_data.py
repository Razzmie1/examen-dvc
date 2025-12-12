import pandas as pd
from sklearn.preprocessing import StandardScaler

X_train_df = pd.read_csv("data/processed/X_train.csv")
X_test_df = pd.read_csv("data/processed/X_test.csv")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df)
X_test_scaled = scaler.transform(X_test_df)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_df.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_df.columns)

X_train_scaled_df.to_csv("data/processed/X_train_scaled.csv", index=False)
X_test_scaled_df.to_csv("data/processed/X_test_scaled.csv", index=False)