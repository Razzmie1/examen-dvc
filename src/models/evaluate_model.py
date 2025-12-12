import pandas as pd 
import pickle
import json
from sklearn.metrics import mean_squared_error, r2_score

X_test = pd.read_csv('data/processed/X_test_scaled.csv').to_numpy()
y_test_df = pd.read_csv('data/processed/y_test.csv')
y_test = y_test_df.to_numpy().squeeze()

with open('models/trained_model.pkl', 'rb') as f:
    trained_model = pickle.load(f)

y_pred = trained_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics = {
    "mean_squared_error": mse,
    "r2_score": r2
}

with open('metrics/scores.json', 'w') as f:
    json.dump(metrics, f)

y_pred_df = pd.DataFrame(y_pred, columns=y_test_df.columns)
y_pred_df.to_csv('data/predictions.csv', index=False)
