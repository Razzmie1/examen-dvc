import pandas as pd 
import pickle
from sklearn.ensemble import RandomForestRegressor

X_train = pd.read_csv('data/processed/X_train_scaled.csv').to_numpy()
y_train = pd.read_csv('data/processed/y_train.csv').to_numpy().squeeze()

with open('models/best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)

rf_regressor = RandomForestRegressor(**best_params, n_jobs=-1)
rf_regressor.fit(X_train, y_train)

with open('models/trained_model.pkl', 'wb') as f:
    pickle.dump(rf_regressor, f)