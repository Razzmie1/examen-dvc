import pandas as pd 
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

X_train = pd.read_csv('data/processed/X_train_scaled.csv').to_numpy()
y_train = pd.read_csv('data/processed/y_train.csv').to_numpy().squeeze()

rf_regressor = RandomForestRegressor(n_jobs=-1)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20],
}

grid_search = GridSearchCV(
    estimator=rf_regressor,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)

with open('models/best_params.pkl', 'wb') as f:
    pickle.dump(grid_search.best_params_, f)
