from jpype import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool


def train(data):
    X = data.drop(columns=['los_freq'])
    Y = data['los_freq']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
    model = CatBoostRegressor()
    train_pool = Pool(X_train, y_train)
    test_pool = Pool(X_test, y_test)
    grid = {'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 10],
            'l2_leaf_reg': [1, 3, 5, 7, 9],
            }

    # The plot option can be used if training is performed in Jupyter notebook.
    grid_search_result = model.grid_search(grid, X=X_train, y=y_train, cv=3, plot=False)

    # choose parameters based on grid_search results
    model = CatBoostRegressor(
        iterations=1000,
        depth=6,
        learning_rate=0.1,
        l2_leaf_reg=3,
        loss_function='RMSE',
        eval_metric='R2',
        random_seed=1234
    )

    # The plot option can be used if training is performed in Jupyter notebook.
    model.fit(train_pool, eval_set=test_pool, plot=False)
    preds = model.predict(test_pool)
    print(preds)
    model.save_model("model_weights")
    return model