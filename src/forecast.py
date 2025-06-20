import lightgbm as lgb
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from copy import deepcopy
import pandas as pd
import os
from pathlib import Path
import sys
from questdb.ingress import Sender, TimestampNanos
import requests
import pandas as pd

rpth = Path(os.path.abspath(__file__)).parent.parent/"src"
sys.path.append(str(rpth))
for  v in rpth.iterdir():
    if v.is_dir() and v.name != "__pycache__":
         sys.path.append(str(v))

import qstdb

def lgb_train_model(train_x, train_y, train_df, test_x, test_df, model_params, target):
    lgb_train = lgb.Dataset(train_x, label = train_y, categorical_feature=model_params.get('categorical_feature', []))
    params = deepcopy(model_params)
    model = lgb.train(params, lgb_train, num_boost_round=100)

    test_df['prediction'] = model.predict(test_x)
    test_df['acutual'] = test_df["log_return"]

    result = test_df[['Date', 'Ticker', target, 'prediction', 'acutual']]

    return result

def features():
    sql_str = """SELECT * FROM hdb_resale_transcations;"""
    df = qstdb.query(sql_str)
    idx = "month"
    feat_cols = ["town", "flat_type", "storey_range", "floor_area_sqm", "flat_model", "remaining_lease"]
    target = "resale_price"
    df = df.sort_values(by = "month", ascending=True)[[idx] + feat_cols + [target]]
    # feature prepare
    
    #split
    r = [0.7, 0.1, 0.2]
    n_total = df.shape[0]
    dt = df["month"].tolist()
    split_dt = {
        "train": dt[:int(n_total*r[0])], 
        "dev": dt[int(n_total*r[0]):int(n_total*sum(r[:2]))], 
        "test": dt[int(n_total*sum(r[:2])):]
    }

    #Train set
    df_train = df[df["month"].isin(split_dt["train"])][feat_cols]
    df_dev = df[df["month"].isin(split_dt["dev"])][feat_cols]
    df_test = df[df["month"].isin(split_dt["test"])][feat_cols]


def train():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=10,
        random_state=42
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

    params = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [500, 1000, 1500]
    }

    grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred, squared=False))
print("R²:", r2_score(y_test, y_pred))

