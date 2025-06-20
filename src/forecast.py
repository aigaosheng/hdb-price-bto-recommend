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
from datetime import datetime

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
    sql_str = """SELECT * FROM hdb_resale_transactions;"""
    df = qstdb.query(sql_str)
    df["month"] = df["month"].apply(lambda x: datetime.strptime(x, "%Y-%m"))

    def fill_hole_lease(x):
        if x.remaining_lease:
            return x.remaining_lease
        else:
            return int(x.month.year) - int(x.lease_commence_date) + 1 if x.lease_commence_date else None
    df["remaining_lease"] = df.apply(lambda x: fill_hole_lease(x), axis = 1)
    idx = "month"
    feat_cols = ["town", "flat_type", "storey_range", "floor_area_sqm", "remaining_lease"]
    target = "resale_price"
    df = df.sort_values(by = "month", ascending=True)[[idx] + feat_cols + [target]]
    # feature prepare
    def floor_cat(floor_tag):
        x = list(map(lambda x: x.strip(), floor_tag.strip().split(" ")))
        if int(x[2]) <= 4:
            return 1 #"low"
        elif int(x[2]) <= 8:
            return 2 #middle
        else:
            return 3 #"high"
    def rem_lease(s):
        try:
            x = list(map(lambda x: x.strip(), s.strip().split(" ")))
        except:
            try:
                x = int(s)
                return x
            except:
                return None
        try:
            y = int(x)
        except:
            y = 0
        try:
            y2 = int(2)
            y2 = round(y2 / 12, 2)
        except:
            y2 = 0
        return y + y2
    def flat_type(s):
        if s == "MULTI GENERATION":
            return "MULTI-GENERATION"
        else:
            return s
    df["storey_range"] = df["storey_range"].apply(lambda x: floor_cat(x))
    df["remaining_lease"] = df["remaining_lease"].apply(lambda x: rem_lease(x))
    df["flat_type"] = df["flat_type"].apply(lambda x: flat_type(x))
    df.dropna(how = "any", inplace=True)

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

    #flat type
    flat_set = dict(map(lambda x: (x[0], x[1]), enumerate(df_train["flat_type"].unique())))

    print(f"{df_train["flat_type"].unique()}, ** {df_train["storey_range"].unique()}, ** {df_train["flat_model"].unique()}")
    print(f"{df_dev["flat_type"].unique()}, ** {df_dev["storey_range"].unique()}, ** {df_dev["flat_model"].unique()}")
    print(f"{df_test["flat_type"].unique()}, ** {df_test["storey_range"].unique()}, ** {df_test["flat_model"].unique()}")

    return {"train": df_train, "dev": df_dev, "test": df_test}

features()

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

