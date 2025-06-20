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
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import json
import pickle
import joblib

rpth = Path(os.path.abspath(__file__)).parent.parent/"src"
sys.path.append(str(rpth))
for  v in rpth.iterdir():
    if v.is_dir() and v.name != "__pycache__":
         sys.path.append(str(v))

import qstdb

def lgb_train_model(train_x, train_y, df_train, test_x, test_y, df_test, dev_x, dev_y, df_dev, model_params, target, checkpoint_name = None):
    lgb_train = lgb.Dataset(train_x, label = train_y, categorical_feature=model_params.get('categorical_feature', []))
    params = deepcopy(model_params)
    valid_data = lgb.Dataset(dev_x, label=dev_y, reference=lgb_train)  

    model = lgb.train(params, 
        lgb_train, 
        num_boost_round = 100,
        valid_sets=[valid_data],
        valid_names=['valid'],
        callbacks=[
                lgb.early_stopping(stopping_rounds = 20),
                lgb.log_evaluation(10)  # Print metrics every 10 rounds
            ]                      
    )

    df_test[f"{target}_log"] = df_test[target]
    df_test['predict_log'] = model.predict(test_x)
    df_test['predict'] = df_test['predict_log'].apply(lambda x: int(np.exp(x)))

    if checkpoint_name:
        # model.save_model(checkpoint_name, num_iteration=model.best_iteration)
        joblib.dump(model, checkpoint_name + ".pkl")

    return df_test

def fill_hole_lease(x):
    if x.remaining_lease:
        return x.remaining_lease
    else:
        return int(x.month.year) - int(x.lease_commence_date) + 1 if x.lease_commence_date else None
    
def floor_cat(floor_tag):
    x = list(map(lambda x: x.strip(), floor_tag.strip().split(" ")))
    if int(x[2]) <= 4:
        return "1" #"low"
    elif int(x[2]) <= 8:
        return "2" #middle
    else:
        return "3" #"high"
    
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
    y = y + y2
    return y 

def flat_type_norm(s):
    if s == "MULTI GENERATION":
        return "MULTI-GENERATION"
    else:
        return s

def train_eval(testid = "regres_lgb_20250620", checkpoint_path = "./checkpoint"):
    checkpoint_name = os.path.join(checkpoint_path, f"{testid}.txt")
    checkpoint_name_scale = os.path.join(checkpoint_path, f"{testid}.scale")

    sql_str = """SELECT * FROM hdb_resale_transactions;"""
    df = qstdb.query(sql_str)
    df["month"] = df["month"].apply(lambda x: datetime.strptime(x, "%Y-%m"))

    df["remaining_lease"] = df.apply(lambda x: fill_hole_lease(x), axis = 1)
    idx = "month"
    feat_cols = ["town", "flat_type", "storey_range", "floor_area_sqm", "remaining_lease"]
    target = "resale_price"
    df = df.sort_values(by = "month", ascending=True)[[idx] + feat_cols + [target]]
    # feature prepare
    df["storey_range"] = df["storey_range"].apply(lambda x: floor_cat(x)).astype("category")
    df["remaining_lease"] = df["remaining_lease"].apply(lambda x: rem_lease(x))
    df["flat_type"] = df["flat_type"].apply(lambda x: flat_type_norm(x))
    df.dropna(how = "any", inplace=True)
    df = df[df["remaining_lease"] > 0]
    df["remaining_lease"] = df["remaining_lease"].apply(np.log)

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
    df_train = df[df["month"].isin(split_dt["train"])]
    df_dev = df[df["month"].isin(split_dt["dev"])]
    df_test = df[df["month"].isin(split_dt["test"])]

    #flat type, twon_set, flat_set
    twon_set = dict(map(lambda x: (x[1], str(x[0] + 1)), enumerate(df_train["town"].unique())))
    df_train["town_tag"] = df_train["town"].apply(lambda x: twon_set.get(x, None)).astype("category")
    df_dev["town_tag"] = df_dev["town"].apply(lambda x: twon_set.get(x, None)).astype("category")
    df_test["town_tag"] = df_test["town"].apply(lambda x: twon_set.get(x, None)).astype("category")

    flat_set = dict(map(lambda x: (x[1], str(x[0] + 1)), enumerate(df_train["flat_type"].unique())))
    df_train["flat_type_tag"] = df_train["flat_type"].apply(lambda x: flat_set.get(x, None)).astype("category")
    df_dev["flat_type_tag"] = df_dev["flat_type"].apply(lambda x: flat_set.get(x, None)).astype("category")
    df_test["flat_type_tag"] = df_test["flat_type"].apply(lambda x: flat_set.get(x, None)).astype("category")

    df_train["floor_area_sqm"] = df_train["floor_area_sqm"].apply(np.log)
    df_dev["floor_area_sqm"] = df_dev["floor_area_sqm"].apply(np.log)
    df_test["floor_area_sqm"] = df_test["floor_area_sqm"].apply(np.log)

    df_train["resale_price_raw"] = df_train["resale_price"].copy()
    df_dev["resale_price_raw"] = df_dev["resale_price"].copy()
    df_test["resale_price_raw"] = df_test["resale_price"].copy()
    df_train["resale_price"] = df_train["resale_price"].apply(np.log)
    df_dev["resale_price"] = df_dev["resale_price"].apply(np.log)
    df_test["resale_price"] = df_test["resale_price"].apply(np.log)

    feat_meta = {
        "cat": ["storey_range", "town_tag", "flat_type_tag"],
        "num": ["floor_area_sqm", "remaining_lease"],
        "target": "resale_price",
    }
    #
    scaler = StandardScaler()
    train_x_num = df_train[feat_meta["num"]]
    train_x_scaled_num = scaler.fit_transform(train_x_num)
    train_y = df_train[feat_meta["target"]]
    
    dev_x_num = df_dev[feat_meta["num"]]
    dev_x_scaled_num = scaler.transform(dev_x_num)
    dev_y = df_dev[feat_meta["target"]]

    test_x_num = df_test[feat_meta["num"]]
    test_x_scaled_num = scaler.transform(test_x_num)
    test_y = df_test[feat_meta["target"]]

    # Train the model
    train_x_scaled_num = pd.DataFrame(train_x_scaled_num, columns=feat_meta["num"])
    dev_x_scaled_num = pd.DataFrame(dev_x_scaled_num, columns=feat_meta["num"])
    test_x_scaled_num = pd.DataFrame(test_x_scaled_num, columns=feat_meta["num"])

    train_x_scaled = pd.concat([train_x_scaled_num, df_train[feat_meta["cat"]].reset_index(drop=True)], axis=1)
    dev_x_scaled = pd.concat([dev_x_scaled_num, df_dev[feat_meta["cat"]].reset_index(drop=True)], axis=1)
    test_x_scaled = pd.concat([test_x_scaled_num, df_test[feat_meta["cat"]].reset_index(drop=True)], axis=1)
    
    model_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'categorical_feature': feat_meta["cat"],
            'seed': 20250620,
            'verbose': -1
    }
    result = lgb_train_model(train_x_scaled, train_y, df_train, test_x_scaled, test_y, df_test, dev_x_scaled, dev_y, df_dev, model_params, feat_meta["target"], checkpoint_name)

    #evaluation metric

    y_test = result["resale_price_raw"]
    y_pred = result["predict"]
    metric = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    #output save to db
    conf = f"""http::addr={os.getenv("QDB_HOST")}:{os.getenv("QDB_PORT")};username={os.getenv("QDB_USER")};password={os.getenv("QDB_PWD")};""" 

    hdb_resale_price_predict_tb = "hdb_resale_predict_price"   
    hdb_resale_price_experiment_meta_tb = "hdb_resale_predict_price_meta"   
    with Sender.from_conf(conf) as sender:
        result["testid"] = testid
        sender.dataframe(result, table_name = hdb_resale_price_predict_tb, at = TimestampNanos.now())
        sender.flush() 

        dd = deepcopy(model_params)
        dd["categorical_feature"] = json.dumps(dd["categorical_feature"])
        dd = dict(map(lambda x: (x[0], [x[1]]), dd.items()))
        dd["feature_meta"] = json.dumps(feat_meta)
        dd["evaluate_metric"] = json.dumps(metric)
        dd["testid"] = [testid]
        dd = pd.DataFrame(dd)
        sender.dataframe(dd, table_name = hdb_resale_price_experiment_meta_tb, at = TimestampNanos.now())
        sender.flush() 

    with open(checkpoint_name_scale, "wb") as fo:
        pickle.dump([scaler, feat_meta, twon_set, flat_set], fo) 

if __name__ == "__main__":
    train_eval()


