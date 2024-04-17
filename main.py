import re
import os
import sys
import optuna
import logging
import argparse
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
import lightgbm as lgb

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import (
    StackingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)

import preprocessing
import featureselection


ROOT = Path("./home-credit-credit-risk-model-stability/parquet_files")
TRAIN = ROOT / "train"
TEST = ROOT / "test"

SCHEMAS = {
    "date_decision": pl.Date, 
    "^.*_\d.*[D]$" : pl.Date, 
    "case_id" : pl.Int32,  
    "WEEK_NUM" : pl.Int32, 
    "num_group1" : pl.Int32, 
    "num_group2" : pl.Int32,
    "^.*_\d.*[A|P]$" : pl.Float64, 
    "^.*_\d.*[M]$" : pl.String, 
}

REGEXES = ["^.*_base.*$", "^.*[a-z]_0.*$", "^.*[a-z]_1.*$", "^.*[a-z]_2.*$"]

def feature_engineering(data: pd.DataFrame) -> list: 
    X = data.loc[:, ~data.columns.isin(['case_id', 'target'])]
    y = data["target"]
    features = ['case_id', 'WEEK_NUM'] + featureselection.FeatureSelection(X,y).get_information_gain_features(num_features=10)
    return features

def gini_stability(X, y, y_pred, w_fallingrate=88.0, w_resstd=-0.5):
    base = pd.DataFrame(
        {
            "WEEK_NUM": X["WEEK_NUM"], 
            "target": y, 
            "score": y_pred,
        }
    )

    gini_in_time = base.sort_values("WEEK_NUM").groupby("WEEK_NUM")[["target", "score"]].apply(lambda x: 2*roc_auc_score(x["target"], x["score"])-1).tolist()
    
    x = np.arange(len(gini_in_time))
    y = gini_in_time
    a, b = np.polyfit(x, y, 1)
    y_hat = a*x + b
    residuals = y - y_hat
    res_std = np.std(residuals)
    avg_gini = np.mean(gini_in_time)
    return avg_gini + w_fallingrate * min(0, a) + w_resstd * res_std

def perform_xgboost(X: pd.DataFrame, y: pd.DataFrame): 

    def objective(trial):
        train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.25)
        dtrain = xgb.DMatrix(train_X, label=train_y)
        dvalid = xgb.DMatrix(valid_X, label=valid_y)

        param = {
            "verbosity": 0,
            "device": "cuda", 
            "objective": "binary:logistic",
            "tree_method": "auto",
            "booster": "gbtree",
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "max_depth": trial.suggest_int("max_depth", 3, 9, step=2), 
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 10), 
            "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
            "gamma":  trial.suggest_float("gamma", 1e-8, 1.0, log=True), 
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]), 
            "eval_metric": "rmsle", 
        }

        bst = xgb.train(param, dtrain)
        preds = bst.predict(dvalid)
        pred_labels = np.rint(preds)
        roc = roc_auc_score(valid_y, pred_labels)
        stability = gini_stability(valid_X, valid_y, pred_labels)
        return roc, stability
    
    study = optuna.create_study(direction='maximize')
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study.optimize(objective, n_trials=10)
    return study

def perform_lgb(X: pd.DataFrame, y: pd.DataFrame):

    def objective(trial):
        train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.25)
        dtrain = lgb.Dataset(train_X, label=train_y)
        dvalid = lgb.Dataset(valid_X, label=valid_y)

        param = {
            "objective": "binary",
            "metric": "cross_entropy",
            "verbosity": -1,
            "boosting_type": "gbdt",  
            "device_type": "cuda",          
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'num_leaves': trial.suggest_int('num_leaves', 2, 512),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        }

        bst = lgb.train(param, dtrain)
        preds = bst.predict(dvalid)
        pred_labels = np.rint(preds)
        roc = roc_auc_score(valid_y, pred_labels)
        stability = gini_stability(valid_X, valid_y, pred_labels)
        return roc, stability
    
    study = optuna.create_study(direction='maximize')
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study.optimize(objective, n_trials=10)
    return study
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Home Credit Risk Model Stability', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', choices=['lgb', 'xgboost'], default='lgb', help='Choose the model to train')
    parser.add_argument('--print_viz', default="False", help='Print the visualization from model training/study')
    parser.add_argument('--disable_preprocess', default="False", help='Disables the preprocessing step')
    args = parser.parse_args()

    if not args.disable_preprocess: 
        train = preprocessing.Preprocessing(TRAIN, REGEXES, SCHEMAS).preprocessing(0.80)
        features = feature_engineering(train, args.print_viz)
        train.loc[:, train.columns.isin(features)].to_csv(os.curdir + "/data/train/X_train.csv")
        train["target"].to_csv(os.curdir + "/data/train/y_train.csv")

    try: 
        X = pd.read_csv(os.curdir + "/data/train/X_train.csv")
        y = pd.read_csv(os.curdir + "/data/train/y_train.csv")
    except: 
        raise FileNotFoundError
    
    if args.model == 'lgb':
            study = perform_lgb(X,y)
    elif args.model == 'xgboost': 
        study = perform_xgboost(X,y)
    if args.print_viz: 
        optuna.visualization.plot_optimization_history(study).savefig(f"{args.model}_optimization_history.png")
        optuna.visualization.plot_slice(study).savefig(f"{args.model}_slice.png")
        optuna.visualization.plot_param_importances(study).savefig(f"{args.model}_param_importances.png")
        optuna.visualization.plot_plot_terminator_improvement(study).savefig(f"{args.model}_terminator_improvement.png")
        
