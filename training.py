import re
import os
import sys
import time
import pickle
import joblib
import optuna
import logging
import argparse
import pandas as pd
import polars as pl
import xgboost as xgb
import lightgbm as lgb

from pathlib import Path
from sklearn.model_selection import (
    train_test_split,
)
from sklearn.metrics import roc_auc_score
from optuna.integration import LightGBMPruningCallback

import preprocessing as preprocessing
import featureselection


ROOT = Path("./home-credit-credit-risk-model-stability/parquet_files")
TRAIN = ROOT / "train"
TEST = ROOT / "test"

SCHEMAS = {
    "date_decision": pl.Date,
    "^.*_\d.*[D]$": pl.Date,
    "case_id": pl.Int32,
    "WEEK_NUM": pl.Int32,
    "num_group1": pl.Int32,
    "num_group2": pl.Int32,
    "^.*_\d.*[A|P]$": pl.Float64,
    "^.*_\d.*[M]$": pl.String,
}

REGEXES = ["^.*_base.*$", "^.*[a-z]_0.*$", "^.*[a-z]_1.*$", "^.*[a-z]_2.*$"]

class SaveBestModel(xgb.callback.TrainingCallback):
    def __init__(self, cvboosters):
        self._cvboosters = cvboosters
    
    def after_training(self, model):
        self._cvboosters[:] = [cvpack.bst for cvpack in model.cvfolds]
        return model


def feature_engineering(X: pd.DataFrame, y: pd.DataFrame, num_features: int, plot: bool) -> list:
    X = X.loc[:, ~X.columns.isin(["case_id", "WEEK_NUM"])]
    features = featureselection.FeatureSelection(X, y).get_information_gain_features(
        num_features=num_features, plot=plot
    )
    return features


def perform_xgboost(X: pd.DataFrame, y: pd.DataFrame, cv: bool):

    def objective(trial):
        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "n_estimators": 600,
            "random_state": 42,
        }

        if param["booster"] == "gbtree" or param["booster"] == "dart":
            param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

        if cv: 
            cvboosters = []
            pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-auc")
            dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
            bst = xgb.cv(param, dtrain, num_boost_round=param["n_estimators"], nfold=5, callbacks=[pruning_callback, SaveBestModel(cvboosters)])
            trial.set_user_attr(key="best_booster", value=cvboosters)
            auc = bst["test-auc-mean"].values[-1]
        else:
            pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
            train_X, valid_X, train_y, valid_y = train_test_split(
                X, y, test_size=0.25, shuffle=True, random_state=42
            )
            dtrain = xgb.DMatrix(train_X, label=train_y, enable_categorical=True)
            dvalid = xgb.DMatrix(valid_X, label=valid_y, enable_categorical=True)
            bst = xgb.train(
                params=param,
                dtrain=dtrain,
                num_boost_round=param["n_estimators"],
                evals=[(dtrain, "train"), (dvalid, "validation")],
                callbacks=[pruning_callback]
            )
            trial.set_user_attr(key="best_booster", value=bst)
            end_iteration = (
                bst.best_iteration + 1 if bst.best_iteration else param["n_estimators"]
            )
            preds = bst.predict(dvalid, iteration_range=(0, end_iteration))
            auc = roc_auc_score(valid_y, preds)
        return auc

    def get_best_booster(study, trial):
        if study.best_trial.number == trial.number:
            if isinstance(trial.user_attrs["best_booster"], list): 
                study.set_user_attr(
                    key="best_booster", value=trial.user_attrs["best_booster"][trial.number]
                )
            else: 
                study.set_user_attr(
                    key="best_booster", value=trial.user_attrs["best_booster"]
                )

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(pruner=pruner, direction="maximize")
    study.optimize(
        objective, n_trials=100, show_progress_bar=True, callbacks=[get_best_booster]
    )
    return study


def perform_lgb(X: pd.DataFrame, y: pd.DataFrame, cv: bool):

    def objective(trial):
        param = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "device_type": "cpu",
            "colsample_bytree": trial.suggest_float("colsample_bytree", 1e-8, 1, log=True),
            "colsample_bynode": trial.suggest_float("colsample_bynod", 1e-8, 1, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "n_estimators": 600,
            "extra_trees": True,
            "tree_learner": "voting",
        }  

        pruning_callback = LightGBMPruningCallback(trial, "auc")    

        if cv: 
            dtrain = lgb.Dataset(X, label=y)
            bst = lgb.cv(param, dtrain, callbacks=[pruning_callback], return_cvbooster=True)
            trial.set_user_attr(key="best_booster", value=bst["cvbooster"])
            auc = bst["valid auc-mean"][0]
        else:
            train_X, valid_X, train_y, valid_y = train_test_split(
                X, y, test_size=0.25, shuffle=True, random_state=42
            )
            dtrain = lgb.Dataset(train_X, label=train_y)
            dvalid = lgb.Dataset(valid_X, label=valid_y, reference=dtrain)
            bst = lgb.train(
                param,
                dtrain,
                valid_sets=dvalid,
                callbacks=[pruning_callback],
            )
            trial.set_user_attr(key="best_booster", value=bst)
            preds = bst.predict(valid_X, num_iterations=bst.best_iteration)
            auc = roc_auc_score(valid_y, preds)
        return auc

    def get_best_booster(study, trial):
        if study.best_trial.number == trial.number:
            study.set_user_attr(
                key="best_booster", value=trial.user_attrs["best_booster"]
            )

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(pruner=pruner, direction="maximize")
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study.optimize(
        objective, n_trials=100, show_progress_bar=True, callbacks=[get_best_booster]
    )
    return study


def save_model(model, model_type, time_str):
    model_pkl_file = os.curdir + f"/model/{model_type}_model_{time_str}.pkl"

    with open(model_pkl_file, "wb") as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Home Credit Risk Model Stability",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["lgb", "xgboost"],
        default="xgboost",
        help="Choose the model to train",
    )
    parser.add_argument(
        "--save_study",
        action="store_true",
        help="Save the study for the model selected",
    )
    parser.add_argument(
        "--disable_preprocess",
        action="store_true",
        help="Disables the preprocessing step",
    )
    parser.add_argument(
        "--disable_model",
        action="store_true",
        help="Disables the study for the model selected",
    )
    parser.add_argument(
        "--save_params",
        action="store_true",
        help="Outputs the best params from the study for the model selected",
    )
    parser.add_argument(
        "--cv",
        action="store_true", 
        help="Applies cross validation to the model selected",
    )
    parser.add_argument(
        "--save_viz", 
        action="store_true", 
        help="Save the visualizations retrieved from feature engineeering", 
    )
    parser.add_argument(
        "--num_features", 
        type=int, 
        default=None, 
        help="Specifies the number of features that you want to keep to train with the model. "
    )
    parser.add_argument(
        "--save_test", 
        action="store_true", 
        help="Specify whether you want to download the test data. "
    )
    args = parser.parse_args()

    if not args.disable_preprocess:
        base, X, y = preprocessing.Preprocessing(TRAIN, REGEXES, SCHEMAS).preprocessing(
            0.80
        )
        features = feature_engineering(X, y.to_numpy().ravel(), args.num_features, args.save_viz)
        if args.save_test: 
            test_base, test_X, test_y = preprocessing.Preprocessing(TRAIN, REGEXES, SCHEMAS).preprocessing(
                0.80
            )
            test_base.to_csv(os.curdir + "/data/test/base.csv")
            test_X[features].to_csv(os.curdir + "/data/test/X_test.csv")
            test_y.to_csv(os.curdir + "/data/test/y_test.csv")
        base.to_csv(os.curdir + "/data/train/base.csv")
        X[features].to_csv(os.curdir + "/data/train/X_train.csv")
        y.to_csv(os.curdir + "/data/train/y_train.csv")
    try:
        X = pd.read_csv(os.curdir + "/data/train/X_train.csv")
        y = pd.read_csv(os.curdir + "/data/train/y_train.csv")["target"]
        base = pd.read_csv(os.curdir + "/data/train/base.csv")
    except:
        raise FileNotFoundError

    study = None
    bst = None
    if not args.disable_model:
        groups = base["WEEK_NUM"]
        if args.model == "lgb":
            X = X.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
            study = perform_lgb(X, y, args.cv)
            bst = study.user_attrs["best_booster"]
        elif args.model == "xgboost":
            study = perform_xgboost(X, y, args.cv)
            bst = study.user_attrs["best_booster"]

    if study and bst:
        try: 
            os.mkdir(os.curdir + "/model")
            os.mkdir(os.curdir + "/trials")
            os.mkdir(os.curdir + "/study")
        except FileExistsError: 
            pass
        timestr = time.strftime("%Y%m%d-%H%M%S")
        save_model(bst, args.model, timestr)
        if args.save_params:
            study.trials_dataframe().to_csv(
                os.curdir + f"/trials/{args.model}_trials_{timestr}.csv"
            )
        if args.save_study:
            joblib.dump(study, os.curdir + f"/study/{args.model}_study_{timestr}.pkl")
