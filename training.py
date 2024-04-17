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
    GroupKFold,
    StratifiedGroupKFold,
)
from sklearn.metrics import roc_auc_score

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


def feature_engineering(X: pd.DataFrame, y: pd.DataFrame, plot: bool) -> list:
    X = X.loc[:, ~X.columns.isin(["case_id", "WEEK_NUM"])]
    features = featureselection.FeatureSelection(X, y).get_information_gain_features(
        plot=plot
    )
    return features


def perform_xgboost(X: pd.DataFrame, y: pd.DataFrame, groups: pd.DataFrame, kfold: str):

    def objective(trial):
        param = {
            "verbosity": 0,
            "device": "cuda",
            "objective": "binary:logistic",
            "tree_method": "auto",
            "booster": trial.suggest_categorical("booster", ["gbtree"]),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "eval_metric": "auc",
            "n_estimators": 600,
            "random_state": 42,
            "max_depth": trial.suggest_int("max_depth", 3, 9, step=2),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
            "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            ),
        }

        if kfold:
            cv_scores = []
            skf = None
            if kfold == "stratify":
                skf = StratifiedGroupKFold(n_splits=5, shuffle=False)
            elif kfold == "group":
                skf = GroupKFold(n_splits=5)
            if skf:    
                for idx_train, idx_valid in skf.split(X, y, groups=groups):
                    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
                    X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]

                    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
                    dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
                    watchlist = [(dtrain, "train"), (dvalid, "valid")]

                    model = xgb.train(
                        param,
                        dtrain,
                        num_boost_round=param["n_estimators"],
                        evals=watchlist,
                    )
                    trial.set_user_attr(key="best_booster", value=model)
                    y_pred_valid = model.predict(
                        dvalid, iteration_range=(0, param["n_estimators"])
                    )
                    auc_score = roc_auc_score(y_valid, y_pred_valid)
                    cv_scores.append(auc_score)
                return max(cv_scores)
        else:
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
                early_stopping_rounds=25,
                verbose_eval=100,
            )
            trial.set_user_attr(key="best_booster", value=bst)
            end_iteration = (
                bst.best_iteration + 1 if bst.best_iteration else param["n_estimators"]
            )
            preds = bst.predict(dvalid, iteration_range=(0, end_iteration))
            roc = roc_auc_score(valid_y, preds)
        return roc

    def get_best_booster(study, trial):
        if study.best_trial.number == trial.number:
            study.set_user_attr(
                key="best_booster", value=trial.user_attrs["best_booster"]
            )

    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective, n_trials=50, show_progress_bar=True, callbacks=[get_best_booster]
    )
    return study


def perform_lgb(X: pd.DataFrame, y: pd.DataFrame, groups: pd.DataFrame, kfold: str):

    def objective(trial):
        param = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "rf"]),
            "device_type": "cpu",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 512),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
            "n_estimators": 600,
            "random_state": 42,
        }

        if kfold:
            cv_scores = []
            skf = None
            if kfold == "stratify":
                skf = StratifiedGroupKFold(n_splits=5, shuffle=False)
            elif kfold == "group":
                skf = GroupKFold(n_splits=5)

            if skf: 
                for idx_train, idx_valid in skf.split(X, y, groups=groups):
                    train_X, train_y = X.iloc[idx_train], y.iloc[idx_train]
                    valid_X, valid_y = X.iloc[idx_valid], y.iloc[idx_valid]

                    dtrain = lgb.Dataset(train_X, label=train_y)
                    dvalid = lgb.Dataset(valid_X, label=valid_y, reference=dtrain)

                    model = lgb.train(
                        param,
                        dtrain,
                        num_boost_round=param["n_estimators"],
                        valid_sets=dvalid,
                    )
                    trial.set_user_attr(key="best_booster", value=model)

                    y_pred_valid = model.predict(
                        valid_X, iteration_range=(0, param["n_estimators"])
                    )
                    auc_score = roc_auc_score(valid_y, y_pred_valid)
                    cv_scores.append(auc_score)

                return max(cv_scores)
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
                callbacks=[lgb.log_evaluation(100), lgb.early_stopping(25)],
            )
            trial.set_user_attr(key="best_booster", value=bst)
            preds = bst.predict(valid_X, num_iterations=bst.best_iteration)
            roc = roc_auc_score(valid_y, preds)
        return roc

    def get_best_booster(study, trial):
        if study.best_trial.number == trial.number:
            study.set_user_attr(
                key="best_booster", value=trial.user_attrs["best_booster"]
            )

    study = optuna.create_study(direction="maximize")
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study.optimize(
        objective, n_trials=50, show_progress_bar=True, callbacks=[get_best_booster]
    )
    return study


def save_model(model, model_type, time_str):
    model_pkl_file = os.curdir() + f"/model/{model_type}_model_{time_str}.pkl"

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
        "--kfold",
        choices=["stratify", "group"],
        default=None,
        help="Apply StratifiedKFold technique to the model selected",
    )
    args = parser.parse_args()

    if not args.disable_preprocess:
        base, X, y = preprocessing.Preprocessing(TRAIN, REGEXES, SCHEMAS).preprocessing(
            0.80
        )
        features = feature_engineering(X, y.to_numpy().ravel(), args.print_viz)
        breakpoint()
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
            study = perform_lgb(X, y, groups, args.kfold)
            bst = study.user_attrs["best_booster"]
        elif args.model == "xgboost":
            study = perform_xgboost(X, y, groups, args.kfold)
            bst = study.user_attrs["best_booster"]

    if study and bst:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        save_model(bst, args.model, timestr)
        if args.save_params:
            study.trials_dataframe().to_csv(
                os.curdir() + f"/trials/{args.model}_trials_{timestr}.csv"
            )
        if args.save_study:
            joblib.dump(study, os.curdir() + f"/study/{args.model}_study_{timestr}.pkl")
