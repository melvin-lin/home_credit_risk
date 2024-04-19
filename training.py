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
import catboost as cb

from catboost import Pool
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from optuna.integration import CatBoostPruningCallback, LightGBMPruningCallback

import preprocessing as preprocessing
import featureselection


ROOT = Path("./home-credit-credit-risk-model-stability/parquet_files")
TRAIN = ROOT / "train"
TEST = ROOT / "test"

SCHEMAS = {
    "date_decision": pl.Date,
    r"^.*_\d.*[D]$": pl.Date,
    "case_id": pl.Int32,
    "WEEK_NUM": pl.Int32,
    "num_group1": pl.Int32,
    "num_group2": pl.Int32,
    r"^.*_\d.*[A|P]$": pl.Float64,
    r"^.*_\d.*[M]$": pl.String,
}

REGEXES = ["^.*_base.*$", "^.*[a-z]_0.*$", "^.*[a-z]_1.*$", "^.*[a-z]_2.*$"]

class SaveBestModel(xgb.callback.TrainingCallback):
    def __init__(self, cvboosters):
        self._cvboosters = cvboosters

    def after_training(self, model):
        self._cvboosters[:] = [cvpack.bst for cvpack in model.cvfolds]
        return model


def feature_engineering(
    X: pd.DataFrame, y: pd.DataFrame, num_features: int, plot: bool
) -> list:
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
            "booster": "gbtree",
            "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1, log=True),
            "lambda": trial.suggest_float("lambda", 1, 10),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "n_estimators": 600,
            "random_state": 42,
            "max_depth": trial.suggest_int("max_depth", 2, 10, step=2),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        }

        if cv:
            cvboosters = []
            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial, "test-auc"
            )
            dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
            bst = xgb.cv(
                param,
                dtrain,
                num_boost_round=param["n_estimators"],
                nfold=5,
                callbacks=[pruning_callback, SaveBestModel(cvboosters)],
            )
            trial.set_user_attr(key="best_booster", value=cvboosters)
            auc = bst["test-auc-mean"].values[-1]
        else:
            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial, "validation-auc"
            )
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
                callbacks=[pruning_callback],
            )
            trial.set_user_attr(key="best_booster", value=bst)
            preds = bst.predict(dvalid, iteration_range=(0, param["n_estimators"]))
            auc = roc_auc_score(valid_y, preds)
        return auc

    def get_best_booster(study, trial):
        if study.best_trial.number == trial.number:
            if isinstance(trial.user_attrs["best_booster"], list):
                study.set_user_attr(
                    key="best_booster",
                    value=trial.user_attrs["best_booster"][trial.number],
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


def perform_catboost(X: pd.DataFrame, y: pd.DataFrame, cv: bool):

    def objective(trial):
        param = {
            "objective": trial.suggest_categorical(
                "objective", ["Logloss", "CrossEntropy"]
            ),
            "colsample_bylevel": trial.suggest_float(
                "colsample_bylevel", 0.01, 1, log=True
            ),
            "depth": trial.suggest_int("depth", 2, 12, step=2),
            "boosting_type": trial.suggest_categorical(
                "boosting_type", ["Ordered", "Plain"]
            ),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
            "use_best_model": True,
            "n_estimators": 600,
            "random_state": 42,
            "eval_metric": "AUC",
        }

        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 0, 10
            )
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

        pruning_callback = CatBoostPruningCallback(trial, "AUC")  

        if cv:
            dtrain = Pool(data=X, label=y)
            cb_cv, bst = cb.cv(
                pool=dtrain,
                params=param,
                fold_count=5,
                inverted=True,
                early_stopping_rounds=100,
                return_models=True,
                stratified=True,
            )
            trial.set_user_attr(key="best_booster", value=bst)
            auc = cb_cv["test-AUC-mean"].values[-1]
        else:
            train_X, valid_X, train_y, valid_y = train_test_split(
                X, y, test_size=0.25, shuffle=True, random_state=42
            )
            dtrain = Pool(data=train_X, label=train_y)
            gbm = cb.CatBoostClassifier(**param)
            gbm.fit(
                train_X,
                train_y,
                eval_set=[(valid_X, valid_y)],
                early_stopping_rounds=100,
                callbacks=[pruning_callback],
            )
            pruning_callback.check_pruned()

            trial.set_user_attr(key="best_booster", value=gbm)
            preds = gbm.predict(valid_X, num_iterations=gbm.best_iteration_)
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


def perform_lgb(X: pd.DataFrame, y: pd.DataFrame):

    def objective(trial):
        param = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "device_type": "cpu",
            "colsample_bytree": 0.8,
            "colsample_bynode": 0.8,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "n_estimators": 600,
            "tree_learner": "voting",
        }

        pruning_callback = LightGBMPruningCallback(trial, "auc")

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
        choices=["lgb", "xgboost", "catboost"],
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
        help="Specifies the number of features that you want to keep to train with the model. ",
    )
    parser.add_argument(
        "--save_test",
        action="store_true",
        help="Specify whether you want to download the test data. ",
    )
    args = parser.parse_args()

    if not args.disable_preprocess:
        base, X, y = preprocessing.Preprocessing(
            TRAIN, REGEXES, SCHEMAS, mode="train"
        ).preprocessing(0.80)
        features = feature_engineering(
            X, y.to_numpy().ravel(), args.num_features, args.save_viz
        )
        try:
            os.mkdir(os.curdir + "/data")
            os.mkdir(os.curdir + "/data/test")
            os.mkdir(os.curdir + "/data/train")
        except FileExistsError:
            pass
        if args.save_test:
            test_base, test_X, _ = preprocessing.Preprocessing(
                TEST, REGEXES, SCHEMAS, mode="test"
            ).preprocessing(0.80)
            test_base.to_csv(os.curdir + "/data/test/base.csv")
            features = list(set(features).intersection(list(test_X.columns)))
            test_X[features].to_csv(os.curdir + "/data/test/X_test.csv")
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
            study = perform_lgb(X, y)
            bst = study.user_attrs["best_booster"]
        elif args.model == "xgboost":
            study = perform_xgboost(X, y, args.cv)
            bst = study.user_attrs["best_booster"]
        elif args.model == "catboost":
            study = perform_catboost(X, y, args.cv)
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
