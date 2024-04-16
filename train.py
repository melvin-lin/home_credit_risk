import re
import os
import sys
import optuna
import logging
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
import lightgbm as lgb

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import (
    StackingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)


class Train:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def objective(self, trial):
        classes = list(set(self.y))
        train_x, valid_x, train_y, valid_y = train_test_split(
            self.X.select(pl.exclude(["case_id", "WEEK_NUM"])),
            self.y,
            test_size=0.2,
            random_state=42,
        )

        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        clf = SGDClassifier(alpha=alpha)

        for step in range(100):
            clf.partial_fit(train_x, train_y, classes=classes)

            # Report intermediate objective value.
            intermediate_value = 1.0 - clf.score(valid_x, valid_y)
            trial.report(intermediate_value, step)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()

        return 1.0 - clf.score(valid_x, valid_y)
