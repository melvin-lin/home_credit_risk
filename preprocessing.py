import numpy as np
import pandas as pd
import polars as pl
import aggregate

from enum import Enum
from itertools import compress
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector


class FileType(str, Enum): 
        BASE = 'base'
        DEPTH = 'depth'

class Preprocessing:

    def __init__(self, directory: str, regexes: list, schemas: dict):
        self.directory = directory
        self.regexes = regexes
        self.schemas = schemas
        self.datastore = {}

    def handle_dates(self, df: pl.DataFrame) -> pl.DataFrame:
        date_schemas = ["date_decision", "^.*_\d.*[D]$"]
        df = df.with_columns(((pl.col(date_schemas[1]) - pl.col(date_schemas[0])).dt.total_days()).cast(pl.Float64))
        df = df.drop(date_schemas[0], "MONTH")
        return df

    def encoding(self, df: pl.DataFrame): 
        categorical_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ('encode', OrdinalEncoder())
            ]
        )
        numeric_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ('scale', StandardScaler())
            ]
        )
        transformer = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, make_column_selector(dtype_include=np.number)),
                ("categorical", categorical_pipeline, make_column_selector(dtype_include=[object, bool]))
            ]
        )
        transformed = transformer.fit_transform(df.to_pandas())
        return pd.DataFrame(transformed, columns=transformer.get_feature_names_out(df.to_pandas().columns))
    
    def remove_nulls(self, df: pl.DataFrame, threshold: float) -> pl.DataFrame: 
        df = df.fill_nan(None)
        df = df.select(compress(df.columns, df.select(pl.all().is_null().mean() <= threshold).row(0)))
        return df
    
    def join_data(self) -> pl.DataFrame: 
        self.datastore = aggregate.Aggregate(self.directory, self.regexes, self.schemas).populate_datastore()
        df = self.datastore[FileType.BASE]
        for depth, df in self.datastore[FileType.DEPTH].items(): 
            df = df.join(df, on="case_id", how="left")
        df = df.join(self.datastore[FileType.BASE], on="case_id", how="left")
        return df

    def preprocessing(self, threshold): 
        df = self.join_data()
        df = self.handle_dates(df)
        df = df.select(pl.exclude("^.*_right$"))
        df = self.remove_nulls(df, threshold)
        base = df.select(["case_id", "WEEK_NUM", "target"]).to_pandas()
        X = self.encoding(df.select(pl.exclude(["case_id", "WEEK_NUM", "target"])))
        y = df.select("target").to_pandas()
        return base, X, y