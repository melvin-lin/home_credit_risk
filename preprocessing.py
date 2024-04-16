
import pandas as pd
import polars as pl
import aggregate

from enum import Enum
from itertools import compress
from sklearn.preprocessing import LabelEncoder


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
    
    def encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        le = LabelEncoder()
        for col in df.columns:
            if df[col].dtype in ['object', 'str', 'bool']:
                if len(list(df[col].unique())) < 3: 
                    le.fit(df[col])
                    df[col] = le.transform(df[col])
        return pd.get_dummies(df, dtype=float)
    
    def clean_data(self, df: pl.DataFrame, threshold: float) -> pl.DataFrame: 
        df = df.fill_nan(None)
        df = df.select(compress(df.columns, df.select(pl.all().is_null().mean() <= threshold).row(0)))
        unique = df.select(pl.all().n_unique() < 2)
        unique = unique.select(compress(unique.columns, unique.row(0)))
        df = df.drop(unique.columns)
        return df
    
    def join_data(self) -> pl.DataFrame: 
        self.datastore = aggregate.Aggregate(self.directory, self.regexes, self.schemas).populate_datastore()
        df = self.datastore[FileType.BASE]
        for depth, df in self.datastore[FileType.DEPTH].items(): 
            df = df.join(df, on="case_id", how="left")
        df = df.join(self.datastore[FileType.BASE], on="case_id", how="left")
        return df

    def preprocessing(self, threshold) -> pd.DataFrame: 
        df = self.join_data()
        df = self.handle_dates(df)
        df = df.select(pl.exclude("^.*_right$"))
        df = self.clean_data(df, threshold)
        df = df.fill_null(strategy="backward")
        df = self.encoding(df.to_pandas())
        df = df.fillna(df.mean())
        return df