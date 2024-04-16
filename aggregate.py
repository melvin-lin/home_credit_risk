import re
import os
import polars as pl

import preprocessing

class Aggregate: 

    def __init__(self, directory: str, regexes: list, schemas: dict): 
        self.directory = directory
        self.regexes = regexes
        self.schemas = schemas

    def set_table_dtypes(self, df: pl.DataFrame) -> pl.DataFrame:
        schema = dict((key,value) for key, value in self.schemas.items() if any(re.match(key, col) for col in df.columns))
        return df.cast(schema)
    
    def get_dataframe(self, path, depth=None) -> pl.DataFrame: 
        df = pl.read_parquet(path).pipe(self.set_table_dtypes)
        return df.group_by("case_id").agg(self.get_aggregations(df, ["mean"])) if depth else df
    
    def get_aggregations(self, df: pl.DataFrame, expressions: list): 
        aggregations = []
        for transform in ["P", "M", "A", "D", "T", "L"]: 
            regex = f"^.*_\d.*[{transform}]$"
            cols = [col for col in df.columns if re.match(regex, col)]
            aggregations += self.get_expressions(cols, expressions)
        for transform in range(1,3): 
            regex = f"^.*_group[{transform}]$"
            cols = [col for col in df.columns if re.match(regex, col)]
            aggregations += self.get_expressions(cols, expressions)
        return aggregations

    def get_expressions(self, columns: list, expressions: list): 
        aggregations = []
        if "mean" in expressions:
            aggregations += [pl.mean(col).alias(f"{col}_mean") for col in columns] 
        if "sum" in expressions: 
            aggregations += [pl.sum(col).alias(f"{col}_sum") for col in columns] 
        if "count" in expressions: 
            aggregations += [pl.count(col).alias(f"{col}_count") for col in columns] 
        if "max" in expressions: 
            aggregations += [pl.max(col).alias(f"{col}_max") for col in columns] 
        if "min" in expressions: 
            aggregations += [pl.min(col).alias(f"{col}_min") for col in columns]  
        return aggregations
    
    def populate_datastore(self):
        datastore = {preprocessing.FileType.BASE: [], preprocessing.FileType.DEPTH: {}}
        paths = self.get_filenames()
        for filetype, values in paths.items(): 
            if isinstance(values, dict): 
                for depth, files in values.items(): 
                    datastore[preprocessing.FileType.DEPTH][depth] = self.read_files(files, depth)
            else: 
                datastore[preprocessing.FileType.BASE] = self.read_files(values)
        return datastore

    def read_files(self, paths, depth=None): 
        dfs = []
        for path in paths: 
            df = self.get_dataframe(path, depth)
            dfs.append(df)
        df = pl.concat(dfs, how="diagonal_relaxed")
        return df.unique(subset="case_id")
        
    def get_filenames(self): 
        paths = {preprocessing.FileType.BASE: [], preprocessing.FileType.DEPTH: {}}
        for filename in os.listdir(self.directory): 
            for i in range(len(self.regexes)): 
                regex = re.compile(self.regexes[i])
                if regex.match(filename):
                    path = [os.path.join(self.directory, filename)]
                    if i == 0: 
                        paths[preprocessing.FileType.BASE] += path
                    else: 
                        try: 
                            paths[preprocessing.FileType.DEPTH][i-1] += path
                        except: 
                            paths[preprocessing.FileType.DEPTH].update({i-1: path})
        return paths