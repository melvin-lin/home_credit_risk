import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif

class FeatureSelection:

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame):
        self.X = X
        self.y = y

    def get_information_gain_features(self, plot: bool = False) -> list:
        ig = {}
        gain = mutual_info_classif(self.X, self.y)
        for i in range(len(self.X.columns)):
            ig[self.X.columns[i]] = gain[i]
        ig_sorted = dict(sorted(ig.items(), key=lambda item: item[1], reverse=True))
        optimized = {col: gain for col, gain in ig_sorted.items() if gain != 0}

        if plot:
            sns.set_theme(style="whitegrid")
            sns.set_theme(rc={"figure.figsize": (12, 8)})
            sns.barplot(x=list(sorted.values()), y=list(sorted.keys()))
            plt.title("Information Gain of Features")
            plt.xlabel("Information Gain")
            plt.ylabel("Feature Name")
            plt.savefig("features.png")
        return list(optimized.keys())
