import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif

class FeatureSelection:

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame):
        self.X = X
        self.y = y

    def get_information_gain_features(self, num_features: int = None, plot: bool = False) -> list:
        ig = {}
        gain = mutual_info_classif(self.X, self.y)
        for i in range(len(self.X.columns)):
            ig[self.X.columns[i]] = gain[i]
        ig_sorted = dict(sorted(ig.items(), key=lambda item: item[1], reverse=True))
        optimized = {col: gain for col, gain in ig_sorted.items() if gain != 0}

        if plot:
            sns.set_theme(style="whitegrid")
            sns.set_theme(rc={"figure.figsize": (50, 30)})
            sns.barplot(x=list(ig_sorted.values()), y=list(ig_sorted.keys()))
            plt.title("Information Gain of Features")
            plt.xlabel("Information Gain")
            plt.ylabel("Feature Name")
            plt.savefig("features_20.png", bbox_inches='tight')
            plt.clf()
            sns.barplot(x=list(optimized.values()), y=list(optimized.keys()))
            plt.title("Information Gain of Optimized Features")
            plt.xlabel("Information Gain")
            plt.ylabel("Feature Name")
            plt.savefig("features_optimized_20.png", bbox_inches='tight')

        if num_features:
            return list(optimized.keys())[:num_features]
        else: 
            return list(optimized.keys())
