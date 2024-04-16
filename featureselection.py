import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif

class FeatureSelection: 
    
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame): 
        self.X = X
        self.y = y

    def get_information_gain_features(self, num_features: int, plot: bool=False): 
        ig = mutual_info_classif(self.X, self.y)
        ig_dict = {}
        for i in range(len(self.X.columns)):
            ig_dict[self.X.columns[i]] = ig[i]
        ig_dict_sorted = dict(sorted(ig_dict.items(), key=lambda item: item[1], reverse=True))
        if plot: 
            sns.set_theme(style="whitegrid")
            sns.set_theme(rc={'figure.figsize':(12,8)})
            sns.barplot(x=list(ig_dict_sorted.values()), y=list(ig_dict_sorted.keys()))
            plt.title('Information Gain of Features')
            plt.xlabel('Information Gain')
            plt.ylabel('Feature Name')
            plt.show()
        return list(ig_dict_sorted.keys())[:num_features]