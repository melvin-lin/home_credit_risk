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
            fig, ax = plt.subplots(figsize =(16, 9))
            ax.barh(list(ig_sorted.values()), list(ig_sorted.keys()))
            for s in ['top', 'bottom', 'left', 'right']:
                ax.spines[s].set_visible(False)
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.xaxis.set_tick_params(pad = 5)
            ax.yaxis.set_tick_params(pad = 10)
            ax.grid(b = True, color ='grey',
                    linestyle ='-.', linewidth = 0.5,
                    alpha = 0.2)
            ax.invert_yaxis()

            for i in ax.patches:
                plt.text(i.get_width()+0.2, i.get_y()+0.5, 
                        str(round((i.get_width()), 2)),
                        fontsize = 10, fontweight ='bold',
                        color ='grey')
            
            ax.set_title('Information Gain of Features',
                        loc ='left', )
            plt.savefig('features.png')
 

            # plt.clf()
            # sns.set_theme(style="whitegrid")
            # sns.set_theme(rc={"figure.figsize": (12, 8)})
            # sns.barplot(x=list(ig_sorted.values()), y=list(ig_sorted.keys()))
            # plt.title("Information Gain of Features")
            # plt.xlabel("Information Gain")
            # plt.ylabel("Feature Name")
            # plt.show()

            # if num_features: 
            #     plt.clf()
            #     sns.barplot(x=list(optimized.values())[:num_features], y=list(optimized.keys())[:num_features])
            # else: 
            #     plt.clf()
            #     sns.barplot(x=list(optimized.values()), y=list(optimized.keys()))
            # plt.title("Information Gain of Optimized Features")
            # plt.xlabel("Information Gain")
            # plt.ylabel("Feature Name")
            # plt.savefig(f"features_optimized_{num_features if num_features is not None else len(list(optimized.keys()))}.png", bbox_inches='tight')

        if num_features:
            return list(optimized.keys())[:num_features]
        return list(optimized.keys())
