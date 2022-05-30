import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.special import rel_entr
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator

class Measurements:
    def __init__(self, logger, model, dataset):
        self.logger = logger
        self.original_model = model
        self.original_dataset = dataset
    
    def norm_model_prob(self, model, input):
        output = np.nan_to_num(model.predict(input))
        # normalize output vector
        output /= np.sum(output)
        return output


    def kl_divergence(self, model):
        # retrieve the KL Divergence by summing the relative entropy of two probabilities
        vec = rel_entr(self.original_model, model)
        return np.sum(vec)

    def kmeans(self, df):
        mms = MinMaxScaler()
        mms.fit(df)
        data_transformed = mms.transform(df)

        ssd = []
        K = 7
        centers = []
        for k in range(1, K):
            km = KMeans(n_clusters=k)
            km = km.fit(data_transformed)
            ssd.append(km.inertia_)
            centers.append(km.cluster_centers_)

        x = range(1, K)

        self.logger.debug(f"{len(ssd)} {len(x)}")

        kn = KneeLocator(x, ssd, curve='convex', direction='decreasing')
        self.logger.debug(f"\'Optimal\' amount of clusters found: {kn.knee}")

        return pd.DataFrame(centers[kn.knee-1])


    def mmd_linear(self, X, Y):
        XX = np.dot(X, X.T)
        YY = np.dot(Y, Y.T)
        XY = np.dot(X, Y.T)
        return XX.mean() + YY.mean() - 2 * XY.mean()


    def disagreement_coefficient(self, model, dataset):
        original_prediction = (self.original_model.predict(dataset._df) > 0.5).flatten()
        modified_prediction = (model.predict(dataset._df) > 0.5).flatten()
        
        correct = original_prediction != modified_prediction
        return correct.sum()/len(original_prediction)


    def measure(self, model, dataset, filename, counterfactuals, features, round, name):

        df = dataset.df.drop(columns=[dataset.target] ,inplace=False)
        # contour plot
        min1, max1 = df["feature1"].min()-1, df["feature1"].max()+1
        min2, max2 = df["feature2"].min()-1, df["feature2"].max()+1

        x1grid = np.linspace(-0.1, 1.1, 100)
        x2grid = np.linspace(-0.1, 1.1, 100)

        xx, yy = np.meshgrid(x1grid, x2grid)

        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
        
        grid = np.hstack((r1,r2))
        yhat = model.predict(grid)
        original_yhat = self.original_model.predict(grid)

        zz = yhat.reshape(xx.shape)
        # end contour plot
        negative = dataset._df[dataset._df[dataset.target] == 0].drop(columns=[dataset.target])
        positive = dataset._df[dataset._df[dataset.target] == 1].drop(columns=[dataset.target])

        res = {
            "Accuracy": model.get_test_accuracy(), 
            "Boundary Width Negative": np.square(model.predict(negative) - 0.5).sum() / negative.shape[0],
            "Boundary Width Positive": np.square(model.predict(positive) - 0.5).sum() / positive.shape[0],
            "Count": counterfactuals.shape[0], 
            "Disagreement": self.disagreement_coefficient(model, dataset),
            "F1-score": model.get_F1_score(),
            "Mean Counterfactual Probability": 0 if (counterfactuals.shape[0] == 0) else model.predict(counterfactuals).mean(),
            "MMD domain": self.mmd_linear(
                dataset._df.loc[:, dataset._df.columns != dataset.target], 
                self.original_dataset._df.loc[:, self.original_dataset._df.columns != self.original_dataset.target]
                ),
            "MMD model": self.mmd_linear(yhat, original_yhat),
            "MMD probabilities": self.mmd_linear(
                self.original_model.predict(dataset._df.loc[:, dataset._df.columns != dataset.target]), 
                model.predict(dataset._df.loc[:, dataset._df.columns != dataset.target])
                ),
        }


        # calc mean, cov ...
        # print(negative)
        for x in features:
            res[f"mean negative {x}"] = negative[x].mean()
            res[f"mean positive {x}"] = positive[x].mean()

        kmeans_centers = self.kmeans(df)


        fig, ax = plt.subplots()

        fig.set_size_inches(10, 7.5)
        fig.set_dpi(300)
        
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.contourf(xx, yy, zz, cmap="GnBu")
        plt.clim(0, 1)
        cb = plt.colorbar()
        cb.set_label("probability")
        plt.scatter(dataset._df["feature1"].tolist(), dataset._df["feature2"].tolist(), c=dataset._df["target"])
        kmeans_centers.plot.scatter(x=0, y=1, ax=ax, marker="*", c="red")
        ax.ticklabel_format(useOffset=False)
        ax.set_title(f'Scatterplot {filename} {name} {round}')
        ax.set_xlabel("Feature1")
        ax.set_ylabel("Feature2")
        fig.savefig(f"{filename}_{name}_{round}.png")
        plt.close()

        kmeans_centers.to_csv(f'{filename}_{name}_{round}_kmeans_centers.csv', sep=',', encoding='utf-8', index=False)
        df.cov().to_csv(f'{filename}_{name}_{round}_cov.csv', sep=',', encoding='utf-8', index=False)

        self.logger.debug(f"{name} {res}")
        
        return pd.Series(res)