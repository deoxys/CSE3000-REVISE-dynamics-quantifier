import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_classification

class SyntheticDatasetGenerator():

    def __init__(self, logger):
        self.logger = logger


    def generate(self, dataset_name, n_samples):
        dataset, label = make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, flip_y=0, class_sep=2)

        feature_labels = ["feature1", "feature2"]

        df = pd.DataFrame(dataset, columns=feature_labels)
        df = self.normalize_dataset(df, feature_labels)


        plt.scatter(df["feature1"].tolist(), df["feature2"].tolist(), c=label)
        plt.xlabel("Feature1")
        plt.ylabel("Feature2")
        plt.savefig(f"{dataset_name}_scatterplot.png")

        df["target"] = label

        df.to_csv(dataset_name, sep=',', encoding='utf-8', index=False)

        self.logger.debug(f"Saved created dataset to csv: {dataset_name}")


    def normalize_dataset(self, dataset, features):
        min_val = dataset.min()
        for f in features:
            dataset[f] = dataset[f].apply(lambda x: x - min_val[f])
        dataset = (dataset-dataset.min())/(dataset.max()-dataset.min())
        return dataset