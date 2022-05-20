# from asyncore import file_dispatcher
# from fileinput import filename
import warnings
import os
import datetime
import time
from turtle import filling
from xml.dom import NO_MODIFICATION_ALLOWED_ERR
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from carla.data.catalog import OnlineCatalog, CsvCatalog
from carla import MLModelCatalog
from custom_model import CustomModel
from carla.recourse_methods import Revise
from carla.recourse_methods import Wachter
from carla.models.negative_instances import predict_negative_instances

from scipy.special import rel_entr
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator

from itertools import combinations

import copy
import argparse

import logging

logger = logging.Logger("qlogger")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh = logging.FileHandler("test.log")
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

training_params = {"lr": 0.005, "epochs": 10, "batch_size": 1, "hidden_size": [4]}

def create_dataset(dataset_name, num):
    dataset, label = make_classification(n_samples=num, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, flip_y=0, class_sep=2)

    feature_labels = ["feature 1", "feature 2"]

    df = pd.DataFrame(dataset, columns=feature_labels)
    df = normalize_dataset(df, feature_labels)


    plt.scatter(df["feature 1"].tolist(), df["feature 2"].tolist(), c=label)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig(f"{dataset_name}_scatterplot.png")

    df["class"] = label

    df.to_csv(dataset_name, sep=',', encoding='utf-8', index=False)

    logger.debug(f"Saved created dataset to csv: {dataset_name}")


def normalize_dataset(dataset, features):
    min_val = dataset.min()
    for f in features:
        dataset[f] = dataset[f].apply(lambda x: x - min_val[f])
    dataset = (dataset-dataset.min())/(dataset.max()-dataset.min())
    return dataset

# reads the csv containing the data, assumes that the last column is the target and not a feature
def read_dataset(filename):
    dataset = pd.read_csv(filename)
    features = dataset.columns.tolist()[0:-1]
    dataset.index.name = 'factual_id'
    return (dataset, features)


def load_custom_dataset_model(filename):

    dataset, features = read_dataset(filename)
    dataset = CsvCatalog(file_path=filename, continuous=features, categorical=[], immutables=[], target="class")
    model = CustomModel(dataset, model_type="ann", backend="pytorch", load_online=False, cache=False)
    model.train(
        learning_rate=training_params["lr"],
        epochs=training_params["epochs"],
        batch_size=training_params["batch_size"],
        hidden_size=training_params["hidden_size"],
        )
    while (model.get_test_accuracy() < 0.8):
        logger.debug(f'model accuracy was below the set threshold: {model.get_test_accuracy()}')
        model.train(
        learning_rate=training_params["lr"],
        epochs=training_params["epochs"],
        batch_size=training_params["batch_size"],
        hidden_size=training_params["hidden_size"],
        )
    logger.debug(f'model accuracy: {model.get_test_accuracy()}')
    return (dataset, model, features)


# load a catalog dataset
def load_real_model(dataset_name):
    dataset = OnlineCatalog(dataset_name)
    model = MLModelCatalog(dataset, "ann", backend="pytorch")
    return (dataset, model)


def load_revise(dataset, dataset_name, model):
    hyperparams = {
        "data_name": dataset_name, 
        "vae_params": {
            "layers": [len(model.feature_input_order), 512, 256, 8],
        }
    }
    return Revise(model, dataset, hyperparams)


def load_wachter(dataset, dataset_name, model):
    hyperparams = {
        "loss_type": "BCE", 
        "t_max_min": 1 / 60,
        "binary_cat_features": False
    }
    return Wachter(model, hyperparams)


# get factuals from the data to generate counterfactual examples
def get_factuals(model, dataset, random=False, sample=None):
    factuals = predict_negative_instances(model, dataset.df)
    return factuals


def norm_model_prob(model, input):
    output = np.nan_to_num(model.predict(input))
    # normalize output vector
    output /= np.sum(output)
    return output


def kl_divergence( base_model, modified_model):
    # retrieve the KL Divergence by summing the relative entropy of two probabilities
    vec = rel_entr(base_model, modified_model)
    return np.sum(vec)


def run_recourse_method(factuals, recourse_method, recourse_name):
    start = time.time()
    counterfactuals = recourse_method.get_counterfactuals(factuals)
    end = time.time()
    
    count = counterfactuals.dropna(inplace=False)
    # print(count)
    count = counterfactuals.shape[0]
    logger.debug(f"It took {end-start:.2f} seconds to find {count} counterfactuals for {recourse_name}")
    return (counterfactuals, count)


def kmeans(df):
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

    logger.debug(f"{len(ssd)} {len(x)}")

    kn = KneeLocator(x, ssd, curve='convex', direction='decreasing')
    logger.debug(f"\'Optimal\' amount of clusters found: {kn.knee}")

    return pd.DataFrame(centers[kn.knee-1])



def measurements(model, dataset, filename, counterfactuals, features, round, name):

    res = {"Count": counterfactuals.shape[0], "Accuracy": model.get_test_accuracy()}

    negative = dataset._df[dataset._df[dataset.target] == 0]
    positive = dataset._df[dataset._df[dataset.target] == 1]


    # calc mean, cov ...
    # print(negative)
    for x in features:
        res[f"mean negative {x}"] = negative[x].mean()
        res[f"mean positive {x}"] = positive[x].mean()
        res[f"median negative {x}"] = negative[x].median()
        res[f"median positive {x}"] = positive[x].median()

    df = dataset.df.drop(columns=[dataset.target] ,inplace=False)
    kmeans_centers = kmeans(df)

    # print(kmeans_centers)

    fig, ax = plt.subplots()
    plt.scatter(dataset._df["feature 1"].tolist(), dataset._df["feature 2"].tolist(), c=dataset._df["class"])
    kmeans_centers.plot.scatter(x=0, y=1, ax=ax, marker="*", c="red")
    ax.set_title(f'Scatterplot {filename} {name} {round}')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    plt.savefig(f"{filename}_{name}_{round}_clusters.png")
    plt.close()

    kmeans_centers.to_csv(f'{filename}_{name}_{round}_kmeans_centers.csv', sep=',', encoding='utf-8', index=False)
    df.cov().to_csv(f'{filename}_{name}_{round}_cov.csv', sep=',', encoding='utf-8', index=False)

    logger.debug(f"{name} {res}")
    return pd.Series(res)


def run_rounds(df_res, rounds, batch, model, dataset, factuals, recourse_function, recourse_name, filename):    
    for i in range(1, rounds+1):
        recourse = recourse_function(dataset, "custom", model)

        try:
            (counterfactuals, count) = run_recourse_method(factuals, recourse, recourse_name)
        except ValueError:
            logger.error(f"retrying retrieval of counterfactuals because of an error with the model")

        # print(count)
        counterfactuals["factual_id"] = factuals.index
        counterfactuals.set_index("factual_id", inplace=True, drop=True)
        counterfactuals.dropna(inplace=True)
        # print(revise_counterfactuals)
        # print(revise_factuals)

        if (counterfactuals.shape[0] >= batch):
            
            samples = counterfactuals.sample(batch)
            df = dataset._df
            logger.debug(f"Counterfactuals added: {samples.shape[0]}")
            df.index.name = "factual_id"
            df.loc[samples.index, :] = samples[:]
            # print(df.loc[samples.index[0]])
            # print(df.loc[samples.index])

            df.to_csv(f"{filename}-{recourse_name}-{i}", sep=',', encoding='utf-8', index=False)
        else:
            logger.debug(f"not enough counterfactuals to use with {recourse_name} to complete all rounds")


        dataset, features = read_dataset(f"{filename}-{recourse_name}-{i}")
        dataset = CsvCatalog(file_path=f"{filename}-{recourse_name}-{i}", continuous=features, categorical=[], immutables=[], target="class")
        model.train(
            learning_rate=training_params["lr"],
            epochs=training_params["epochs"],
            batch_size=training_params["batch_size"],
            hidden_size=training_params["hidden_size"],
        )

        res = measurements(model, dataset, filename, counterfactuals, features, round=i, name=recourse_name)
        df_res = df_res.append(res, ignore_index=True)
        factuals = get_factuals(model, dataset)
    
    return df_res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--custom', help="use a custom dataset", action="store_true")
    parser.add_argument('-d', '--dataset', help="create an artificial dataset using the given name")
    parser.add_argument('-n', '--num', default=200, type=int, help="number of datapoints for custom dataset")
    parser.add_argument('-m', '--model', help="filename of the dataset to train the model on")
    parser.add_argument('-b', '--batch', default=5, type=int, help="batch size of added counterfactuals per round")
    parser.add_argument('-r', '--rounds', default=5, type=int, help="amount of rounds to add counterfactuals and retrain the model")

    args = parser.parse_args()

    if args.dataset:
        create_dataset(args.dataset, args.num)

    if args.custom:

        start = time.time()

        logger.debug("process started")

        dataset, model, features = load_custom_dataset_model(args.model)
        logger.debug(hash(model))

        df_revise = pd.DataFrame()
        df_wachter = pd.DataFrame()
        res = measurements(model, dataset, args.model, pd.DataFrame(), features, round=0, name="Start")

        df_revise = df_revise.append(res, ignore_index=True)
        df_wachter = df_wachter.append(res, ignore_index=True)

        logger.debug("starting revise")
        dataset_revise = copy.deepcopy(dataset)
        model_revise = copy.deepcopy(model)
        revise_factuals = get_factuals(model_revise, dataset_revise)
        df_revise = run_rounds(df_revise, args.rounds, args.batch, model_revise, dataset_revise, revise_factuals, load_revise, "REVISE", args.model)
        df_revise.to_csv(f'df_revise_{args.model}.csv', sep=',', encoding='utf-8', index=False)
        logger.debug("done with revise")

        logger.debug("starting wachter")
        dataset_wachter = copy.deepcopy(dataset)
        model_wachter = copy.deepcopy(model)
        wachter_factuals = get_factuals(model_wachter, dataset_wachter)
        df_wachter = run_rounds(df_wachter, args.rounds, args.batch, model_wachter, dataset_wachter, wachter_factuals, load_wachter, "Wachter", args.model)
        df_wachter.to_csv(f'df_wachter_{args.model}.csv', sep=',', encoding='utf-8', index=False)
        logger.debug("done with wachter")

        end = time.time()

        logger.debug(hash(model))
        logger.debug(f"whole process took: {end-start} seconds.")

    if not args.custom:
        dataset, model = load_real_model("adult")
        # TODO: almost the same code with the custom dataset


if __name__ == "__main__":
    main()
