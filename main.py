from asyncore import file_dispatcher
import warnings
import os
import datetime
import time
from xml.dom import NO_MODIFICATION_ALLOWED_ERR
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from carla.data.catalog import OnlineCatalog, CsvCatalog
from carla import MLModelCatalog
from carla.recourse_methods import Revise
from carla.recourse_methods import Wachter
from carla.models.negative_instances import predict_negative_instances

from scipy.special import rel_entr
from scipy.stats import wasserstein_distance
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import copy
import argparse
import pickle

import sys

result = open('result - {}.txt'.format(datetime.datetime.now().strftime("%d-%b-%Y (%H.%M.%S)")), 'w')
training_params = {"lr": 0.002, "epochs": 50, "batch_size": 512, "hidden_size": [18, 9, 3]}

def create_dataset(dataset_name):
    dataset, label = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, flip_y=0, class_sep=2)

    plt.scatter(dataset[:,0], dataset[:,1], c=label)
    plt.savefig(f"{dataset_name}_scatterplot.png")
    
    feature_labels = ["feature 1", "feature 2"]

    df = pd.DataFrame(dataset, columns=feature_labels)
    df["class"] = label

    df.to_csv(dataset_name, sep=',', encoding='utf-8', index=False)

    print(f"saved dataset to csv: {dataset}")


# reads the csv containing the data, assumes that the last column is the target and not a feature
def read_dataset(filename):
    dataset = pd.read_csv(filename)
    features = dataset.columns.tolist()[0:-1]
    print(dataset)
    print(features)

    return (dataset, features)


def load_custom_dataset_model(filename):

    dataset, features = read_dataset(filename)
    dataset = CsvCatalog(file_path=filename, continuous=features, categorical=[], immutables=[], target="class")
    
    model = MLModelCatalog(dataset, "ann", backend="pytorch", load_online=False)
    model.train(
        learning_rate=training_params["lr"],
        epochs=training_params["epochs"],
        batch_size=training_params["batch_size"],
        hidden_size=training_params["hidden_size"]
    )

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


def load_wachter(model):
    hyperparams = {
        "loss_type": "BCE", 
        "binary_cat_features": False
    }
    return Wachter(model, hyperparams)


# get factuals from the data to generate counterfactual examples
def get_factuals(model, dataset, random=True, sample=None):
    factuals = None
    if random:
        if sample is None:
            factuals = predict_negative_instances(model, dataset.df).reset_index(drop=True)
        else: 
            factuals = predict_negative_instances(model, dataset.df).sample(sample).reset_index(drop=True)

    else:
        factuals = predict_negative_instances(model, dataset.df).iloc[:100].reset_index(drop=True)
    return factuals


# generate counterfactual examples
def get_counterfactuals(recourse_method, factuals, timing = False):
    start = time.time()

    counterfactuals = recourse_method.get_counterfactuals(factuals)
    count = np.count_nonzero(counterfactuals.iloc[:, 1])
    if timing:
        end = time.time()
        print("It took {:.2f} seconds to find {} counterfactuals for REVISE".format(end-start, count), file=result)
        
    # Sort the columns of the counterfactuals so that it has the same order as that
    # of the factuals. This makes it possible to combine the 2 dataframes.
    counterfactuals = counterfactuals.reindex(sorted(counterfactuals.columns), axis=1)
    counterfactuals = counterfactuals.combine_first(factuals)

    return (count, counterfactuals)


def norm_model_prob(model, input):
    output = np.nan_to_num(model.predict(input))
    # normalize output vector
    output /= np.sum(output)
    return output


def kl_divergence(model, factuals, counterfactuals):
    # retrieve the KL Divergence by summing the relative entropy of two probabilities
    norm_fact = norm_model_prob(model, factuals)
    norm_count = norm_model_prob(model, counterfactuals)

    # print(norm_fact)
    # print(norm_count)
    vec = rel_entr(norm_fact, norm_count)
    return np.sum(vec)


def run_recourse_method(factuals, recourse_method, recourse_name):
    start = time.time()
    counterfactuals = recourse_method.get_counterfactuals(factuals)
    end = time.time()
    
    count = np.count_nonzero(counterfactuals.iloc[:, 1])
    print(f"It took {end-start:.2f} seconds to find {count} counterfactuals for {recourse_name}", file=result)
        
    # Sort the columns of the counterfactuals so that it has the same order as that
    # of the factuals. This makes it possible to combine the 2 dataframes.
    counterfactuals = counterfactuals.reindex(sorted(counterfactuals.columns), axis=1)
    counterfactuals = counterfactuals.combine_first(factuals)
    return (counterfactuals, count)


def measurements(model, dataset, factuals, counterfactuals, count, name):
    kl_div = kl_divergence(model, factuals, counterfactuals)
    print(f"measurements for recourse method: {name}")
    print(f"KL Divergence: {kl_div}")
    # TODO: fix this to run for every step and save the data somewhere


def run_revise(dataset, model, factuals):
    recourse_method = load_revise(dataset, "custom", model)

    (count, counterfactuals) = get_counterfactuals(recourse_method, factuals, timing=True)

    kl_div = kl_divergence(model, factuals, counterfactuals)
    print(norm_model_prob(model, factuals).shape)
    print("The KL divergence of REVISE in this dataset is: {}".format(kl_div), file=result)

    return counterfactuals


def run_wachter(dataset, model, factuals):
    hyperparams = {
        "loss_type": "BCE", 
        "binary_cat_features": False
    }
    recourse_method = Wachter(model, hyperparams)

    start = time.time()
    counterfactuals = recourse_method.get_counterfactuals(factuals)
    print(counterfactuals.iloc[:, 1].isnull())
    print(counterfactuals.iloc[:, 1].isnull().sum())
    count = np.count_nonzero(counterfactuals.iloc[:, 1])

    end = time.time()
    print("It took {:.2f} seconds to find {} counterfactuals for WACHTER".format(end-start, count), file=result)

    # Sort the columns of the counterfactuals so that it has the same order as that
    # of the factuals. This makes it possible to combine the 2 dataframes.
    df_cfs = counterfactuals.reindex(sorted(counterfactuals.columns), axis=1)
    df_cfs = df_cfs.combine_first(factuals)


    kl_div = kl_divergence(model, factuals, df_cfs)
    print("The KL divergence of WACHTER in this dataset is: {}".format(kl_div), file=result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--custom', help="use a custom dataset", action="store_true")
    parser.add_argument('-d', '--dataset', help="create an artificial dataset using the given name")
    parser.add_argument('-m', '--model', help="filename of the dataset to train the model on")

    args = parser.parse_args()

    if args.dataset:
        create_dataset(args.dataset)

    if args.custom:
        dataset, model, features = load_custom_dataset_model(args.model)

        model_revise = copy.copy(model)
        model_wachter = copy.copy(model)


        mean_features = map(lambda x: f"mean {x}", features)
        cov_features = map(lambda x: f"cov {x}", features)

        columns = ["Count", "KL Divergence"]
        columns.extend(mean_features)
        columns.extend(cov_features)

        df_revise = pd.DataFrame(columns=columns)
        df_wachter = pd.DataFrame(columns=columns)

        print(df_revise)

        factuals = get_factuals(model_revise, dataset, random=True, sample=10)
        revise = load_revise(dataset, "custom", model_revise)
        wachter = load_wachter(model_wachter)
        (revise_counterfactuals, revise_count) = run_recourse_method(factuals, revise, "REVISE")
        (wachter_counterfactuals, wachter_count) = run_recourse_method(factuals, wachter, "Wachter")

        measurements(model_revise, dataset, factuals, revise_counterfactuals, revise_count, name="REVISE")
        measurements(model_wachter, dataset, factuals, wachter_counterfactuals, wachter_count, name="Wachter")


    if not args.custom:
        dataset, model = load_real_model("adult")

    # factuals = run_revise(dataset, model)

    # # START WATCHER

    # run_wachter(dataset, model, factuals)

    print("SUCCESS", file=result)
    result.close()


if __name__ == "__main__":
    main()
