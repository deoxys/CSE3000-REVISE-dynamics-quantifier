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

import sys

result = open('result - {}.txt'.format(datetime.datetime.now().strftime("%d-%b-%Y (%H.%M.%S)")), 'w')

def load_custom_dataset_model():

    dataset, label = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, flip_y=0, class_sep=2)

    plt.scatter(dataset[:,0], dataset[:,1], c=label)
    plt.savefig("scatter.png")
    
    feature_labels = ["feature 1", "feature 2"]

    df = pd.DataFrame(dataset, columns=feature_labels)
    df["class"] = label

    print(df)
    df.to_csv("dataset", sep=',', encoding='utf-8', index=False)

    dataset = CsvCatalog(file_path="dataset", continuous=feature_labels, categorical=[], immutables=[], target="class")
    
    training_params = {"lr": 0.002, "epochs": 50, "batch_size": 512, "hidden_size": [18, 9, 3]}
    model = MLModelCatalog(dataset, "ann", backend="pytorch", load_online=False)
    model.train(
        learning_rate=training_params["lr"],
        epochs=training_params["epochs"],
        batch_size=training_params["batch_size"],
        hidden_size=training_params["hidden_size"]
    )    
    
    return (dataset, model)

# load a catalog dataset
def load_real_model(dataset_name):
    dataset = OnlineCatalog(dataset_name)
    model = MLModelCatalog(dataset, "ann", backend="pytorch")
    return (dataset, model)

def load_Revise(dataset, dataset_name, model):
    hyperparams = {
        "data_name": dataset_name, 
        "vae_params": {
            "layers": [len(model.feature_input_order), 512, 256, 8],
        }
    }
    return Revise(model, dataset, hyperparams)


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

    print(norm_fact)
    print(norm_count)
    vec = rel_entr(norm_fact, norm_count)
    return np.sum(vec)


def run_revise(dataset, model):
    recourse_method = load_Revise(dataset, "adult", model)

    factuals = get_factuals(model, dataset, random=True, sample=50)
    (count, counterfactuals) = get_counterfactuals(recourse_method, factuals, timing=True)

    kl_div = kl_divergence(model, factuals, counterfactuals)
    print(norm_model_prob(model, factuals).shape)
    was_dist = wasserstein_distance(norm_model_prob(model, factuals).flatten(), norm_model_prob(model, counterfactuals).flatten())
    print(was_dist)
    print("The KL divergence of REVISE in this dataset is: {}".format(kl_div), file=result)

    return factuals


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


dataset, model = load_custom_dataset_model()

# dataset, model = load_real_model("adult")

factuals = run_revise(dataset, model)

# START WATCHER

run_wachter(dataset, model, factuals)

print("SUCCESS", file=result)

result.close()