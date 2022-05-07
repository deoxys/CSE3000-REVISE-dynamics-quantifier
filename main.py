from asyncore import file_dispatcher
from fileinput import filename
import warnings
import os
import datetime
import time
from xml.dom import NO_MODIFICATION_ALLOWED_ERR
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
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

result = open('result - {}.txt'.format(datetime.datetime.now().strftime("%d-%b-%Y (%H.%M.%S)")), 'w')
training_params = {"lr": 0.002, "epochs": 50, "batch_size": 16, "hidden_size": [18, 9, 3]}

def create_dataset(dataset_name):
    dataset, label = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, flip_y=0, class_sep=2)

    feature_labels = ["feature 1", "feature 2"]

    df = pd.DataFrame(dataset, columns=feature_labels)
    df = normalize_dataset(df, feature_labels)


    plt.scatter(df["feature 1"].tolist(), df["feature 2"].tolist(), c=label)
    plt.savefig(f"{dataset_name}_scatterplot.png")

    df["class"] = label

    df.to_csv(dataset_name, sep=',', encoding='utf-8', index=False)

    print(f"saved dataset to csv: {df}")


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
    plt.scatter(dataset["feature 1"].tolist(), dataset["feature 2"].tolist(), c=dataset["class"])
    plt.savefig(f"{filename}_scatterplot.png")
    plt.close()
    print(dataset)
    return (dataset, features)


def load_custom_dataset_model(filename):

    dataset, features = read_dataset(filename)
    dataset.index.name = 'factual_id'
    dataset = CsvCatalog(file_path=filename, continuous=features, categorical=[], immutables=[], target="class")
    model = MLModelCatalog(dataset, "ann", backend="pytorch", load_online=False, cache=False)
    model.train(
        learning_rate=training_params["lr"],
        epochs=training_params["epochs"],
        batch_size=training_params["batch_size"],
        hidden_size=training_params["hidden_size"],
        force_train=True
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
def get_factuals(model, dataset, random=False, sample=None):
    factuals = predict_negative_instances(model, dataset.df)
    return factuals.iloc[:]


def norm_model_prob(model, input):
    output = np.nan_to_num(model.predict(input))
    # normalize output vector
    output /= np.sum(output)
    return output


def kl_divergence( base_model, modified_model):
    # retrieve the KL Divergence by summing the relative entropy of two probabilities

    # print(norm_fact)
    # print(norm_count)
    vec = rel_entr(base_model, modified_model)
    return np.sum(vec)


def run_recourse_method(factuals, recourse_method, recourse_name):
    start = time.time()
    counterfactuals = recourse_method.get_counterfactuals(factuals)
    end = time.time()
    
    # print("run_recourse_method")
    # print(counterfactuals)
    count = counterfactuals.dropna(inplace=False)
    count = counterfactuals.shape[0]
    print(f"It took {end-start:.2f} seconds to find {count} counterfactuals for {recourse_name}", file=result)
    print(f"It took {end-start:.2f} seconds to find {count} counterfactuals for {recourse_name}")
        
    # Sort the columns of the counterfactuals so that it has the same order as that
    # of the factuals. This makes it possible to combine the 2 dataframes.
    # counterfactuals = counterfactuals.reindex(sorted(counterfactuals.columns), axis=1)
    # counterfactuals = counterfactuals.combine_first(factuals)

    return (counterfactuals, count)


def measurements(df, basemodel, model, dataset, factuals, counterfactuals, features, name):
    kl_div = kl_divergence(basemodel, model)
    print(f"measurements for recourse method: {name}")
    print(f"KL Divergence: {kl_div}")

    res = {"Count": counterfactuals.shape[0], "KL Divergence": kl_div }

    # calc mean, cov ...
    for x in features:
        res[f"mean {x}"] = dataset.df[x].mean()


    print(res, file=result)
    df.append(res, ignore_index=True)
    return res
    # TODO: fix this to run for every step and save the data somewhere

def run_rounds():
    print("")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--custom', help="use a custom dataset", action="store_true")
    parser.add_argument('-d', '--dataset', help="create an artificial dataset using the given name")
    parser.add_argument('-m', '--model', help="filename of the dataset to train the model on")
    parser.add_argument('-b', '--batch', default=5, type=int, help="batch size of added counterfactuals per round")
    parser.add_argument('-r', '--rounds', default=5, type=int, help="amount of rounds to add counterfactuals and retrain the model")

    args = parser.parse_args()

    if args.dataset:
        create_dataset(args.dataset)

    if args.custom:
        dataset, model, features = load_custom_dataset_model(args.model)
        dataset_revise = copy.copy(dataset)
        dataset_wachter = copy.copy(dataset)

        model_revise = copy.copy(model)
        model_wachter = copy.copy(model)


        mean_features = map(lambda x: f"mean {x}", features)
        cov_features = map(lambda x: f"cov {x}", features)

        columns = ["Count", "KL Divergence"]
        columns.extend(mean_features)
        columns.extend(cov_features)

        df_revise = pd.DataFrame(columns=columns)
        df_wachter = pd.DataFrame(columns=columns)

        revise_factuals = get_factuals(model_revise, dataset_revise)
        wachter_factuals = get_factuals(model_wachter, dataset_wachter)

        revise_base_model_prob = norm_model_prob(model_revise, dataset._df)
        wachter_base_model_prob = norm_model_prob(model_wachter, dataset._df)

        for i in range(args.rounds):
            revise = load_revise(dataset_revise, "custom", model_revise)
            (revise_counterfactuals, count) = run_recourse_method(revise_factuals, revise, "REVISE")
            
            revise_counterfactuals["factual_id"] = revise_factuals.index
            revise_counterfactuals.set_index("factual_id", inplace=True, drop=True)
            revise_counterfactuals.dropna(inplace=True)
            # print(revise_counterfactuals)
            # print(revise_factuals)
            print(count)
            if (count >= args.batch):
                
                samples = revise_counterfactuals.sample(args.batch)
                df = copy.copy(dataset_revise.df)
                print(samples)
                print(samples.index)
                df.index.name = 'factual_id'
                df.loc[samples.index, :] = samples[:]
                print(df.loc[samples.index[0]])
                print(df.loc[samples.index])

                df.to_csv(f"{args.model}-revise-{i}", sep=',', encoding='utf-8', index=False)
            else:
                print("not enough counterfactuals to use with Revise to complete all rounds")

            dataset_revise, model_revise, features = load_custom_dataset_model(f"{args.model}-revise-{i}")

            measurements(df_revise, revise_base_model_prob, norm_model_prob(model_revise, dataset_revise._df), dataset_revise, revise_factuals, revise_counterfactuals, features, name="REVISE")

            revise_factuals = get_factuals(model_revise, dataset_revise)

        # for i in range(args.rounds):
        #     wachter = load_wachter(model_wachter)
        #     (wachter_counterfactuals, count) = run_recourse_method(wachter_factuals, wachter, "Wachter")
            
        #     if count >= args.batch:
        #         wachter_counterfactuals.sample(args.batch)
        #         wachter_factuals.loc[wachter.index, :] = wachter_counterfactuals[:]
        #     else:
        #         print("not enough counterfactuals to use with Wachter to complete all rounds")

        #     model_wachter = MLModelCatalog(dataset, "ann", backend="pytorch", load_online=False)
        #     model_wachter.train(
        #         learning_rate=training_params["lr"],
        #         epochs=training_params["epochs"],
        #         batch_size=training_params["batch_size"],
        #         hidden_size=training_params["hidden_size"]
        #     )

        #     measurements(df_wachter, wachter_base_model_prob, norm_model_prob(model_wachter, wachter_factuals), dataset, wachter_factuals, wachter_counterfactuals, features, name="Wachter")

        #     wachter_factuals = get_factuals(model_revise, dataset)

        print(df_revise)
        # print(df_wachter)

    if not args.custom:
        dataset, model = load_real_model("adult")

    # factuals = run_revise(dataset, model)

    # # START WATCHER

    # run_wachter(dataset, model, factuals)

    print("SUCCESS", file=result)
    result.close()


if __name__ == "__main__":
    main()
