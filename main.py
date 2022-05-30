
from dis import dis
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
from custom_linear_model import CustomLinearModel
from synthetic_dataset import SyntheticDatasetGenerator as SDG
from measurements import Measurements
from carla.recourse_methods import Revise
from carla.recourse_methods import Wachter
from carla.models.negative_instances import predict_negative_instances

from scipy.special import rel_entr
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator

from carla.models.catalog.ANN_TORCH import AnnModel as ann_torch
from carla.models.catalog.Linear_TORCH.model_linear import LinearModel as linear

# from custom_model_ann import CustomAnnModel as ann_torch

from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier

import copy
import argparse

import logging

logger = logging.Logger("qlogger")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh = logging.FileHandler("test.log")
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

# reads the csv containing the data, assumes that the last column is the target and not a feature
def read_dataset(filename):
    dataset = pd.read_csv(filename)
    features = dataset.columns.tolist()[0:-1]
    dataset.index.name = 'factual_id'
    return (dataset, features)


def load_custom_dataset_model(filename, model_type):

    dataset, features = read_dataset(filename)
    print(dataset)
    dataset = CsvCatalog(file_path=filename, continuous=features, categorical=[], immutables=[], target="target")

    print(dataset._df)

    hidden_layers = [4]

    net = None 
    model = None

    parameters = {
        'lr': [0.005, 0.01, 0.02, 0.05, 0.1],
        'max_epochs': [10, 20, 30, 40, 50],
        'batch_size': [1, 5, 10, 20, 40]
    }

    if model_type == 'linear':

        net = NeuralNetClassifier(
            module=linear,
            module__dim_input=len(dataset._df_test.columns)-1,
            # module__hidden_layers=hidden_layers,
            module__num_of_classes=2,
            lr=0.01,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
        )
        model = CustomLinearModel(dataset, model_type="linear", backend="pytorch", load_online=False, cache=False)

    elif model_type == 'ann1':
        net = NeuralNetClassifier(
            module=ann_torch,
            module__input_layer=len(dataset._df_test.columns)-1,
            module__hidden_layers=hidden_layers,
            module__num_of_classes=2,
            lr=0.01,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
        )
        model = CustomModel(dataset, model_type="linear", backend="pytorch", load_online=False, cache=False, hidden_size=hidden_layers)
    
    elif model_type == 'ann2':
        hidden_layers = [8, 4]
        net = NeuralNetClassifier(
            module=ann_torch,
            module__input_layer=len(dataset._df_test.columns)-1,
            module__hidden_layers=hidden_layers,
            module__num_of_classes=2,
            lr=0.01,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
        )
        model = CustomModel(dataset, model_type="linear", backend="pytorch", load_online=False, cache=False, hidden_size=hidden_layers)

    global gs
    gs = GridSearchCV(estimator=net, param_grid=parameters, cv=2, scoring='accuracy', verbose=0, n_jobs=-1)
    
    X = dataset._df_train[list(set(dataset._df_train.columns) - {dataset.target})]
    y = dataset._df_train[dataset.target]

    gs.fit(np.array(X, dtype=np.float32), y)

    logger.debug(f"Best parameters set found on development set: {gs.best_params_}")

    # logger.debug(classification_report(y_true, y_pred))

    model.train(
        learning_rate=gs.best_params_['lr'],
        epochs=gs.best_params_['max_epochs'],
        batch_size=gs.best_params_['batch_size'],
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
            "train": True,
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
def get_factuals(model, dataset):
    factuals = predict_negative_instances(model, dataset.df)
    return factuals


def run_recourse_method(factuals, recourse_method, recourse_name):
    start = time.time()
    counterfactuals = recourse_method.get_counterfactuals(factuals)
    end = time.time()
    
    count = counterfactuals.dropna(inplace=False)
    # print(count)
    count = counterfactuals.shape[0]
    logger.debug(f"It took {end-start:.2f} seconds to find {count} counterfactuals for {recourse_name}")
    return (counterfactuals, count)



def run_rounds(df_res, measurement, rounds, batch, model, dataset, factuals, recourse_function, recourse_name, filename):    
    for i in range(1, rounds+1):
        recourse = recourse_function(dataset, "custom", model)

        try:
            (counterfactuals, count) = run_recourse_method(factuals, recourse, recourse_name)
        except ValueError:
            logger.error(f"error with the retrieval of counterfactuals")

        counterfactuals["factual_id"] = factuals.index
        counterfactuals.set_index("factual_id", inplace=True, drop=True)
        counterfactuals.dropna(inplace=True)

        if (counterfactuals.shape[0] >= batch):
            
            samples = counterfactuals.sample(batch)
            df = dataset._df
            logger.debug(f"Counterfactuals added: {samples.shape[0]}")
            df.index.name = "factual_id"
            df.loc[samples.index, :] = samples[:]


            df.to_csv(f"{filename}-{recourse_name}-{i}", sep=',', encoding='utf-8', index=False)
        else:
            logger.debug(f"not enough counterfactuals to use with {recourse_name} to complete all rounds {counterfactuals.shape}")


        dataset, features = read_dataset(f"{filename}-{recourse_name}-{i}")
        dataset = CsvCatalog(file_path=f"{filename}-{recourse_name}-{i}", continuous=features, categorical=[], immutables=[], target="target")
        model.data = dataset
        model.train(
            learning_rate=gs.best_params_['lr'],
            epochs=gs.best_params_['max_epochs'],
            batch_size=gs.best_params_['batch_size'],
        )

        res = measurement.measure(model, dataset, filename, counterfactuals, features, round=i, name=recourse_name)
        df_res = df_res.append(res, ignore_index=True)
        factuals = get_factuals(model, dataset)
    
    return df_res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--custom', help="use a custom dataset", action="store_true")
    parser.add_argument('-d', '--dataset', help="create an artificial dataset using the given name")
    parser.add_argument('-n', '--num', default=200, type=int, help="number of datapoints for custom dataset")
    parser.add_argument('-m', '--model', help="filename of the dataset to train the model on")
    parser.add_argument('-t', '--type', help="model type (linear, ann1 or ann2)")
    parser.add_argument('-b', '--batch', default=5, type=int, help="batch size of added counterfactuals per round")
    parser.add_argument('-r', '--rounds', default=5, type=int, help="amount of rounds to add counterfactuals and retrain the model")

    args = parser.parse_args()

    if args.dataset:
        sdg = SDG(logger)
        sdg.generate(args.dataset, args.num)

    if args.custom:

        start = time.time()

        logger.debug("process started")

        dataset, model, features = load_custom_dataset_model(args.model, args.type)
        logger.debug(hash(model))

        df_revise = pd.DataFrame()
        df_wachter = pd.DataFrame()
        measure_revise = Measurements(logger, model, dataset)
        measure_wachter = Measurements(logger, model, dataset)

        res = measure_revise.measure(model, dataset, args.model, pd.DataFrame(), features, round=0, name="Start")

        df_revise = df_revise.append(res, ignore_index=True)
        df_wachter = df_wachter.append(res, ignore_index=True)

        logger.debug("starting revise")
        dataset_revise = copy.deepcopy(dataset)
        revise_model = copy.deepcopy(model)
        revise_factuals = get_factuals(revise_model, dataset_revise)
        df_revise = run_rounds(df_revise, measure_revise, args.rounds, args.batch, revise_model, dataset_revise, revise_factuals, load_revise, "REVISE", args.model)
        df_revise.to_csv(f'{args.model}_df_revise.csv', sep=',', encoding='utf-8', index=False)
        logger.debug("done with revise")

        logger.debug("starting wachter")
        wachter_dataset = copy.deepcopy(dataset)
        wachter_model = copy.deepcopy(model)
        wachter_factuals = get_factuals(wachter_model, wachter_dataset)
        df_wachter = run_rounds(df_wachter, measure_wachter, args.rounds, args.batch, wachter_model, wachter_dataset, wachter_factuals, load_wachter, "Wachter", args.model)
        df_wachter.to_csv(f'{args.model}_df_wachter.csv', sep=',', encoding='utf-8', index=False)
        logger.debug("done with wachter")

        end = time.time()

        logger.debug(hash(wachter_model))
        logger.debug(hash(revise_model))
        logger.debug(hash(model))
        logger.debug(f"whole process took: {end-start} seconds.")

    if not args.custom:
        dataset, model = load_real_model("adult")
        # TODO: almost the same code with the custom dataset


if __name__ == "__main__":
    main()
