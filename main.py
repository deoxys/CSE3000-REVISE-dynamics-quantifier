import warnings
import os
import datetime
import time
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from carla.data.catalog import OnlineCatalog
from carla import MLModelCatalog
from carla.recourse_methods import Revise
from carla.recourse_methods import Wachter
from carla.models.negative_instances import predict_negative_instances

from scipy.special import rel_entr
from scipy.stats import wasserstein_distance
import numpy as np

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
def get_factuals(model, dataset, filename = datetime.datetime.now(), random=True, sample=None):
    factuals = None
    if random:
        if sample is None:
            factuals = predict_negative_instances(model, dataset.df).reset_index(drop=True)
        else: 
            factuals = predict_negative_instances(model, dataset.df).sample(sample).reset_index(drop=True)

    else:
        factuals = predict_negative_instances(model, dataset.df).iloc[:10].reset_index(drop=True)
    factuals.to_csv("{} factuals.csv".format(filename), sep='\t', encoding='utf-8')
    return factuals


# generate counterfactual examples
def get_counterfactuals(recourse_method, factuals, filename, timing = False):
    start = time.time()

    counterfactuals = recourse_method.get_counterfactuals(factuals)
    numberOfNaNs = counterfactuals.iloc[:, 0].isna().sum()
    if timing:
        end = time.time()
        print("It took {} seconds to find {} counterfactuals for REVISE".format(counterfactuals.shape[0]-numberOfNaNs, end-start))
    
    counterfactuals.to_csv("{} counterfactuals_REVISE.csv".format(filename), sep='\t', encoding='utf-8')
    
    # Sort the columns of the counterfactuals so that it has the same order as that
    # of the factuals. This makes it possible to combine the 2 dataframes.
    counterfactuals = counterfactuals.reindex(sorted(counterfactuals.columns), axis=1)
    counterfactuals = counterfactuals.combine_first(factuals)

    return counterfactuals

def norm_model_prob(model, input):
    output = np.nan_to_num(model.predict(input))
    # normalize output vector
    output /= np.sum(output)
    return output

def kl_divergence(model, factuals, counterfactuals):
    # retrieve the KL Divergence by summing the relative entropy of two probabilities
    vec = rel_entr(norm_model_prob(model, factuals), norm_model_prob(model, counterfactuals))
    return np.sum(vec)


def run_revise(dataset, model):
    recourse_method = load_Revise(dataset, "adult", model)

    ct = datetime.datetime.now()
    factuals = get_factuals(model, dataset, ct, random=False)
    counterfactuals = get_counterfactuals(recourse_method, factuals, ct, timing=True)

    kl_div = kl_divergence(model, factuals, counterfactuals)
    print(norm_model_prob(model, factuals).shape)
    was_dist = wasserstein_distance(norm_model_prob(model, factuals).flatten(), norm_model_prob(model, counterfactuals).flatten())
    print(was_dist)
    print("The KL divergence of REVISE in this dataset is: {}".format(kl_div))

    return factuals


dataset, model = load_real_model("adult")

factuals = run_revise(dataset, model)

# START WATCHER

hyperparams = {
    "loss_type": "BCE", 
    "binary_cat_features": False
}
recourse_method = Wachter(model, hyperparams)
df_cfs = recourse_method.get_counterfactuals(factuals)

# Sort the columns of the counterfactuals so that it has the same order as that
# of the factuals. This makes it possible to combine the 2 dataframes.
df_cfs = df_cfs.reindex(sorted(df_cfs.columns), axis=1)
# print(counterfactuals)
df_cfs = df_cfs.combine_first(factuals)

kl_div = kl_divergence(model, factuals, df_cfs)
print("The KL divergence of WACHTER in this dataset is: {}".format(kl_div))

print("SUCCESS")