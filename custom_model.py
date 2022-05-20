from carla.models.api import MLModel
from carla.data.catalog.online_catalog import DataCatalog
from typing import Any, List, Union
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

from carla.models.catalog.load_model import save_model
from carla.models.catalog.train_model import train_model, _training_torch, DataFrameDataset

class CustomModel(MLModel):
    
    def __init__(self,
        data: DataCatalog,
        model_type: str,
        backend: str = "pytorch",
        models_home: str = "./models/",
        **kws,
    ):
        self._model_type = model_type
        self._backend = backend
        self._continuous = data.continuous
        self._categorical = data.categorical
        
        super().__init__(data)

        if data._identity_encoding:
            encoded_features = data.categorical
        else:
            encoded_features = list(
                data.encoder.get_feature_names(data.categorical)
            )

        self._catalog = None
        self._feature_input_order = list(
            np.sort(data.continuous + encoded_features)
        )

        self._model = None


    @property
    def model_type(self) -> str:
        """
        Describes the model type
        E.g., ann, linear
        Returns
        -------
        backend : str
            model type
        """
        return self._model_type


    @property
    def raw_model(self) -> Any:
        """
        Returns the raw ML model built on its framework
        Returns
        -------
        ml_model : tensorflow, pytorch, sklearn model type
            Loaded model
        """
        return self._model


    @property
    def backend(self) -> str:
        """
        Describes the type of backend which is used for the ml model.
        E.g., tensorflow, pytorch, sklearn, ...
        Returns
        -------
        backend : str
            Used framework
        """
        return self._backend


    @property
    def feature_input_order(self) -> List[str]:
        """
        Saves the required order of feature as list.
        Prevents confusion about correct order of input features in evaluation
        Returns
        -------
        ordered_features : list of str
            Correct order of input features for ml model
        """
        return self._feature_input_order


    def predict(
        self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]
    ) -> Union[np.ndarray, pd.DataFrame, torch.Tensor]:
        """
        One-dimensional prediction of ml model for an output interval of [0, 1]
        Shape of input dimension has to be always two-dimensional (e.g., (1, m), (n, m))
        Parameters
        ----------
        x : np.Array, pd.DataFrame, or backend specific (tensorflow or pytorch tensor)
            Tabular data of shape N x M (N number of instances, M number of features)
        Returns
        -------
        output : np.ndarray, or backend specific (tensorflow or pytorch tensor)
            Ml model prediction for interval [0, 1] with shape N x 1
        """

        if len(x.shape) != 2:
            raise ValueError(
                "Input shape has to be two-dimensional, (instances, features)."
            )

        if self._backend == "pytorch":
            return self.predict_proba(x)[:, 1].reshape((-1, 1))
        else:
            raise ValueError(
                'Incorrect backend value. Please use only "pytorch" or "tensorflow".'
            )


    def predict_proba(
        self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]
    ) -> Union[np.ndarray, pd.DataFrame, torch.Tensor]:
        """
        Two-dimensional probability prediction of ml model
        Shape of input dimension has to be always two-dimensional (e.g., (1, m), (n, m))
        Parameters
        ----------
        x : np.Array, pd.DataFrame, or backend specific (tensorflow or pytorch tensor)
            Tabular data of shape N x M (N number of instances, M number of features)
        Returns
        -------
        output : np.ndarray, or backend specific (tensorflow or pytorch tensor)
            Ml model prediction with shape N x 2
        """

        # order data (column-wise) before prediction
        x = self.get_ordered_features(x)

        if len(x.shape) != 2:
            raise ValueError("Input shape has to be two-dimensional")

        if self._backend == "pytorch":

            # Keep model and input on the same device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(device)

            if isinstance(x, pd.DataFrame):
                _x = x.values
            elif isinstance(x, torch.Tensor):
                _x = x.clone()
            else:
                _x = x.copy()

            # If the input was a tensor, return a tensor. Else return a np array.
            tensor_output = torch.is_tensor(x)
            if not tensor_output:
                _x = torch.Tensor(_x)

            # input, tensor_output = (
            #     (torch.Tensor(x), False) if not torch.is_tensor(x) else (x, True)
            # )

            _x = _x.to(device)
            output = self._model(_x)

            if tensor_output:
                return output
            else:
                return output.detach().cpu().numpy()
        else:
            raise ValueError(
                'Incorrect backend value. Please use only "pytorch" or "tensorflow".'
            )


    def get_test_accuracy(self):
        # get preprocessed data
        df_test = self.data.df_test

        x_test = df_test[list(set(df_test.columns) - {self.data.target})]
        y_test = df_test[self.data.target]

        prediction = (self.predict(x_test) > 0.5).flatten()
        correct = prediction == y_test
        return correct.mean()

    def get_F1_score(self):
        df_test = self.data.df_test

        x_test = df_test[list(set(df_test.columns) - {self.data.target})]
        y_test = df_test[self.data.target]

        prediction = (self.predict(x_test) > 0.5).flatten()

        return f1_score(y_test, prediction)

    def train(
    self,
    learning_rate=None,
    epochs=None,
    batch_size=None,
    hidden_size=[18, 9, 3],
    n_estimators=5,
    max_depth=5,
    force_train=False
    ):
        """
        Parameters
        ----------
        filename: String
            name of the
        learning_rate: float
            Learning rate for the training.
        epochs: int
            Number of epochs to train for.
        batch_size: int
            Number of samples in each batch
        force_train: bool
            Force training, even if model already exists in cache.
        hidden_size: list[int]
            hidden_size[i] contains the number of nodes in layer [i]
        n_estimators: int
            Number of estimators in forest.
        max_depth: int
            Max depth of trees in the forest.
        Returns
        -------
        """
        layer_string = "_".join([str(size) for size in hidden_size])
        if self.model_type == "linear" or self.model_type == "forest":
            save_name = f"{self.model_type}"
        elif self.model_type == "ann":
            save_name = f"{self.model_type}_layers_{layer_string}"
        else:
            raise NotImplementedError("Model type not supported:", self.model_type)

        df_train = self.data.df_train
        df_test = self.data.df_test

        x_train = df_train[list(set(df_train.columns) - {self.data.target})]
        y_train = df_train[self.data.target]
        x_test = df_test[list(set(df_test.columns) - {self.data.target})]
        y_test = df_test[self.data.target]

        # order data (column-wise) before training
        x_train = self.get_ordered_features(x_train)
        x_test = self.get_ordered_features(x_test)

        # if model loading failed or force_train flag set to true.
        if self._model is None or force_train:
            self._model = train_model(
                self,
                x_train,
                y_train,
                x_test,
                y_test,
                learning_rate,
                epochs,
                batch_size,
                hidden_size,
                n_estimators,
                max_depth,
            )

            save_model(
                model=self._model,
                save_name=save_name,
                data_name=self.data.name,
                backend=self.backend,
            )
        else:
            print("Finetuning model")
            train_dataset = DataFrameDataset(x_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataset = DataFrameDataset(x_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

            _training_torch(
                self._model,
                train_loader,
                test_loader,
                learning_rate,
                epochs,
            )

            save_model(
                model=self._model,
                save_name=save_name,
                data_name=self.data.name,
                backend=self.backend,
            )
