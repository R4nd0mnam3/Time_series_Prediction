import torch
import torch.nn as nn
import numpy as np
import itertools
import classes.tools as tools

class GRU(tools.train_test_split):
    """
    Description :
    GRU-based time series forecasting model with train-test split, hyperparameter tuning, 
    and prediction functionality.
    
    Arguments :
    - dependent_time_series (array-like): Input time series data.
    - train_test_ratio (float, optional): Ratio for splitting the data into train/test sets.
    - split_index (int, optional): Custom index for train-test split.
    - device (str, optional): Computational device ('cpu' or 'cuda').
    """
    def __init__(self, dependent_time_series, train_test_ratio=None, split_index=None, device=None):
        super().__init__(dependent_time_series, train_test_ratio)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.train_test_split()

    class GRUNet(nn.Module):
        """
        Description :
        Defines the GRU neural network architecture used for time series regression.
        
        Arguments :
        - input_size (int): Number of input features.
        - hidden_size (int): Number of hidden units in the GRU layer.
        - num_layers (int): Number of stacked GRU layers.
        """
        def __init__(self, input_size, hidden_size, num_layers):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            """
            Description :
            Defines the forward pass of the GRU network.
            
            Arguments :
            - x (Tensor): Input tensor of shape (batch_size, seq_length, input_size).
            """
            out, _ = self.gru(x)
            out = out[:, -1, :]
            return self.fc(out)

    def create_sequences(self, data, lookback):
        """
        Description :
        Converts a univariate time series into input-output sequences for supervised learning.
        
        Arguments :
        - data (array-like): Input time series data.
        - lookback (int): Number of previous time steps to include in each input sequence.
        """
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i + lookback])
            y.append(data[i + lookback])
        return np.array(X), np.array(y)

    def train_one(self, X_train, y_train, hidden_size, num_layers, l2, lr, epochs):
        """
        Description :
        Trains a single GRU model on the provided training data.
        
        Arguments :
        - X_train (array): Training input sequences.
        - y_train (array): Training target values.
        - hidden_size (int): Number of hidden units in the GRU layer.
        - num_layers (int): Number of stacked GRU layers.
        - l2 (float): L2 regularization coefficient (weight decay).
        - lr (float): Learning rate for the optimizer.
        - epochs (int): Number of training epochs.
        """
        model = self.GRUNet(1, hidden_size, num_layers).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

        X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train).squeeze()
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()

        return model

    def compute_mse(self, model, X_test, y_test):
        """
        Description :
        Computes Mean Squared Error (MSE) between predicted and true values.
        
        Arguments :
        - model (nn.Module): Trained GRU model.
        - X_test (array): Test input sequences.
        - y_test (array): True target values for test data.
        """
        model.eval()
        X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            y_pred = model(X_test).squeeze().cpu().numpy()
            y_true = y_test.cpu().numpy()

        mse = np.mean((y_true - y_pred) ** 2)
        return mse

    def tune_train(self, train_param_grid, model_param):
        """
        Description :
        Tunes learning rate, regularization, and number of epochs using cross-validation
        and early stopping on the training set.
        
        Arguments :
        - train_param_grid (dict): Hyperparameter grid for training (lr, l2, epochs, etc.).
        - model_param (dict): Model configuration parameters (lookback, hidden_size, num_layers).
        """
        lookback = model_param["lookback"]
        hidden_size = model_param["hidden_size"]
        num_layers = model_param["num_layers"]

        epochs_mid, patience, min_delta, n_splits, val_size, min_train = train_param_grid["epochs_mid"], train_param_grid["patience"], train_param_grid["min_delta"],  train_param_grid["n_splits"], train_param_grid["val_size"], train_param_grid["min_train_size"]

        full = np.asarray(self.train_dependent, dtype=float)

        def last_lookback_tail(arr):
            """
            Description :
            Returns the last 'lookback' points of the array (or fewer if shorter).
            
            Arguments :
            - arr (array): Input array to extract context from.
            """
            k = min(len(arr), lookback)
            return arr[-k:]

        split_points = []
        train_end_start = max(min_train, lookback + 1)
        max_train_end = len(full) - val_size
        steps = np.linspace(train_end_start, max_train_end, num=n_splits, dtype=int)
        for train_end in steps:
            val_start = train_end
            val_end = val_start + val_size
            if val_end <= len(full):
                split_points.append((train_end, val_start, val_end))

        def cv_score_for(lr, l2):
            """
            Description :
            Computes average validation MSE across multiple folds for given lr and l2.
            
            Arguments :
            - lr (float): Learning rate for training.
            - l2 (float): L2 regularization coefficient.
            """
            mses = []
            for (train_end, val_start, val_end) in split_points:
                train_arr = full[:train_end]
                val_arr = full[val_start:val_end]
                val_with_ctx = np.concatenate([last_lookback_tail(train_arr), val_arr])

                Xtr, ytr = self.create_sequences(train_arr, lookback)
                if len(ytr) == 0:
                    continue

                Xv_all, yv_all = self.create_sequences(val_with_ctx, lookback)
                if len(yv_all) == 0:
                    continue
                take = min(len(yv_all), val_size)
                Xval, yval = Xv_all[-take:], yv_all[-take:]

                model = self.train_one(Xtr, ytr, hidden_size, num_layers, l2, lr, epochs_mid)
                mses.append(self.compute_mse(model, Xval, yval))

            if not mses:
                return float("inf")
            return float(np.mean(mses))

        best_lr, best_l2, best_cv = None, None, float("inf")
        for lr, l2 in itertools.product(train_param_grid["lrs"], train_param_grid["l2"]):
            cv_mse = cv_score_for(lr, l2)
            if cv_mse < best_cv:
                best_cv = cv_mse
                best_lr, best_l2 = lr, l2

        train_end, val_start, val_end = split_points[-1]
        train_arr = full[:train_end]
        val_arr = full[val_start:val_end]
        val_with_ctx = np.concatenate([last_lookback_tail(train_arr), val_arr])

        Xtr, ytr = self.create_sequences(train_arr, lookback)
        Xv_all, yv_all = self.create_sequences(val_with_ctx, lookback)
        take = min(len(yv_all), val_size)
        Xval, yval = Xv_all[-take:], yv_all[-take:]

        epochs_grid = sorted(train_param_grid["epochs"])
        best_epoch = epochs_grid[0]
        model = self.train_one(Xtr, ytr, hidden_size, num_layers, best_l2, best_lr, best_epoch)
        best_mse = self.compute_mse(model, Xval, yval)

        no_improve = 0
        for ep in epochs_grid[1:]:
            model = self.train_one(Xtr, ytr, hidden_size, num_layers, best_l2, best_lr, ep)
            mse = self.compute_mse(model, Xval, yval)
            if (best_mse - mse) > min_delta:
                best_mse, best_epoch = mse, ep
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        self.training_params = {"lr": best_lr, "l2": best_l2, "epochs": best_epoch}
        print(f"Training results: cv_mse : {best_cv}, val_size : {val_size}, n_splits : {len(split_points)}")

    def tune_model(self, model_param_grid, train_params=None):
        """
        Description :
        Tunes model architecture parameters (lookback, hidden_size, num_layers)
        to minimize test MSE using pre-tuned training hyperparameters.
        
        Arguments :
        - model_param_grid (dict): Grid of model structure parameters to search.
        """
        if train_params is None :
            train_params = self.training_params 
        best_mse = np.inf
        best_params = None

        for lookback, hidden_size, num_layers in itertools.product(
            model_param_grid["lookback"], model_param_grid["hidden_size"], model_param_grid["num_layers"]
        ):
            X_train, y_train = self.create_sequences(self.train_dependent, lookback)
            X_test, y_test = self.create_sequences(np.concatenate([self.train_dependent[-lookback:], self.validation_dependent]), lookback)

            model = self.train_one(X_train, y_train, hidden_size, num_layers, train_params["l2"], train_params["lr"], train_params["epochs"])
            mse = self.compute_mse(model, X_test, y_test)

            print(f"lookback={lookback}, hidden={hidden_size}, layers={num_layers} → MSE={mse:.6f}")

            if mse < best_mse:
                best_mse = mse
                best_params = {
                    "lookback": lookback,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers
                }

        self.model_params = best_params
        print(f"\n✅ Best parameters: lookback={self.model_params['lookback']}, hidden={self.model_params['hidden_size']}, layers={self.model_params['num_layers']}")
        print(f"Best MSE: {best_mse:.6f}")

        X, y = self.create_sequences(self.train_dependent, self.model_params["lookback"])
        self.model = self.train_one(X, y, self.model_params["hidden_size"], train_params["l2"], train_params["lr"], train_params["epochs"])

    def predict(self, data="test"):
        """
        Description :
        Generates predictions using the trained GRU model on either test or train data.
        
        Arguments :
        - data (str): Dataset to predict on ('train' or 'test').
        """
        if data == "validation":
            X, y = self.create_sequences(np.concatenate([self.train_dependent[-self.model_params["lookback"]:], self.validation_dependent]), self.model_params["lookback"])
        elif data == "train":
            X, y = self.create_sequences(self.train_dependent, self.model_params["lookback"])
    
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X).squeeze().cpu().numpy()
        
        return preds, y
    
    def predict_future(self, days):
        """
        Description :
        Predicts future values for a specified number of days using the trained GRU model.

        Arguments :
        - days (int): Number of future time steps to forecast.
        """
        lookback = self.model_params["lookback"]
        self.model.eval()

        # Start from the last 'lookback' known values (train + validation)
        history = np.concatenate([self.train_dependent, self.validation_dependent])[-lookback:].astype(float)
        preds = []

        for _ in range(days):
            x_input = torch.tensor(history[-lookback:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
            
            with torch.no_grad():
                pred = self.model(x_input).item()
            preds.append(pred)

            # Append new prediction for next iteration
            history = np.append(history, pred)

        return np.array(preds)