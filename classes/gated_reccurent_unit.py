import torch
import torch.nn as nn
import numpy as np
import itertools
import classes.tools as tools

class GRU(tools.train_test_split):
    def __init__(self, dependent_time_series, train_test_ratio=None, split_index=None, device=None):
        super().__init__(dependent_time_series, train_test_ratio)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.train_test_split()

    class GRUNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.gru(x)
            out = out[:, -1, :]

            return self.fc(out)

    def create_sequences(self, data, lookback):
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i + lookback])
            y.append(data[i + lookback])

        return np.array(X), np.array(y)

    def train_one(self, X_train, y_train, hidden_size, num_layers, l2, lr, epochs):
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
        model.eval()
        X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            y_pred = model(X_test).squeeze().cpu().numpy()
            y_true = y_test.cpu().numpy()

        mse = np.mean((y_true - y_pred) ** 2)

        return mse

    def tune_train(self, train_param_grid, model_param):
        training_params = {}

        X_train, y_train = self.create_sequences(self.train_dependent, model_param["lookback"])
        X_test, y_test = self.create_sequences(self.test_dependent, model_param["lookback"]) 
                                               
        # Tune Epochs
        l2_init, lr_init = 0.01, 0.001 # Iniating l2 and lr 
        model = self.train_one(X_train, y_train, model_param["hidden_size"], model_param["num_layers"], l2_init, lr_init, train_param_grid["epochs"][0])
        old_mse = self.compute_mse(model, X_test, y_test)


        for epoch in train_param_grid["epochs"][1:] :
            model = self.train_one(X_train, y_train, model_param["hidden_size"], model_param["num_layers"], l2_init, lr_init, epoch)
            
            new_mse = self.compute_mse(model, X_test, y_test)

            if old_mse - new_mse > 0.001 : # If MSE evolves we haven't reached the optimal number of epochs
                old_mse = new_mse
            
            else : # We reached the optimal number of epochs
                training_params["epochs"] = epoch

        
        self.training_params = training_params



    def tune_model(self, model_param_grid):
        best_mse = np.inf
        best_params = None

        for lookback, hidden_size, num_layers in itertools.product(
            model_param_grid["lookback"], model_param_grid["hidden_size"], model_param_grid["num_layers"]
        ):
            X_train, y_train = self.create_sequences(self.train_dependent, lookback)
            X_test, y_test = self.create_sequences(self.test_dependent, lookback) # Not relevant as we won't have this data

            model = self.train_one(X_train, y_train, hidden_size, num_layers, self.training_params["l2"], self.training_params["lr"], self.training_params["epochs"])
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
        print(f"\n✅ Best parameters: lookback={self.model_params['lookback']}, hidden={self.model_params['hidden_size']}, layers={self.model_params['num_layers']}, l2={self.model_params['l2']}")
        print(f"Best MSE: {best_mse:.6f}")

        # Final training with best parameters
        X, y = self.create_sequences(self.train_dependent, self.model_params["lookback"])
        self.model = self.train_one(X, y, self.model_params["hidden_size"], self.model_params["num_layers"], self.training_params["l2"], self.training_params["lr"], self.training_params["epochs"])

    def predict(self, data="test"):
        if data == "test":
            X, y = self.create_sequences(self.test_dependent, self.best_params["lookback"])
            print(y)
        elif data == "train":
            X, y = self.create_sequences(self.train_dependent, self.best_params["lookback"])
    
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X).squeeze().cpu().numpy()
        
        return preds, y