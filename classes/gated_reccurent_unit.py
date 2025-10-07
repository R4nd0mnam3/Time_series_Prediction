import torch
import torch.nn as nn
import numpy as np
import itertools
import classes.tools as tools

class GRU(tools.train_test_split):
    def __init__(self, dependent_time_series, train_test_ratio=None, split_index=None, epochs=50, lr=0.001, device=None):
        super().__init__(dependent_time_series, train_test_ratio)
        self.epochs = epochs
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

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

    def train_one(self, X_train, y_train, hidden_size, num_layers, l2):
        model = self.GRUNet(1, hidden_size, num_layers).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=l2)

        X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        for _ in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_train).squeeze()
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()

        return model

    def compute_mse(self, model, X_val, y_val):
        model.eval()
        X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1).to(self.device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            y_pred = model(X_val).squeeze().cpu().numpy()
            y_true = y_val.cpu().numpy()

        mse = np.mean((y_true - y_pred) ** 2)
        return mse

    def tune(self, param_grid):
        self.train_test_split()
        series = self.train_dependent
        best_mse = np.inf
        best_params = None

        for lookback, hidden_size, num_layers, l2 in itertools.product(
            param_grid["lookback"], param_grid["hidden_size"], param_grid["num_layers"], param_grid["l2"]
        ):
            X, y = self.create_sequences(series, lookback)
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            model = self.train_one(X_train, y_train, hidden_size, num_layers, l2)
            mse = self.compute_mse(model, X_val, y_val)

            print(f"lookback={lookback}, hidden={hidden_size}, layers={num_layers}, l2={l2} → MSE={mse:.6f}")

            if mse < best_mse:
                best_mse = mse
                best_params = {
                    "lookback": lookback,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "l2" : l2
                }

        self.best_params = best_params
        print(f"\n✅ Best parameters: lookback={best_params['lookback']}, hidden={best_params['hidden_size']}, layers={best_params['num_layers']}, l2={best_params['num_layers']}")
        print(f"Best MSE: {best_mse:.6f}")

        # Final training with best parameters
        X, y = self.create_sequences(self.train_dependent, best_params["lookback"])
        self.model = self.train_one(X, y, best_params["hidden_size"], best_params["num_layers"], best_params["l2"])

    def predict(self, data="test"):
        if data == "test":
            X, y = self.create_sequences(self.test_dependent, self.best_params["lookback"])
        else:
            X, y = self.create_sequences(self.train_dependent, self.best_params["lookback"])
    
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X).squeeze().cpu().numpy()
        return preds, y