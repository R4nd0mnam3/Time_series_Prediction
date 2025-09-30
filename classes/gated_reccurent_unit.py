import torch
import torch.nn as nn
import numpy as np
import classes.tools as tools

class GRU(tools.train_test_split):
    def __init__(self, dependent_time_series, train_test_ratio=0.8, lookback=10, hidden_size=64, num_layers=2, epochs=50, lr=0.001, device=None):
        super().__init__(dependent_time_series, train_test_ratio)
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # placeholders
        self.model = None
        self.train_X, self.train_y = None, None
        self.test_X, self.test_y = None, None

    def create_sequences(self, data, lookback):
        """
        Convert 1D time series into sequences of shape (samples, lookback, 1)
        """
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback])
            y.append(data[i+lookback])
        return np.array(X), np.array(y)

    def prepare_data(self):
        """
        Prepares train/test datasets with lookback windowing
        """
        self.train_test_split()
        self.train_X, self.train_y = self.create_sequences(self.train_dependent, self.lookback)
        self.test_X, self.test_y = self.create_sequences(self.test_dependent, self.lookback)

        # reshape for GRU: (samples, timesteps, features)
        self.train_X = torch.tensor(self.train_X, dtype=torch.float32).unsqueeze(-1).to(self.device)
        self.train_y = torch.tensor(self.train_y, dtype=torch.float32).to(self.device)
        self.test_X = torch.tensor(self.test_X, dtype=torch.float32).unsqueeze(-1).to(self.device)
        self.test_y = torch.tensor(self.test_y, dtype=torch.float32).to(self.device)

    class GRUNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.gru(x)  # out: (batch, seq_len, hidden_size)
            out = out[:, -1, :]   # last time step
            return self.fc(out)

    def build_model(self):
        self.model = self.GRUNet(input_size=1, hidden_size=self.hidden_size, num_layers=self.num_layers).to(self.device)

    def train(self):
        self.prepare_data()
        self.build_model()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(self.train_X)
            loss = criterion(outputs.squeeze(), self.train_y)
            loss.backward()
            optimizer.step()

            if (epoch+1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.6f}")

    def predict(self, data="test"):
        """
        Predict on 'test' or 'train' dataset
        """
        self.model.eval()
        if data == "test":
            with torch.no_grad():
                preds = self.model(self.test_X).cpu().numpy()
            return preds, self.test_y.cpu().numpy()
        elif data == "train":
            with torch.no_grad():
                preds = self.model(self.train_X).cpu().numpy()
            return preds, self.train_y.cpu().numpy()
        else:
            raise ValueError("data must be 'train' or 'test'")
