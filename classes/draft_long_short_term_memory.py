import torch
import torch.nn as nn
import numpy as np
import classes.tools as tools

class LSTM(tools.train_test_split):
    def __init__(self, dependent_time_series, train_test_ratio=0.8, input_size=1, hidden_size=50, num_layers=1, output_size=1, device=None):
        super().__init__(dependent_time_series, train_test_ratio)

        # Device
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Model parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # Define LSTM network
        self.model = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        # Loss and optimizer placeholders
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.fc.parameters()), lr=0.001)

        self.model.to(self.device)
        self.fc.to(self.device)

    def create_sequences(self, data, seq_length):
        """
        Converts a time series into input/output sequences for LSTM.
        """
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:(i+seq_length)]
            y = data[i+seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def fit(self, seq_length=10, epochs=20, lr=0.001):
        """
        Train the LSTM model
        """
        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + list(self.fc.parameters()), lr=lr)

        # Prepare data
        X_train, y_train = self.create_sequences(self.train_dependent, seq_length)
        X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).to(self.device)

        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            out, _ = self.model(X_train)
            out = self.fc(out[:, -1, :])   # last hidden state
            loss = self.criterion(out, y_train)
            loss.backward()
            self.optimizer.step()

            if (epoch+1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    def predict(self, seq_length=10, mode="test"):
        """
        Make predictions for train or test set
        """
        data = self.train_dependent if mode == "train" else self.test_dependent
        X, y = self.create_sequences(data, seq_length)
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(self.device)

        self.model.eval()
        with torch.no_grad():
            out, _ = self.model(X)
            preds = self.fc(out[:, -1, :])
        return preds.cpu().numpy().flatten(), y
