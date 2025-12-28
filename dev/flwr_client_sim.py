# flwr_client_sim.py

import flwr as fl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from mlp_model import MLPRegressor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- UTILS: TORCH <-> NUMPY PARAMS ----------
def get_numpy_params(model: nn.Module):
    return [p.detach().cpu().numpy() for p in model.parameters()]


def set_numpy_params(model: nn.Module, params_list):
    sd = model.state_dict()
    new_sd = {}
    idx = 0
    for k, v in sd.items():
        new_sd[k] = torch.tensor(params_list[idx], dtype=v.dtype)
        idx += 1
    model.load_state_dict(new_sd)


# ---------- CLIENT CLASS FOR SIMULATION ----------
class FlowerSimClient(fl.client.NumPyClient):
    def __init__(self, user_csv: str, input_dim: int, local_epochs: int, batch_size: int, lr: float):
        self.user_csv = user_csv
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr

        # Create model
        self.model = MLPRegressor(input_dim=input_dim, hidden_dim=64, output_dim=3).to(DEVICE)
        self.loss_fn = nn.MSELoss()

        # Load data for this user
        self._load_data()

    # Load X, y for each user
    def _load_data(self):
        df = pd.read_csv(self.user_csv)

        target_cols = ["A4", "A5", "A6a"]
        X_cols = [c for c in df.columns if c not in target_cols + ["userid", "bin", "token"]]

        X = df[X_cols].values.astype(np.float32)
        y = df[target_cols].values.astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.scaler = StandardScaler().fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
        self.X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        self.y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

    # ---------- Flower Methods ----------
    def get_parameters(self, config):
        return get_numpy_params(self.model)

    def fit(self, parameters, config):
        set_numpy_params(self.model, parameters)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.local_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.loss_fn(preds, yb)
                loss.backward()
                optimizer.step()

        # Evaluate
        mse = self.evaluate(parameters, {})[0]

        return get_numpy_params(self.model), len(self.X_train), {"mse": float(mse)}

    def evaluate(self, parameters, config):
        set_numpy_params(self.model, parameters)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(self.X_test)
            loss = self.loss_fn(preds, self.y_test).item()

        return float(loss), len(self.X_test), {"mse": float(loss)}
