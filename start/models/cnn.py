"""
1D CNN model for intraday pattern recognition.

Captures local temporal patterns (momentum signatures, reversal patterns)
using convolutional filters over sliding windows.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from start.models.lstm import TimeSeriesDataset  # Reuse sliding window dataset
from start.utils.logger import get_logger

logger = get_logger(__name__)


class CNN1DNet(nn.Module):
    """1D CNN: Conv(16, k=5) → Conv(8, k=3) → AdaptiveAvgPool → FC."""

    def __init__(self, input_size: int):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: (batch, seq_len, features) → need (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # (batch, 8)
        out = self.fc(x)
        return out.squeeze(-1)


class CNNModel:
    """
    1D CNN wrapper with train/predict interface matching classical models.
    """

    name = "cnn"

    def __init__(
        self,
        seq_len: int = 20,
        batch_size: int = 64,
        max_epochs: int = 50,
        patience: int = 5,
        lr: float = 1e-3,
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr = lr
        self.model = None
        self.device = torch.device("cpu")
        self._feature_mean = None
        self._feature_std = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CNNModel":
        """Train CNN with early stopping."""
        X_arr = X.values.astype(np.float32)
        y_arr = y.values.astype(np.float32)

        # Normalize
        self._feature_mean = X_arr.mean(axis=0)
        self._feature_std = X_arr.std(axis=0) + 1e-8
        X_arr = (X_arr - self._feature_mean) / self._feature_std

        # Train/val split
        split = int(len(X_arr) * 0.8)
        train_ds = TimeSeriesDataset(X_arr[:split], y_arr[:split], self.seq_len)
        val_ds = TimeSeriesDataset(X_arr[split:], y_arr[split:], self.seq_len)

        if len(train_ds) < self.batch_size or len(val_ds) < 1:
            logger.warning("[cnn] Not enough data for training, skipping")
            return self

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        # Build model
        n_features = X_arr.shape[1]
        self.model = CNN1DNet(n_features)
        self.model.to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.max_epochs):
            self.model.train()
            train_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    out = self.model(xb)
                    val_loss += criterion(out, yb).item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                logger.info(f"[cnn] Early stop at epoch {epoch+1}, best val loss: {best_val_loss:.4f}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), 0.5)

        X_arr = X.values.astype(np.float32)
        X_arr = (X_arr - self._feature_mean) / self._feature_std

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for i in range(self.seq_len, len(X_arr)):
                x_seq = torch.FloatTensor(
                    X_arr[i - self.seq_len : i]
                ).unsqueeze(0).to(self.device)
                out = self.model(x_seq)
                prob = torch.sigmoid(out).item()
                predictions.append(prob)

        pad = [0.5] * min(self.seq_len, len(X))
        all_preds = pad + predictions

        if len(all_preds) < len(X):
            all_preds.extend([0.5] * (len(X) - len(all_preds)))

        return np.array(all_preds[: len(X)])
