"""
LSTM model for temporal pattern recognition.

Small architecture (32 hidden units, 1 layer) designed for 8GB RAM constraint.
Uses 20-bar sliding windows as input sequences.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from start.utils.logger import get_logger

logger = get_logger(__name__)


class TimeSeriesDataset(Dataset):
    """Sliding window dataset for sequence models."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 20):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]
        y_val = self.y[idx + self.seq_len]
        return x_seq, y_val


class LSTMNet(nn.Module):
    """LSTM network: 1 layer, 32 hidden → FC → sigmoid."""

    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take last timestep output
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out.squeeze(-1)


class LSTMModel:
    """
    LSTM wrapper with train/predict interface matching classical models.
    """

    name = "lstm"

    def __init__(
        self,
        hidden_size: int = 32,
        num_layers: int = 1,
        seq_len: int = 20,
        batch_size: int = 64,
        max_epochs: int = 50,
        patience: int = 5,
        lr: float = 1e-3,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr = lr
        self.model = None
        self.device = torch.device("cpu")
        self._feature_mean = None
        self._feature_std = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMModel":
        """Train LSTM with early stopping on validation split."""
        X_arr = X.values.astype(np.float32)
        y_arr = y.values.astype(np.float32)

        # Normalize features
        self._feature_mean = X_arr.mean(axis=0)
        self._feature_std = X_arr.std(axis=0) + 1e-8
        X_arr = (X_arr - self._feature_mean) / self._feature_std

        # Train/validation split (last 20% for early stopping)
        split = int(len(X_arr) * 0.8)
        train_ds = TimeSeriesDataset(X_arr[:split], y_arr[:split], self.seq_len)
        val_ds = TimeSeriesDataset(X_arr[split:], y_arr[split:], self.seq_len)

        if len(train_ds) < self.batch_size or len(val_ds) < 1:
            logger.warning("[lstm] Not enough data for training, skipping")
            return self

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        # Build model
        n_features = X_arr.shape[1]
        self.model = LSTMNet(n_features, self.hidden_size, self.num_layers)
        self.model.to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.max_epochs):
            # Train
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

            # Validate
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
                logger.info(f"[lstm] Early stop at epoch {epoch+1}, best val loss: {best_val_loss:.4f}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary direction (0/1)."""
        probs = self.predict_proba(X)
        return (probs > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of upward movement."""
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

        # Pad the first seq_len predictions with 0.5
        pad = [0.5] * min(self.seq_len, len(X))
        all_preds = pad + predictions

        # Trim or pad to match input length
        if len(all_preds) < len(X):
            all_preds.extend([0.5] * (len(X) - len(all_preds)))

        return np.array(all_preds[: len(X)])
