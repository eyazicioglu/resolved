import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _FeedforwardNN(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class NNClassifier:
    """sklearn-compatible wrapper for _FeedforwardNN"""

    def __init__(
        self,
        input_dim: int,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 2048,
        patience: int = 5,
        device: str = "cpu",
    ) -> None:
        self.input_dim = input_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = torch.device(device)
        self.model: _FeedforwardNN | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NNClassifier":
        self.model = _FeedforwardNN(self.input_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float("inf")
        no_improve = 0

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(dataset)

            if epoch_loss < best_loss - 1e-5:
                best_loss = epoch_loss
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch + 1:3d}/{self.epochs}  loss={epoch_loss:.6f}")

            if no_improve >= self.patience:
                print(f"  early stop at epoch {epoch + 1}  best_loss={best_loss:.6f}")
                break

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model not fitted")
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            prob = self.model(X_t).cpu().numpy()
        return np.column_stack([1 - prob, prob])


def train_neural_network(X_train: np.ndarray, y_train: np.ndarray) -> NNClassifier:
    """train feedforward NN: 64→32→1 with ReLU activations and sigmoid output"""
    input_dim = X_train.shape[1]
    clf = NNClassifier(
        input_dim=input_dim,
        lr=1e-3,
        epochs=100,
        batch_size=2048,
        patience=8,
    )
    clf.fit(X_train, y_train)
    return clf
