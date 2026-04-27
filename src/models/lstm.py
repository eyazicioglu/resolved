import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from torch.utils.data import DataLoader, Dataset


class _MarketDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, X_static: np.ndarray, y: np.ndarray) -> None:
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_static = torch.tensor(X_static, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_seq[idx], self.X_static[idx], self.y[idx]


class _LSTMNet(nn.Module):
    def __init__(
        self,
        seq_dim: int,
        static_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        fc_in = hidden_size + static_dim
        self.head = nn.Sequential(
            nn.Linear(fc_in, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(seq)
        last_hidden = h_n[-1]  # take top-layer final hidden state
        combined = torch.cat([last_hidden, static], dim=1)
        return self.head(combined).squeeze(1)


def pack_X(X_seq: np.ndarray, X_static: np.ndarray) -> np.ndarray:
    """Flatten X_seq time axis and concat with X_static → (n, seq_len*seq_dim + static_dim)."""
    n = X_seq.shape[0]
    return np.concatenate([X_seq.reshape(n, -1), X_static], axis=1)


class LSTMClassifier(ClassifierMixin, BaseEstimator):
    """sklearn-compatible LSTM for market trajectory classification.

    Pass X packed via pack_X(X_seq, X_static); the classifier recovers the original
    shapes internally using seq_dim and static_dim.
    """

    def __init__(
        self,
        seq_dim: int,
        static_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 1024,
        patience: int = 5,
        pos_weight: float | None = None,
        device: str = "cpu",
    ) -> None:
        self.seq_dim = seq_dim
        self.static_dim = static_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.pos_weight = pos_weight  # None = auto (neg/pos ratio)
        self.device = device  # kept as str so BaseEstimator.get_params/set_params work

    def _split_X(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        seq_flat = X.shape[1] - self.static_dim
        seq_len = seq_flat // self.seq_dim
        X_seq = X[:, :seq_flat].reshape(-1, seq_len, self.seq_dim)
        X_static = X[:, seq_flat:]
        return X_seq, X_static

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMClassifier":
        self.classes_ = np.array([0, 1])
        X_seq, X_static = self._split_X(X)
        _device = torch.device(self.device)

        self.model_ = _LSTMNet(
            self.seq_dim, self.static_dim, self.hidden_size, self.num_layers
        ).to(_device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        neg = int((y == 0).sum())
        pos = int((y == 1).sum())
        pw = self.pos_weight if self.pos_weight is not None else neg / pos
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pw], dtype=torch.float32).to(_device)
        )

        dataset = _MarketDataset(X_seq, X_static, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        best_loss = float("inf")
        no_improve = 0

        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for seq_b, static_b, y_b in loader:
                seq_b = seq_b.to(_device)
                static_b = static_b.to(_device)
                y_b = y_b.to(_device)
                optimizer.zero_grad()
                logits = self.model_(seq_b, static_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(y_b)
            epoch_loss /= len(dataset)

            if epoch_loss < best_loss - 1e-5:
                best_loss = epoch_loss
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 5 == 0:
                print(f"  epoch {epoch + 1:3d}/{self.epochs}  loss={epoch_loss:.6f}")

            if no_improve >= self.patience:
                print(f"  early stop at epoch {epoch + 1}  best_loss={best_loss:.6f}")
                break

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("model not fitted")
        X_seq, X_static = self._split_X(X)
        _device = torch.device(self.device)
        self.model_.eval()
        dataset = _MarketDataset(X_seq, X_static, np.zeros(len(X_seq)))
        loader = DataLoader(dataset, batch_size=self.batch_size * 4, shuffle=False, num_workers=0)
        probs = []
        with torch.no_grad():
            for seq_b, static_b, _ in loader:
                seq_b = seq_b.to(_device)
                static_b = static_b.to(_device)
                p = torch.sigmoid(self.model_(seq_b, static_b)).cpu().numpy()
                probs.append(p)
        prob = np.concatenate(probs)
        return np.column_stack([1 - prob, prob])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_lstm(
    X_seq_train: np.ndarray,
    X_static_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 24,
    cv: int = 3,
    epochs: int = 30,
    device: str = "cpu",
) -> tuple[LSTMClassifier, dict]:
    """RandomizedSearchCV over batch_size ∈ {512,1024,2048} and lr ∈ {1e-3,1e-4,1e-5}.

    n_iter=9 covers all 9 combinations (full grid); reduce for faster sweeps.
    n_jobs=1 is required — PyTorch manages its own threading.
    Returns (best_estimator, cv_results_).
    """
    X = pack_X(X_seq_train, X_static_train)

    base = LSTMClassifier(
        seq_dim=X_seq_train.shape[2],
        static_dim=X_static_train.shape[1],
        hidden_size=64,
        num_layers=2,
        epochs=epochs,
        patience=5,
        device=device,
    )

    param_dist = {
        "lr": [1e-3, 1e-4, 1e-5],
        "batch_size": [1024, 2048],
        "pos_weight": [0.5, 1.0, 1.5, 2.0],
    }

    scoring = {
        "roc_auc": "roc_auc",
        "precision_yes": make_scorer(precision_score, pos_label=1, zero_division=0),
        "recall_yes": make_scorer(recall_score, pos_label=1, zero_division=0),
    }

    search = RandomizedSearchCV(
        base,
        param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        refit="roc_auc",
        verbose=2,
        n_jobs=1,
        random_state=42,
    )
    search.fit(X, y_train)
    return search.best_estimator_, search.cv_results_
