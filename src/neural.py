"""scikit-learn-compatible wrapper so the DNN and KAN drop into the same Pipeline as the others.

What this file does
-------------------
``TorchTabularClassifier`` gives PyTorch networks the usual ``fit`` / ``predict_proba`` /
``predict`` interface, so they can sit inside a sklearn ``Pipeline``, be cloned, calibrated
and cross-validated like Logistic Regression or Random Forest. Training uses AdamW, an internal
validation split for early stopping, learning-rate reduction on plateau, and optional
positive-class weighting. ``arch="dnn"`` builds an MLP, ``arch="kan"`` a Kolmogorov-Arnold
Network (layer definitions live in torch_nets.py). torch is imported lazily, so this file can be
imported without PyTorch installed.
"""

from __future__ import annotations

import copy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class TorchTabularClassifier(ClassifierMixin, BaseEstimator):
    """Binary classifier on dense, preprocessed features. arch = "dnn" or "kan"."""

    def __init__(self, arch="dnn", hidden=(128, 64, 32), dropout=(0.30, 0.20, 0.10),
                 kan_hidden=(8,), grid_size=5, spline_order=3, lr=1e-3, weight_decay=1e-5,
                 batch_size=512, max_epochs=100, patience=8, val_fraction=0.15,
                 class_weight=None, random_state=0):
        self.arch = arch
        self.hidden = hidden
        self.dropout = dropout
        self.kan_hidden = kan_hidden
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.class_weight = class_weight
        self.random_state = random_state

    def _build(self, in_dim: int):
        from . import torch_nets
        if self.arch == "dnn":
            return torch_nets.build_mlp(in_dim, self.hidden, self.dropout)
        if self.arch == "kan":
            return torch_nets.build_kan(in_dim, self.kan_hidden, self.grid_size, self.spline_order)
        raise ValueError(f"Unknown arch {self.arch!r}")

    def fit(self, X, y):
        import torch
        from torch import nn

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        order = rng.permutation(len(X))
        n_val = max(1, int(len(X) * self.val_fraction))
        val_idx, tr_idx = order[:n_val], order[n_val:]
        Xt, yt = torch.from_numpy(X[tr_idx]), torch.from_numpy(y[tr_idx])
        Xv, yv = torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx])

        pos_weight = None
        if self.class_weight == "balanced":
            pos = max(float(yt.sum()), 1.0)
            pos_weight = torch.tensor([(len(yt) - pos) / pos])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        module = self._build(X.shape[1])
        opt = torch.optim.AdamW(module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4, min_lr=1e-5)

        best_loss, best_state, wait = float("inf"), None, 0
        self.history_ = {"train_loss": [], "val_loss": []}  # per epoch, for convergence plots
        for _ in range(self.max_epochs):
            module.train()
            perm = torch.from_numpy(rng.permutation(len(Xt)))
            running = 0.0
            for start in range(0, len(perm), self.batch_size):
                idx = perm[start:start + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(module(Xt[idx]).squeeze(1), yt[idx])
                loss.backward()
                opt.step()
                running += float(loss) * len(idx)

            module.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(module(Xv).squeeze(1), yv))
            self.history_["train_loss"].append(running / len(perm))
            self.history_["val_loss"].append(val_loss)
            sched.step(val_loss)
            if val_loss < best_loss - 1e-6:
                best_loss, best_state, wait = val_loss, copy.deepcopy(module.state_dict()), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        module.load_state_dict(best_state)
        module.eval()
        self.module_ = module
        self.best_val_loss_ = best_loss
        return self

    def predict_proba(self, X):
        import torch
        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        out = []
        with torch.no_grad():
            for start in range(0, len(X), 8192):
                out.append(torch.sigmoid(self.module_(X[start:start + 8192]).squeeze(1)))
        p = torch.cat(out).numpy().astype(float)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.module_.parameters()))
