"""
Alternative Denoising Autoencoder (DAE v2) for tabular features.

Why this exists (vs the v1 DAE in `models.py`):
- Uses robust scaling (median / IQR) instead of mean/std (better with outliers).
- Computes reconstruction loss only on observed (non-NaN) inputs to avoid
  training instability when many indicators are missing early in the series.
- Uses random feature masking + Gaussian noise ("masked denoising") to learn
  more stable representations.
- Uses LayerNorm (more stable than BatchNorm for non-i.i.d. batches).

This module is intentionally standalone so it can be swapped in/out without
touching the rest of the ML stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
import warnings

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    PYTORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYTORCH_AVAILABLE = False


@dataclass
class DAEV2Metrics:
    best_monitor_loss: float
    best_train_loss: float
    best_val_loss: Optional[float]
    epochs_ran: int
    internal_val_used: bool
    dae_fit_train_samples: int
    dae_fit_val_samples: int

    def to_dict(self) -> Dict:
        return {
            'best_monitor_loss': float(self.best_monitor_loss),
            'best_train_loss': float(self.best_train_loss),
            'best_val_loss': float(self.best_val_loss) if self.best_val_loss is not None else None,
            'epochs_ran': int(self.epochs_ran),
            'internal_val_used': bool(self.internal_val_used),
            'dae_fit_train_samples': int(self.dae_fit_train_samples),
            'dae_fit_val_samples': int(self.dae_fit_val_samples),
        }


class DenoisingAutoencoderV2:
    """
    Masked denoising autoencoder for tabular (feature) data.

    Interface is intentionally similar to the v1 DAE in `models.py`:
    - fit(X_train, X_val=None)
    - transform(X)
    - save(path) / load(path)
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        *,
        mask_prob: float = 0.15,
        noise_std: float = 0.05,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 512,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        clip_value: Optional[float] = 8.0,
        device: Optional[str] = None,
    ):
        if not PYTORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for DAE v2. Install with: pip install torch"
            )

        self.input_dim = int(input_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.hidden_dims = list(hidden_dims) if hidden_dims is not None else [256, 128]

        self.mask_prob = float(mask_prob)
        self.noise_std = float(noise_std)
        self.dropout_rate = float(dropout_rate)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.early_stopping_patience = int(early_stopping_patience)
        self.clip_value = float(clip_value) if clip_value is not None else None

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = self._build_model()
        self.model.to(self.device)

        # Robust scaler params (median/IQR).
        self.feature_center: Optional[np.ndarray] = None
        self.feature_scale: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False
        self.training_history: List[Dict] = []
        self._warned_feature_mismatch: bool = False

    def _build_model(self) -> nn.Module:
        def _block(in_dim: int, out_dim: int) -> List[nn.Module]:
            return [
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.SiLU(),
                nn.Dropout(self.dropout_rate),
            ]

        encoder_layers: List[nn.Module] = []
        prev_dim = self.input_dim
        for h in self.hidden_dims:
            encoder_layers.extend(_block(prev_dim, int(h)))
            prev_dim = int(h)

        encoder_layers.append(nn.Linear(prev_dim, self.bottleneck_dim))
        encoder_layers.append(nn.LayerNorm(self.bottleneck_dim))
        encoder_layers.append(nn.SiLU())

        decoder_layers: List[nn.Module] = []
        prev_dim = self.bottleneck_dim
        for h in reversed(self.hidden_dims):
            decoder_layers.extend(_block(prev_dim, int(h)))
            prev_dim = int(h)

        decoder_layers.append(nn.Linear(prev_dim, self.input_dim))

        class _AE(nn.Module):
            def __init__(self, encoder: List[nn.Module], decoder: List[nn.Module]):
                super().__init__()
                self.encoder = nn.Sequential(*encoder)
                self.decoder = nn.Sequential(*decoder)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                z = self.encoder(x)
                return self.decoder(z)

            def encode(self, x: torch.Tensor) -> torch.Tensor:
                return self.encoder(x)

        return _AE(encoder_layers, decoder_layers)

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names is None:
            return X
        if list(X.columns) == self.feature_names:
            return X
        aligned = X.reindex(columns=self.feature_names)
        if not self._warned_feature_mismatch:
            missing = [c for c in self.feature_names if c not in X.columns]
            extra = [c for c in X.columns if c not in self.feature_names]
            if missing or extra:
                print(
                    "Warning: DAE v2 input features differ from training; "
                    f"missing={len(missing)}, extra={len(extra)}. "
                    "Columns were aligned by name."
                )
            self._warned_feature_mismatch = True
        return aligned

    def _fit_scaler(self, X_np: np.ndarray) -> None:
        # Robust center/scale per feature.
        with warnings.catch_warnings():
            # Many indicator columns are legitimately all-NaN early in the series.
            # We handle those columns by turning NaN center/scale into 0/1, but we
            # don't want noisy RuntimeWarnings in the console.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            center = np.nanmedian(X_np, axis=0)
            q25 = np.nanpercentile(X_np, 25, axis=0)
            q75 = np.nanpercentile(X_np, 75, axis=0)
            scale = q75 - q25

        center = np.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
        scale[scale < 1e-8] = 1.0

        self.feature_center = center
        self.feature_scale = scale

    def _normalize(self, X_np: np.ndarray, *, fit: bool) -> Tuple[np.ndarray, np.ndarray]:
        if fit:
            self._fit_scaler(X_np)

        if self.feature_center is None or self.feature_scale is None:
            raise RuntimeError("DAE v2 scaler not initialized. Call fit() first.")

        observed_mask = np.isfinite(X_np).astype(np.float32)
        X_norm = (X_np - self.feature_center) / self.feature_scale

        if self.clip_value is not None:
            X_norm = np.clip(X_norm, -self.clip_value, self.clip_value)

        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return X_norm, observed_mask

    def fit(self, X: pd.DataFrame, X_val: Optional[pd.DataFrame] = None, *, verbose: bool = True) -> Dict:
        self.feature_names = list(X.columns)
        self._warned_feature_mismatch = False

        X_np = X.values.astype(np.float32)
        X_norm, X_mask = self._normalize(X_np, fit=True)

        X_val_norm = None
        X_val_mask = None
        if X_val is not None:
            X_val_np = X_val.values.astype(np.float32)
            X_val_norm, X_val_mask = self._normalize(X_val_np, fit=False)

        train_ds = TensorDataset(
            torch.from_numpy(X_norm),
            torch.from_numpy(X_norm),
            torch.from_numpy(X_mask),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val_norm is not None and X_val_mask is not None:
            val_ds = TensorDataset(
                torch.from_numpy(X_val_norm),
                torch.from_numpy(X_val_norm),
                torch.from_numpy(X_val_mask),
            )
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.SmoothL1Loss(reduction='none')

        best_monitor_loss = float('inf')
        best_state = None
        best_train_loss = float('inf')
        best_val_loss: Optional[float] = None
        patience_counter = 0
        min_delta = 1e-5
        self.training_history = []

        if verbose:
            print(f"    Training DAE v2: {self.input_dim} -> {self.hidden_dims} -> {self.bottleneck_dim}")
            print(
                f"    Device: {self.device}, Epochs: {self.epochs}, Batch: {self.batch_size}, "
                f"mask_prob={self.mask_prob:.2f}, noise_std={self.noise_std:.3f}"
            )

        for epoch in range(self.epochs):
            self.model.train()
            train_loss_accum = 0.0

            for batch_x, batch_target, batch_obs_mask in train_loader:
                batch_x = batch_x.to(self.device)
                batch_target = batch_target.to(self.device)
                batch_obs_mask = batch_obs_mask.to(self.device)

                noisy_x = batch_x
                if self.noise_std > 0:
                    noisy_x = noisy_x + torch.randn_like(noisy_x) * self.noise_std

                if self.mask_prob > 0:
                    # Mask only observed entries so we learn to reconstruct real values.
                    rand_mask = (torch.rand_like(noisy_x) < self.mask_prob).float()
                    rand_mask = rand_mask * batch_obs_mask
                    noisy_x = noisy_x * (1.0 - rand_mask)

                optimizer.zero_grad()
                reconstructed = self.model(noisy_x)
                per_elem = criterion(reconstructed, batch_target)
                loss = (per_elem * batch_obs_mask).sum() / (batch_obs_mask.sum() + 1e-8)
                loss.backward()
                optimizer.step()

                train_loss_accum += float(loss.item()) * batch_x.size(0)

            train_loss = train_loss_accum / len(train_loader.dataset)

            val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_loss_accum = 0.0
                with torch.no_grad():
                    for batch_x, batch_target, batch_obs_mask in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_target = batch_target.to(self.device)
                        batch_obs_mask = batch_obs_mask.to(self.device)

                        noisy_x = batch_x
                        if self.noise_std > 0:
                            noisy_x = noisy_x + torch.randn_like(noisy_x) * self.noise_std
                        if self.mask_prob > 0:
                            rand_mask = (torch.rand_like(noisy_x) < self.mask_prob).float()
                            rand_mask = rand_mask * batch_obs_mask
                            noisy_x = noisy_x * (1.0 - rand_mask)

                        reconstructed = self.model(noisy_x)
                        per_elem = criterion(reconstructed, batch_target)
                        loss = (per_elem * batch_obs_mask).sum() / (batch_obs_mask.sum() + 1e-8)
                        val_loss_accum += float(loss.item()) * batch_x.size(0)

                val_loss = val_loss_accum / len(val_loader.dataset)

            monitor_loss = float(val_loss) if val_loss is not None else float(train_loss)
            if (monitor_loss + min_delta) < best_monitor_loss:
                best_monitor_loss = monitor_loss
                best_train_loss = float(train_loss)
                best_val_loss = float(val_loss) if val_loss is not None else None
                patience_counter = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            self.training_history.append(
                {'epoch': int(epoch + 1), 'train_loss': float(train_loss), 'val_loss': float(val_loss) if val_loss is not None else None}
            )

            if verbose and (epoch + 1) % 10 == 0:
                val_str = f", Val Loss: {val_loss:.6f}" if val_loss is not None else ""
                print(f"      Epoch {epoch + 1}/{self.epochs}: Train Loss: {train_loss:.6f}{val_str}")

            if self.early_stopping_patience and patience_counter >= self.early_stopping_patience:
                if verbose:
                    print(f"      Early stopping at epoch {epoch + 1}")
                break

        # Restore best weights.
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_fitted = True
        metrics = {
            'best_monitor_loss': float(best_monitor_loss),
            'best_train_loss': float(best_train_loss),
            'best_val_loss': float(best_val_loss) if best_val_loss is not None else None,
            'epochs_ran': int(self.training_history[-1]['epoch']) if self.training_history else 0,
        }
        return metrics

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("DAE v2 not fitted yet. Call fit() first.")

        X_aligned = self._align_features(X)
        X_np = X_aligned.values.astype(np.float32)
        X_norm, _ = self._normalize(X_np, fit=False)

        x_tensor = torch.from_numpy(X_norm).to(self.device)
        self.model.eval()
        with torch.no_grad():
            encoded = self.model.encode(x_tensor).detach().cpu().numpy()

        cols = [f'dae2_feature_{i}' for i in range(encoded.shape[1])]
        return pd.DataFrame(encoded, index=X.index, columns=cols)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'model_state': {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            'input_dim': int(self.input_dim),
            'bottleneck_dim': int(self.bottleneck_dim),
            'hidden_dims': list(self.hidden_dims),
            'mask_prob': float(self.mask_prob),
            'noise_std': float(self.noise_std),
            'dropout_rate': float(self.dropout_rate),
            'learning_rate': float(self.learning_rate),
            'weight_decay': float(self.weight_decay),
            'batch_size': int(self.batch_size),
            'epochs': int(self.epochs),
            'early_stopping_patience': int(self.early_stopping_patience),
            'clip_value': float(self.clip_value) if self.clip_value is not None else None,
            'feature_center': self.feature_center,
            'feature_scale': self.feature_scale,
            'feature_names': self.feature_names,
            'is_fitted': bool(self.is_fitted),
            'training_history': list(self.training_history),
        }

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

    def load(self, path: Path) -> None:
        with open(Path(path), 'rb') as f:
            save_dict = pickle.load(f)

        self.input_dim = int(save_dict['input_dim'])
        self.bottleneck_dim = int(save_dict['bottleneck_dim'])
        self.hidden_dims = list(save_dict['hidden_dims'])
        self.mask_prob = float(save_dict.get('mask_prob', 0.15))
        self.noise_std = float(save_dict.get('noise_std', 0.05))
        self.dropout_rate = float(save_dict.get('dropout_rate', 0.1))
        self.learning_rate = float(save_dict.get('learning_rate', 1e-3))
        self.weight_decay = float(save_dict.get('weight_decay', 1e-5))
        self.batch_size = int(save_dict.get('batch_size', 512))
        self.epochs = int(save_dict.get('epochs', 100))
        self.early_stopping_patience = int(save_dict.get('early_stopping_patience', 10))
        self.clip_value = save_dict.get('clip_value', 8.0)

        self.feature_center = save_dict.get('feature_center')
        self.feature_scale = save_dict.get('feature_scale')
        self.feature_names = save_dict.get('feature_names')
        self.is_fitted = bool(save_dict.get('is_fitted', True))
        self.training_history = list(save_dict.get('training_history', []))
        self._warned_feature_mismatch = False

        self.model = self._build_model()
        self.model.load_state_dict(save_dict['model_state'])
        self.model.to(self.device)


def fit_dae_v2_model(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame],
    *,
    bottleneck_dim: int = 64,
    epochs: int = 100,
    verbose: bool = True,
    internal_val_ratio: float = 0.1,
    internal_val_min_samples: int = 512,
    mask_prob: float = 0.15,
    noise_std: float = 0.05,
) -> Tuple[Optional[DenoisingAutoencoderV2], Dict]:
    """
    Fit DAE v2 on provided features.
    Mirrors the v1 helper in `models.py` so the training pipeline can swap between implementations.
    """
    if not PYTORCH_AVAILABLE:
        if verbose:
            print("  Warning: PyTorch not available. Skipping DAE v2.")
        return None, {'skipped': True, 'reason': 'PyTorch not installed'}

    if verbose:
        print("  Training Denoising Autoencoder v2...")

    input_dim = int(X_train.shape[1])
    bottleneck_dim = int(bottleneck_dim)
    if bottleneck_dim > input_dim // 2:
        bottleneck_dim = max(32, input_dim // 4)
        if verbose:
            print(f"    Adjusted bottleneck to {bottleneck_dim} (input has {input_dim} features)")

    dae = DenoisingAutoencoderV2(
        input_dim=input_dim,
        bottleneck_dim=bottleneck_dim,
        epochs=int(epochs),
        mask_prob=float(mask_prob),
        noise_std=float(noise_std),
    )

    dae_fit_X = X_train
    dae_fit_X_val = X_val
    internal_val_used = False
    if X_val is None and 0 < float(internal_val_ratio) < 0.5 and len(X_train) >= int(internal_val_min_samples):
        split_idx = int(len(X_train) * (1.0 - float(internal_val_ratio)))
        split_idx = max(1, min(split_idx, len(X_train) - 1))
        dae_fit_X = X_train.iloc[:split_idx]
        dae_fit_X_val = X_train.iloc[split_idx:]
        internal_val_used = True
        if verbose:
            print(f"    DAE v2 internal validation: train={len(dae_fit_X):,}, val={len(dae_fit_X_val):,}")

    metrics = dae.fit(dae_fit_X, dae_fit_X_val, verbose=verbose)
    metrics['internal_val_used'] = bool(internal_val_used)
    metrics['dae_fit_train_samples'] = int(len(dae_fit_X))
    metrics['dae_fit_val_samples'] = int(len(dae_fit_X_val)) if dae_fit_X_val is not None else 0
    metrics['mask_prob'] = float(mask_prob)
    metrics['noise_std'] = float(noise_std)

    return dae, metrics


def apply_dae_v2_preprocessing(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame],
    *,
    bottleneck_dim: int = 64,
    epochs: int = 100,
    verbose: bool = True,
    mask_prob: float = 0.15,
    noise_std: float = 0.05,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[DenoisingAutoencoderV2], Dict]:
    dae, metrics = fit_dae_v2_model(
        X_train,
        X_val,
        bottleneck_dim=bottleneck_dim,
        epochs=epochs,
        verbose=verbose,
        mask_prob=mask_prob,
        noise_std=noise_std,
    )
    if dae is None:
        return X_train, X_val, None, metrics

    X_train_enc = dae.transform(X_train)
    X_val_enc = dae.transform(X_val) if X_val is not None else None
    return X_train_enc, X_val_enc, dae, metrics
