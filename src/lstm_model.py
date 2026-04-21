"""
Project 3 — Stacked LSTM for forward volatility forecasting.

Architecture: LSTM(64) → LSTM(32) → Dense(1) with ReLU output
              (vol is strictly positive, so ReLU is physically motivated)

Supports early stopping and model checkpointing.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class VolLSTM(nn.Module):
    """
    Two-layer stacked LSTM that maps a (batch, seq_len, n_features) tensor
    to a (batch,) vector of annualised vol forecasts.
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden1:    int = 64,
        hidden2:    int = 32,
        dropout:    float = 0.20,
    ):
        super().__init__()
        self.lstm1    = nn.LSTM(input_size, hidden1, batch_first=True)
        self.drop1    = nn.Dropout(dropout)
        self.lstm2    = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.fc       = nn.Linear(hidden2, 1)
        # Softplus instead of ReLU: always positive (vol > 0), smooth gradient
        # everywhere — no dying-neuron problem that ReLU has near zero.
        self.softplus = nn.Softplus()
        # Initialise output bias to ~mean target vol so predictions start
        # in the right range and gradients flow from epoch 1.
        nn.init.constant_(self.fc.bias, 0.15)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out    = self.drop1(out)
        out, _ = self.lstm2(out)
        out    = out[:, -1, :]          # last timestep
        return self.softplus(self.fc(out)).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    epochs:       int   = 100,
    lr:           float = 1e-3,
    batch_size:   int   = 64,
    patience:     int   = 15,          # early-stopping patience
    hidden1:      int   = 64,
    hidden2:      int   = 32,
    dropout:      float = 0.20,
    checkpoint_name: str = "lstm_vol_best.pt",
    device: str | None = None,
) -> tuple[VolLSTM, dict]:
    """
    Train VolLSTM with Adam + MSE loss, early stopping, model checkpoint.

    Returns the best model (loaded from checkpoint) and a history dict
    with keys 'train_loss' and 'val_loss'.
    """
    if device is None:
        # MPS (Apple Silicon GPU) has a known bug where LSTM forward pass
        # returns all-zero hidden states, making training produce no-op gradients.
        # Force CPU — it is fast enough for this model size (~100k params).
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model    = VolLSTM(
        input_size=X_train.shape[2], hidden1=hidden1,
        hidden2=hidden2, dropout=dropout
    ).to(device)
    opt       = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=7, min_lr=1e-5
    )
    loss_fn  = nn.MSELoss()

    Xt = torch.tensor(X_train, device=device)
    yt = torch.tensor(y_train, device=device)
    Xv = torch.tensor(X_val,   device=device)
    yv = torch.tensor(y_val,   device=device)

    dataset   = torch.utils.data.TensorDataset(Xt, yt)
    loader    = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    history   = {"train_loss": [], "val_loss": []}
    best_val  = float("inf")
    no_improve = 0
    ckpt_path  = MODELS_DIR / checkpoint_name

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(Xt)

        # --- validate ---
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1

        if epoch % 10 == 0:
            current_lr = opt.param_groups[0]["lr"]
            print(f"Epoch {epoch:4d}  train_RMSE={epoch_loss**0.5:.5f}  "
                  f"val_RMSE={val_loss**0.5:.5f}  "
                  f"lr={current_lr:.2e}  "
                  f"{'*' if no_improve == 0 else ''}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    # Reload best weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, history


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(
    model: VolLSTM,
    X: np.ndarray,
    device: str | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Run model inference and return numpy array of predictions.

    Always runs on CPU regardless of where the model was trained.
    MPS LSTM inference had known correctness issues in early PyTorch versions;
    CPU is reliable and fast enough for batch inference.
    """
    model = model.cpu()
    model.eval()
    preds = []
    Xt = torch.tensor(X, device="cpu")
    with torch.no_grad():
        for i in range(0, len(Xt), batch_size):
            preds.append(model(Xt[i : i + batch_size]).numpy())
    result = np.concatenate(preds)
    if np.all(result == 0):
        raise RuntimeError(
            "predict() returned all-zero predictions — model weights may be "
            "uninitialised or the checkpoint is corrupt. Re-run training."
        )
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE and MAE in annualised vol units (i.e. percentage points)."""
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae  = float(np.mean(np.abs(y_pred - y_true)))
    # Directional accuracy: did we predict the right sign of change?
    diff_true = np.diff(y_true)
    diff_pred = np.diff(y_pred)
    da = float(np.mean(np.sign(diff_pred) == np.sign(diff_true)))
    return {"rmse": rmse, "mae": mae, "directional_accuracy": da}
