import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pickle
from lstm_model import GridLSTM

# ── 1. Load data ──────────────────────────────────────────
df = pd.read_csv("data/grid_load.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)

# ── 2. Create time features ──────────────────────────────
df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Monday, 6=Sunday

# Normalise time features to 0-1
df["hour_norm"] = df["hour"] / 24.0
df["dow_norm"]  = df["day_of_week"] / 6.0

# ── 3. Scale load separately (need to inverse later) ─────
load_scaler = MinMaxScaler()
df["load_scaled"] = load_scaler.fit_transform(df[["load_mw"]])

with open("data/scaler.pkl", "wb") as f:
    pickle.dump(load_scaler, f)

# ── 4. Build feature matrix ──────────────────────────────
# 3 features per timestep: [load, hour, day_of_week]
features = df[["load_scaled", "hour_norm", "dow_norm"]].values

print(f"Feature matrix shape: {features.shape}")
print(f"Features per timestep: load, hour (0-1), day_of_week (0-1)")
print(f"Sample row: load={features[0][0]:.3f}, hour={features[0][1]:.3f}, dow={features[0][2]:.3f}")

# ── 5. Build sequences ───────────────────────────────────
SEQ_LEN = 672  # 7 days × 24 hours × 4 readings per hour

def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])          # all 3 features as input
        y.append(data[i + seq_len, 0])         # only load as target
    return np.array(X), np.array(y).reshape(-1, 1)

X, y = make_sequences(features, SEQ_LEN)

# Time-respecting split
split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print(f"\nTrain sequences: {len(X_train)}")
print(f"Val sequences:   {len(X_val)}")
print(f"Each sequence shape: {X_train[0].shape}")  # should be (672, 3)

# ── 6. Dataset ────────────────────────────────────────────
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TimeSeriesDataset(X_train, y_train),
                          batch_size=32, shuffle=False)
val_loader   = DataLoader(TimeSeriesDataset(X_val, y_val),
                          batch_size=32, shuffle=False)

# ── 7. Model, loss, optimiser ─────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nTraining on: {device}")

model = GridLSTM(input_size=3, hidden_size=64, num_layers=2, dropout=0.2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params:,}")

# ── 8. Training loop with early stopping ─────────────────
EPOCHS = 50
PATIENCE = 7

best_val_loss = float("inf")
patience_counter = 0
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            val_loss += criterion(pred, y_batch).item()

    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step()

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | train={train_loss:.6f} | val={val_loss:.6f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "data/lstm_best.pt")
        print(f"  Saved best model")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

# ── 9. Plot loss curves ───────────────────────────────────
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses,   label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("LSTM Training — Loss Curves (with temporal features)")
plt.legend()
plt.tight_layout()
plt.savefig("data/lstm_loss_curves.png", dpi=150)
plt.show()
print(f"\nBest val loss: {best_val_loss:.6f}")
print("Loss curves saved to data/lstm_loss_curves.png")