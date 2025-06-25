import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import joblib

# ========= Transformer Model =========

class TransformerBranch(nn.Module):
    def __init__(self, input_dim, emb_dim, n_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads,
                                                   dim_feedforward=ff_dim, dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.input_proj(x)
        return self.encoder(x)

class MultimodalTransformer(nn.Module):
    def __init__(self, pad_len, imu_dim, tof_dim, n_classes, emb_dim=128, heads=4, ff_dim=256, layers=2):
        super().__init__()
        self.imu_branch = TransformerBranch(imu_dim, emb_dim, heads, ff_dim, layers)
        self.tof_branch = TransformerBranch(tof_dim, emb_dim, heads, ff_dim, layers)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        imu = x[:, :, :imu_dim]
        tof = x[:, :, imu_dim:]
        imu_feat = self.imu_branch(imu).mean(dim=1)
        tof_feat = self.tof_branch(tof).mean(dim=1)
        combined = torch.cat([imu_feat, tof_feat], dim=1)
        return self.classifier(combined)

# ========= Dataset =========

class MixupDataset(Dataset):
    def __init__(self, X, y, alpha=0.4):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.alpha = alpha

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
            rand_idx = np.random.randint(0, len(self.X))
            x = lam * x + (1 - lam) * self.X[rand_idx]
            y = lam * y + (1 - lam) * self.y[rand_idx]
        return x, y

# ========= Training =========

RAW_CSV = "train.csv"
SAVE_DIR = "train_test_18class_18"
PAD_PERCENTILE = 95
BATCH_SIZE = 64
EPOCHS = 150
MIXUP_ALPHA = 0.4
LR_INIT = 1e-3
WD = 3e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

# ========= Preprocessing =========

df = pd.read_csv(RAW_CSV)
df["gesture"] = df["gesture"].fillna("unknown")
le = LabelEncoder()
df["gesture_class"] = le.fit_transform(df["gesture"])
meta_cols = {'gesture', 'gesture_class', 'sequence_type', 'behavior', 'orientation',
             'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter'}
feature_cols = [c for c in df.columns if c not in meta_cols]
imu_cols = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
tof_cols = [c for c in feature_cols if c.startswith('thm_') or c.startswith('tof_')]
imu_dim, tof_dim = len(imu_cols), len(tof_cols)
scaler = StandardScaler().fit(df[feature_cols].fillna(0))
lens = df.groupby("sequence_id").size().values
pad_len = int(np.percentile(lens, PAD_PERCENTILE))

def preprocess_sequence(seq_df):
    mat_df = seq_df[feature_cols].ffill().bfill().fillna(0)
    mat_scaled = scaler.transform(mat_df)
    mat = mat_scaled.astype(np.float32)
    if len(mat) >= pad_len:
        mat = mat[:pad_len]
    else:
        pad = np.zeros((pad_len - len(mat), mat.shape[1]), dtype=np.float32)
        mat = np.vstack([mat, pad])
    return mat

# ========= Cross Validation =========

seq_ids = df["sequence_id"].unique()
subject_map = df.drop_duplicates("sequence_id").set_index("sequence_id")["subject"]
groups = [subject_map[sid] for sid in seq_ids]
kf = GroupKFold(n_splits=5)

for fold, (tr_idx, va_idx) in enumerate(kf.split(seq_ids, groups=groups)):
    print(f"\n=== Fold {fold+1} ===")
    train_ids, val_ids = seq_ids[tr_idx], seq_ids[va_idx]
    train_df = df[df["sequence_id"].isin(train_ids)]
    val_df = df[df["sequence_id"].isin(val_ids)]

    X_train, y_train = [], []
    for sid in train_ids:
        seq = train_df[train_df["sequence_id"] == sid]
        X_train.append(preprocess_sequence(seq))
        y_train.append(seq["gesture_class"].iloc[0])
    X_train = np.array(X_train)
    y_train_oh = np.eye(len(le.classes_))[y_train]

    X_val, y_val = [], []
    for sid in val_ids:
        seq = val_df[val_df["sequence_id"] == sid]
        X_val.append(preprocess_sequence(seq))
        y_val.append(seq["gesture_class"].iloc[0])
    X_val = np.array(X_val)
    y_val_oh = np.eye(len(le.classes_))[y_val]

    cw_vals = compute_class_weight('balanced', classes=np.arange(len(le.classes_)), y=y_train)
    class_weight = torch.FloatTensor(cw_vals).to(device)

    train_loader = DataLoader(MixupDataset(X_train, y_train_oh, MIXUP_ALPHA), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MixupDataset(X_val, y_val_oh, 0.0), batch_size=BATCH_SIZE)

    model = MultimodalTransformer(pad_len, imu_dim, tof_dim, len(le.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5 * len(train_loader))
    fold_dir = os.path.join(SAVE_DIR, f"fold{fold+1}")
    os.makedirs(fold_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(fold_dir, "scaler.pkl"))

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            if batch_y.ndim == 2:
                loss = -torch.sum(F.log_softmax(logits, dim=1) * batch_y, dim=1).mean()
                targets = batch_y.argmax(dim=1)
                weights = torch.sum(batch_y * class_weight.unsqueeze(0), dim=1)
                loss = (loss * weights).mean()
            else:
                targets = batch_y.long()
                loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                pred = logits.argmax(dim=1).cpu().numpy()
                label = batch_y.argmax(dim=1).cpu().numpy()
                preds.extend(pred)
                labels.extend(label)

        labels = np.array(labels)
        preds = np.array(preds)
        bin_labels = (labels != 0).astype(int)
        bin_preds = (preds != 0).astype(int)
        mask = labels != 0

        acc_18 = accuracy_score(labels, preds)
        f1_18 = f1_score(labels, preds, average="macro")
        acc_bin = accuracy_score(bin_labels, bin_preds)
        f1_bin = f1_score(bin_labels, bin_preds)
        acc_9 = accuracy_score(labels[mask], preds[mask]) if mask.any() else 0
        f1_9 = f1_score(labels[mask], preds[mask], average="macro") if mask.any() else 0
        f1_avg = (f1_bin + f1_9) / 2

        print(f"Fold {fold+1} | Epoch {epoch+1:3d} | TrainLoss: {total_loss:.4f} | "
              f"Acc18: {acc_18:.4f}, F1_18: {f1_18:.4f} | Acc_bin: {acc_bin:.4f}, "
              f"F1_bin: {f1_bin:.4f} | Acc_9: {acc_9:.4f}, F1_9: {f1_9:.4f} | F1_avg_2+9: {f1_avg:.4f}")

        if (epoch + 1) % 10 == 0:
            model_path = os.path.join(fold_dir, f"model_epoch{epoch+1}_with_meta.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'imu_dim': imu_dim,
                'tof_dim': tof_dim,
                'pad_len': pad_len,
                'n_classes': len(le.classes_),
                'feature_cols': feature_cols,
                'gesture_classes': le.classes_
            }, model_path)
