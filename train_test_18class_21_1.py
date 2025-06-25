import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ========= 特徴量拡張 =========
def feature_eng(df: pd.DataFrame) -> pd.DataFrame:
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))
    df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)

    insert_cols = ['acc_mag', 'rot_angle', 'acc_mag_jerk', 'rot_angle_vel']
    cols = list(df.columns)
    for i, col in enumerate(cols):
        if col.startswith('thm_') or col.startswith('tof_'):
            insert_index = i
            break
    cols_wo_insert = [c for c in cols if c not in insert_cols]
    df = df[cols_wo_insert[:insert_index] + insert_cols + cols_wo_insert[insert_index:]]
    return df

# ========= モデル定義 =========
class MultiScaleConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, k, padding=k//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
            for k in kernel_sizes
        ])

    def forward(self, x):
        return torch.cat([conv(x) for conv in self.convs], dim=1)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class ResidualSEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, pool_size=2, drop=0.3):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.bn_sc = nn.BatchNorm1d(out_ch) if in_ch != out_ch else nn.Identity()
        self.se = SEBlock(out_ch)
        self.pool = nn.MaxPool1d(pool_size)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        res = self.bn_sc(self.shortcut(x))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se(x)
        x = F.relu(x + res)
        x = self.pool(x)
        return self.drop(x)

class AttentionLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        score = torch.tanh(self.fc(x)).squeeze(-1)
        weights = F.softmax(score, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)

class MetaFeatureExtractor(nn.Module):
    def forward(self, x):
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        maxv, _ = x.max(dim=1)
        minv, _ = x.min(dim=1)
        slope = (x[:, -1, :] - x[:, 0, :]) / max(x.size(1) - 1, 1)
        return torch.cat([mean, std, maxv, minv, slope], dim=1)

class GaussianNoise(nn.Module):
    def __init__(self, stddev=0.09):
        super().__init__()
        self.stddev = stddev
    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.stddev
        return x

class ModelVariant_LSTMGRU(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        C = 11
        self.meta = MetaFeatureExtractor()
        self.meta_dense = nn.Sequential(
            nn.Linear(5*C, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.branches = nn.ModuleList([
            nn.Sequential(
                MultiScaleConv1d(1, 12),
                ResidualSEBlock(36, 48),
                ResidualSEBlock(48, 48),
            ) for _ in range(C)
        ])
        self.bigru = nn.GRU(48*C, 128, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2)
        self.bilstm = nn.LSTM(48*C, 128, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2)
        self.noise = GaussianNoise(0.09)
        self.attn = AttentionLayer(256+256+48*C)
        self.head = nn.Sequential(
            nn.Linear(256+256+48*C + 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        meta = self.meta_dense(self.meta(x))
        feats = []
        for i in range(x.shape[2]):
            f = self.branches[i](x[:, :, i].unsqueeze(1))
            feats.append(f.transpose(1, 2))
        x = torch.cat(feats, dim=2)
        gru, _ = self.bigru(x)
        lstm, _ = self.bilstm(x)
        noise = self.noise(x)
        x = torch.cat([gru, lstm, noise], dim=2)
        x = self.attn(x)
        x = torch.cat([x, meta], dim=1)
        return self.head(x), None

# ========= 学習ループ =========
RAW_CSV = "train.csv"
SAVE_DIR = "train_result_lstmgru_new"
PAD_PERCENTILE = 95
BATCH_SIZE = 64
EPOCHS = 100
LR_INIT = 1e-3
WD = 1e-4
PATIENCE = 20
os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(RAW_CSV)
df = feature_eng(df)
df["gesture"] = df["gesture"].fillna("unknown")
le = LabelEncoder()
df["gesture_class"] = le.fit_transform(df["gesture"])

meta_cols = {'gesture', 'gesture_class', 'sequence_type', 'behavior', 'orientation',
             'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter'}
imu_cols = [c for c in df.columns if c not in meta_cols and not (c.startswith("tof") or c.startswith("thm"))]
imu_dim = len(imu_cols)
scaler = StandardScaler().fit(df[imu_cols].fillna(0))
lens = df.groupby("sequence_id").size().values
pad_len = int(np.percentile(lens, PAD_PERCENTILE))

def preprocess_sequence(seq_df):
    mat_df = seq_df[imu_cols].ffill().bfill().fillna(0)
    mat_df_scaled = pd.DataFrame(scaler.transform(mat_df), columns=imu_cols)
    mat = mat_df_scaled.values.astype(np.float32)
    if len(mat) >= pad_len:
        return mat[:pad_len]
    else:
        pad = np.zeros((pad_len - len(mat), mat.shape[1]), dtype=np.float32)
        return np.vstack([mat, pad])

def to_binary(y18): return [0 if y < 9 else 1 for y in y18]
def to_9class(y18): return [y % 9 for y in y18]

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

seq_ids = df["sequence_id"].unique()
subject_map = df.drop_duplicates("sequence_id").set_index("sequence_id")["subject"]
groups = [subject_map[sid] for sid in seq_ids]
kf = GroupKFold(n_splits=5)

for fold, (tr_idx, va_idx) in enumerate(kf.split(seq_ids, groups=groups)):
    print(f"\n=== Fold {fold+1} ===")
    train_ids, val_ids = seq_ids[tr_idx], seq_ids[va_idx]
    train_df = df[df["sequence_id"].isin(train_ids)]
    val_df = df[df["sequence_id"].isin(val_ids)]

    X_train = [preprocess_sequence(train_df[train_df["sequence_id"] == sid]) for sid in train_ids]
    y_train = [train_df[train_df["sequence_id"] == sid]["gesture_class"].iloc[0] for sid in train_ids]
    X_val = [preprocess_sequence(val_df[val_df["sequence_id"] == sid]) for sid in val_ids]
    y_val = [val_df[val_df["sequence_id"] == sid]["gesture_class"].iloc[0] for sid in val_ids]

    y_train_tensor = torch.nn.functional.one_hot(torch.tensor(y_train, dtype=torch.int64), num_classes=len(le.classes_)).float()
    y_val_tensor = torch.nn.functional.one_hot(torch.tensor(y_val, dtype=torch.int64), num_classes=len(le.classes_)).float()

    train_loader = DataLoader(SimpleDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SimpleDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model = ModelVariant_LSTMGRU(num_classes=len(le.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    best_f1 = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits, _ = model(batch_x)
                pred = logits.argmax(dim=1).cpu().numpy()
                labels.extend(batch_y.numpy())
                preds.extend(pred)

        acc_18 = accuracy_score(labels, preds)
        f1_18 = f1_score(labels, preds, average="macro")
        acc_bin = accuracy_score(to_binary(labels), to_binary(preds))
        f1_bin = f1_score(to_binary(labels), to_binary(preds), average="macro")
        acc_9 = accuracy_score(to_9class(labels), to_9class(preds))
        f1_9 = f1_score(to_9class(labels), to_9class(preds), average="macro")
        f1_avg = (f1_18 + f1_9) / 2

        scheduler.step(1 - f1_18)

        print(f"Fold {fold+1} | Epoch {epoch+1:3d} | TrainLoss: {total_loss/len(train_loader.dataset):.4f} | "
              f"Acc18: {acc_18:.4f}, F1_18: {f1_18:.4f} | Acc_bin: {acc_bin:.4f}, "
              f"F1_bin: {f1_bin:.4f} | Acc_9: {acc_9:.4f}, F1_9: {f1_9:.4f} | F1_avg_2+9: {f1_avg:.4f}")

        if f1_18 > best_f1:
            best_f1 = f1_18
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'imu_dim': imu_dim,
                'pad_len': pad_len,
                'n_classes': len(le.classes_),
                'gesture_classes': le.classes_,
                'feature_cols': imu_cols
            }, os.path.join(SAVE_DIR, f"fold{fold+1}_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered")
                break
