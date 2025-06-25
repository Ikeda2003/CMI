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

# ========= SE Block & Residual CNN ==========
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.se(x)
        x = x + shortcut
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x

class Attention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Tanh()
        )

    def forward(self, x):  # x: [B, T, C]
        score = self.attn(x).squeeze(-1)
        weights = F.softmax(score, dim=1).unsqueeze(-1)
        context = torch.sum(x * weights, dim=1)
        return context

class TwoBranchNet(nn.Module):
    def __init__(self, pad_len, imu_dim, tof_dim, n_classes):
        super().__init__()
        self.imu_dim = imu_dim

        self.imu_branch = nn.Sequential(
            ResidualSEBlock(imu_dim, 64, kernel_size=3, dropout=0.1),
            ResidualSEBlock(64, 128, kernel_size=5, dropout=0.1)
        )

        self.tof_branch = nn.Sequential(
            nn.Conv1d(tof_dim, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )

        self.bi_lstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)
        self.bi_gru = nn.GRU(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)

        self.noise = nn.Dropout(p=0.09)
        self.dense_xc = nn.Sequential(
            nn.Linear(256, 16),
            nn.ELU()
        )

        self.attention = Attention(128 * 4 + 16)

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 + 16, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        imu = x[:, :, :self.imu_dim].transpose(1, 2)
        tof = x[:, :, self.imu_dim:].transpose(1, 2)
        x1 = self.imu_branch(imu).transpose(1, 2)
        x2 = self.tof_branch(tof).transpose(1, 2)
        merged = torch.cat([x1, x2], dim=2)
        xa, _ = self.bi_lstm(merged)
        xb, _ = self.bi_gru(merged)
        xc = self.noise(merged)
        xc = self.dense_xc(xc)
        x_cat = torch.cat([xa, xb, xc], dim=2)
        context = self.attention(x_cat)
        return self.classifier(context)

# ========= Dataset ==========
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

# ========= Training Config & Preprocessing ==========
RAW_CSV = "train.csv"
SAVE_DIR = "train_test_18class_19"
PAD_PERCENTILE = 95
BATCH_SIZE = 64
EPOCHS = 160
MIXUP_ALPHA = 0.4
LR_INIT = 5e-4
WD = 3e-3
PATIENCE = 40
os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# ========= Cross Validation ==========
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

    fold_dir = os.path.join(SAVE_DIR, f"fold{fold+1}")
    os.makedirs(fold_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(fold_dir, "scaler.pkl"))

    cw_vals = compute_class_weight('balanced', classes=np.arange(len(le.classes_)), y=y_train)
    class_weight = torch.FloatTensor(cw_vals).to(device)

    train_loader = DataLoader(MixupDataset(X_train, y_train_oh, MIXUP_ALPHA), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MixupDataset(X_val, y_val_oh, 0.0), batch_size=BATCH_SIZE)

    model = TwoBranchNet(pad_len, imu_dim, tof_dim, len(le.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15 * len(train_loader))

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = -torch.sum(F.log_softmax(logits, dim=1) * batch_y, dim=1)
            weights = torch.sum(batch_y * class_weight.unsqueeze(0), dim=1)
            loss = (loss * weights).mean()
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
            torch.save({
                'model_state_dict': model.state_dict(),
                'imu_dim': imu_dim,
                'tof_dim': tof_dim,
                'pad_len': pad_len,
                'n_classes': len(le.classes_),
                'feature_cols': feature_cols,
                'gesture_classes': le.classes_
            }, os.path.join(fold_dir, f"model_epoch{epoch+1}_with_meta.pt"))

#スコアcv: 0.79364 lb:0.71
