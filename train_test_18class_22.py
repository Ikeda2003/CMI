#yukiZのモデル。
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ========= Feature Engineering =========
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

# ========= Model Definition =========
# [Model definitions already included above]
# ========= Model Definition =========
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
        return (x * weights).sum(dim=1), weights

class TwoBranchModel(nn.Module):
    def __init__(self, imu_dim, tof_dim, pad_len, n_classes):
        super().__init__()
        self.imu_dim = imu_dim
        self.tof_dim = tof_dim

        self.imu_branch = nn.Sequential(
            ResidualSEBlock(in_ch=imu_dim, out_ch=64, k=3, drop=0.1),
            ResidualSEBlock(in_ch=64, out_ch=128, k=5, drop=0.1)
        )
        self.tof_branch = nn.Sequential(
            nn.Conv1d(tof_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2)
        )
        self.bi_lstm = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.bi_gru = nn.GRU(256, 128, batch_first=True, bidirectional=True)
        self.noise_fc = nn.Sequential(
            nn.Conv1d(256, 16, kernel_size=1),
            nn.ReLU()
        )
        self.attn = AttentionLayer(256 + 256 + 16)
        self.head = nn.Sequential(
            nn.Linear(256 + 256 + 16, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        imu, tof = x[:, :, :self.imu_dim], x[:, :, self.imu_dim:]
        imu = imu.permute(0, 2, 1)
        tof = tof.permute(0, 2, 1)
        x1 = self.imu_branch(imu).permute(0, 2, 1)
        x2 = self.tof_branch(tof).permute(0, 2, 1)
        merged = torch.cat([x1, x2], dim=2)
        lstm_out, _ = self.bi_lstm(merged)
        gru_out, _ = self.bi_gru(merged)
        noise = self.noise_fc(merged.transpose(1, 2)).transpose(1, 2)
        concat = torch.cat([lstm_out, gru_out, noise], dim=2)
        x, _ = self.attn(concat)
        return self.head(x), None

# ========= Dataset =========
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ========= Dataset =========
# [Dataset definition already included above]

# ========= Main Training =========
RAW_CSV = "train.csv"
SAVE_DIR = "train_test_18class_22"
PAD_PERCENTILE = 95
BATCH_SIZE = 64
EPOCHS = 100
LR_INIT = 5e-4
WD = 3e-3
PATIENCE = 20
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Load and preprocess data
df = pd.read_csv(RAW_CSV)
df = feature_eng(df)
df["gesture"] = df["gesture"].fillna("unknown")
le = LabelEncoder()
df["gesture_class"] = le.fit_transform(df["gesture"])

meta_cols = {"gesture", "gesture_class", "sequence_id", "row_id", "subject", "phase", "sequence_counter"}
all_cols = list(df.columns)
imu_cols = [c for c in all_cols if c.startswith("linear_acc") or c.startswith("rot_") or c in ["acc_mag", "rot_angle"]]
tof_cols = [c for c in all_cols if c.startswith("thm_") or c.startswith("tof_")]
final_cols = imu_cols + tof_cols
imu_dim, tof_dim = len(imu_cols), len(tof_cols)

lens = df.groupby("sequence_id").size().values
pad_len = int(np.percentile(lens, PAD_PERCENTILE))

seq_ids = df["sequence_id"].unique()
grouped = df.groupby("sequence_id")

X, y = [], []
for sid in seq_ids:
    group = grouped.get_group(sid)
    mat = group[final_cols].ffill().bfill().fillna(0).to_numpy()
    if len(mat) >= pad_len:
        mat = mat[:pad_len]
    else:
        pad = np.zeros((pad_len - len(mat), mat.shape[1]))
        mat = np.vstack([mat, pad])
    X.append(mat)
    y.append(group["gesture_class"].iloc[0])

subject_map = df.drop_duplicates("sequence_id").set_index("sequence_id")["subject"]
groups = [subject_map[sid] for sid in seq_ids]
kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr_idx, va_idx) in enumerate(kf.split(seq_ids, [y[i] for i in range(len(y))], groups)):
    print(f"=== Fold {fold+1} ===")
    fold_dir = os.path.join(SAVE_DIR, f"fold{fold+1}")
    os.makedirs(fold_dir, exist_ok=True)

    scaler = StandardScaler().fit(np.vstack([X[i] for i in tr_idx]))
    joblib.dump(scaler, os.path.join(fold_dir, "scaler.pkl"))

    X_train = [scaler.transform(X[i]) for i in tr_idx]
    y_train = [y[i] for i in tr_idx]
    X_val = [scaler.transform(X[i]) for i in va_idx]
    y_val = [y[i] for i in va_idx]

    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model = TwoBranchModel(imu_dim, tof_dim, pad_len, n_classes=len(le.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=WD)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out, _ = model(xb)
                pred = out.argmax(1).cpu().numpy()
                preds.extend(pred)
                trues.extend(yb.numpy())

        acc_18 = accuracy_score(trues, preds)
        f1_18 = f1_score(trues, preds, average="macro")
        acc_bin = accuracy_score([0 if y < 9 else 1 for y in trues], [0 if y < 9 else 1 for y in preds])
        f1_bin = f1_score([0 if y < 9 else 1 for y in trues], [0 if y < 9 else 1 for y in preds], average="macro")
        acc_9 = accuracy_score([y % 9 for y in trues], [y % 9 for y in preds])
        f1_9 = f1_score([y % 9 for y in trues], [y % 9 for y in preds], average="macro")
        f1_avg = (f1_bin + f1_9) / 2

        print(f"Epoch {epoch+1:3d} | TrainLoss: {total_loss/len(train_loader.dataset):.4f} | "
              f"Acc18: {acc_18:.4f}, F1_18: {f1_18:.4f} | Acc_bin: {acc_bin:.4f}, "
              f"F1_bin: {f1_bin:.4f} | Acc_9: {acc_9:.4f}, F1_9: {f1_9:.4f} | F1_avg_2+9: {f1_avg:.4f}")

        if f1_18 > best_f1:
            best_f1 = f1_18
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(fold_dir, "best.pt"))
            with open(os.path.join(fold_dir, "config.txt"), "w") as f:
                f.write(f"imu_dim={imu_dim}\ntof_dim={tof_dim}\npad_len={pad_len}\nclasses={','.join(le.classes_)}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping")
                break

"""
=== Fold 1 ======
Epoch   1 | TrainLoss: 2.3134 | Acc18: 0.4457, F1_18: 0.3656 | Acc_bin: 0.8626, F1_bin: 0.8621 | Acc_9: 0.4823, F1_9: 0.4159 | F1_avg_2+9: 0.6390
Epoch   2 | TrainLoss: 1.7777 | Acc18: 0.5190, F1_18: 0.4665 | Acc_bin: 0.8907, F1_bin: 0.8905 | Acc_9: 0.5510, F1_9: 0.5214 | F1_avg_2+9: 0.7060
Epoch   3 | TrainLoss: 1.5582 | Acc18: 0.5347, F1_18: 0.5138 | Acc_bin: 0.8953, F1_bin: 0.8952 | Acc_9: 0.5674, F1_9: 0.5520 | F1_avg_2+9: 0.7236
Epoch   4 | TrainLoss: 1.4061 | Acc18: 0.5628, F1_18: 0.5456 | Acc_bin: 0.9182, F1_bin: 0.9179 | Acc_9: 0.5857, F1_9: 0.5697 | F1_avg_2+9: 0.7438
Epoch   5 | TrainLoss: 1.3075 | Acc18: 0.5491, F1_18: 0.5442 | Acc_bin: 0.9136, F1_bin: 0.9133 | Acc_9: 0.5694, F1_9: 0.5670 | F1_avg_2+9: 0.7401
Epoch   6 | TrainLoss: 1.2249 | Acc18: 0.4895, F1_18: 0.5030 | Acc_bin: 0.8776, F1_bin: 0.8775 | Acc_9: 0.5209, F1_9: 0.5288 | F1_avg_2+9: 0.7032
Epoch   7 | TrainLoss: 1.1615 | Acc18: 0.5497, F1_18: 0.5359 | Acc_bin: 0.9149, F1_bin: 0.9145 | Acc_9: 0.5753, F1_9: 0.5795 | F1_avg_2+9: 0.7470
Epoch   8 | TrainLoss: 1.1071 | Acc18: 0.5720, F1_18: 0.5667 | Acc_bin: 0.9116, F1_bin: 0.9114 | Acc_9: 0.5955, F1_9: 0.5902 | F1_avg_2+9: 0.7508
Epoch   9 | TrainLoss: 1.0327 | Acc18: 0.5700, F1_18: 0.5729 | Acc_bin: 0.9143, F1_bin: 0.9140 | Acc_9: 0.5969, F1_9: 0.6048 | F1_avg_2+9: 0.7594
Epoch  10 | TrainLoss: 0.9990 | Acc18: 0.6027, F1_18: 0.5952 | Acc_bin: 0.9162, F1_bin: 0.9160 | Acc_9: 0.6283, F1_9: 0.6211 | F1_avg_2+9: 0.7686
Epoch  11 | TrainLoss: 0.9650 | Acc18: 0.5897, F1_18: 0.5878 | Acc_bin: 0.9175, F1_bin: 0.9172 | Acc_9: 0.6126, F1_9: 0.6095 | F1_avg_2+9: 0.7634
Epoch  12 | TrainLoss: 0.9251 | Acc18: 0.5726, F1_18: 0.5701 | Acc_bin: 0.8874, F1_bin: 0.8874 | Acc_9: 0.6054, F1_9: 0.5936 | F1_avg_2+9: 0.7405
Epoch  13 | TrainLoss: 0.8955 | Acc18: 0.6191, F1_18: 0.6290 | Acc_bin: 0.9130, F1_bin: 0.9128 | Acc_9: 0.6499, F1_9: 0.6433 | F1_avg_2+9: 0.7780
Epoch  14 | TrainLoss: 0.8564 | Acc18: 0.5975, F1_18: 0.6021 | Acc_bin: 0.9005, F1_bin: 0.9005 | Acc_9: 0.6283, F1_9: 0.6277 | F1_avg_2+9: 0.7641
Epoch  15 | TrainLoss: 0.8579 | Acc18: 0.6106, F1_18: 0.5970 | Acc_bin: 0.9202, F1_bin: 0.9199 | Acc_9: 0.6387, F1_9: 0.6178 | F1_avg_2+9: 0.7688
Epoch  16 | TrainLoss: 0.8255 | Acc18: 0.5910, F1_18: 0.6024 | Acc_bin: 0.9188, F1_bin: 0.9184 | Acc_9: 0.6145, F1_9: 0.6136 | F1_avg_2+9: 0.7660
Epoch  17 | TrainLoss: 0.8260 | Acc18: 0.6152, F1_18: 0.6168 | Acc_bin: 0.8973, F1_bin: 0.8972 | Acc_9: 0.6479, F1_9: 0.6380 | F1_avg_2+9: 0.7676
Epoch  18 | TrainLoss: 0.7882 | Acc18: 0.6119, F1_18: 0.6276 | Acc_bin: 0.9202, F1_bin: 0.9201 | Acc_9: 0.6420, F1_9: 0.6451 | F1_avg_2+9: 0.7826
Epoch  19 | TrainLoss: 0.7789 | Acc18: 0.6237, F1_18: 0.6341 | Acc_bin: 0.9293, F1_bin: 0.9289 | Acc_9: 0.6453, F1_9: 0.6524 | F1_avg_2+9: 0.7907
Epoch  20 | TrainLoss: 0.7455 | Acc18: 0.5884, F1_18: 0.5813 | Acc_bin: 0.9018, F1_bin: 0.9016 | Acc_9: 0.6191, F1_9: 0.5994 | F1_avg_2+9: 0.7505
Epoch  21 | TrainLoss: 0.7361 | Acc18: 0.6243, F1_18: 0.6279 | Acc_bin: 0.9274, F1_bin: 0.9271 | Acc_9: 0.6446, F1_9: 0.6451 | F1_avg_2+9: 0.7861
Epoch  22 | TrainLoss: 0.7201 | Acc18: 0.6302, F1_18: 0.6272 | Acc_bin: 0.9208, F1_bin: 0.9208 | Acc_9: 0.6459, F1_9: 0.6332 | F1_avg_2+9: 0.7770
Epoch  23 | TrainLoss: 0.7132 | Acc18: 0.5864, F1_18: 0.5687 | Acc_bin: 0.8973, F1_bin: 0.8972 | Acc_9: 0.6073, F1_9: 0.5814 | F1_avg_2+9: 0.7393
Epoch  24 | TrainLoss: 0.7079 | Acc18: 0.6145, F1_18: 0.6101 | Acc_bin: 0.9136, F1_bin: 0.9135 | Acc_9: 0.6453, F1_9: 0.6427 | F1_avg_2+9: 0.7781
Epoch  25 | TrainLoss: 0.6876 | Acc18: 0.6086, F1_18: 0.5951 | Acc_bin: 0.9143, F1_bin: 0.9142 | Acc_9: 0.6342, F1_9: 0.6062 | F1_avg_2+9: 0.7602
Epoch  26 | TrainLoss: 0.6652 | Acc18: 0.6158, F1_18: 0.6207 | Acc_bin: 0.9182, F1_bin: 0.9180 | Acc_9: 0.6427, F1_9: 0.6397 | F1_avg_2+9: 0.7789
Epoch  27 | TrainLoss: 0.6585 | Acc18: 0.6080, F1_18: 0.6166 | Acc_bin: 0.9274, F1_bin: 0.9268 | Acc_9: 0.6296, F1_9: 0.6337 | F1_avg_2+9: 0.7803
Epoch  28 | TrainLoss: 0.6420 | Acc18: 0.6342, F1_18: 0.6285 | Acc_bin: 0.9202, F1_bin: 0.9199 | Acc_9: 0.6590, F1_9: 0.6576 | F1_avg_2+9: 0.7887
Epoch  29 | TrainLoss: 0.6173 | Acc18: 0.6165, F1_18: 0.6189 | Acc_bin: 0.9143, F1_bin: 0.9138 | Acc_9: 0.6453, F1_9: 0.6378 | F1_avg_2+9: 0.7758
Epoch  30 | TrainLoss: 0.6223 | Acc18: 0.6257, F1_18: 0.6209 | Acc_bin: 0.9267, F1_bin: 0.9265 | Acc_9: 0.6525, F1_9: 0.6461 | F1_avg_2+9: 0.7863
Epoch  31 | TrainLoss: 0.6166 | Acc18: 0.6178, F1_18: 0.6324 | Acc_bin: 0.9280, F1_bin: 0.9275 | Acc_9: 0.6374, F1_9: 0.6493 | F1_avg_2+9: 0.7884
Epoch  32 | TrainLoss: 0.6012 | Acc18: 0.6211, F1_18: 0.6157 | Acc_bin: 0.9221, F1_bin: 0.9218 | Acc_9: 0.6479, F1_9: 0.6367 | F1_avg_2+9: 0.7792
Epoch  33 | TrainLoss: 0.6052 | Acc18: 0.6041, F1_18: 0.6065 | Acc_bin: 0.9123, F1_bin: 0.9121 | Acc_9: 0.6361, F1_9: 0.6254 | F1_avg_2+9: 0.7688
Epoch  34 | TrainLoss: 0.5675 | Acc18: 0.6152, F1_18: 0.6150 | Acc_bin: 0.9175, F1_bin: 0.9173 | Acc_9: 0.6387, F1_9: 0.6389 | F1_avg_2+9: 0.7781
Epoch  35 | TrainLoss: 0.5977 | Acc18: 0.6257, F1_18: 0.6329 | Acc_bin: 0.9267, F1_bin: 0.9265 | Acc_9: 0.6525, F1_9: 0.6505 | F1_avg_2+9: 0.7885
Epoch  36 | TrainLoss: 0.5709 | Acc18: 0.6178, F1_18: 0.6140 | Acc_bin: 0.9110, F1_bin: 0.9109 | Acc_9: 0.6387, F1_9: 0.6335 | F1_avg_2+9: 0.7722
Epoch  37 | TrainLoss: 0.5734 | Acc18: 0.6263, F1_18: 0.6322 | Acc_bin: 0.9202, F1_bin: 0.9199 | Acc_9: 0.6505, F1_9: 0.6475 | F1_avg_2+9: 0.7837
Epoch  38 | TrainLoss: 0.5416 | Acc18: 0.6237, F1_18: 0.6298 | Acc_bin: 0.9306, F1_bin: 0.9302 | Acc_9: 0.6459, F1_9: 0.6475 | F1_avg_2+9: 0.7888
Epoch  39 | TrainLoss: 0.5631 | Acc18: 0.6263, F1_18: 0.6260 | Acc_bin: 0.9306, F1_bin: 0.9303 | Acc_9: 0.6473, F1_9: 0.6486 | F1_avg_2+9: 0.7895
Early stopping
=== Fold 2 ===
Epoch   1 | TrainLoss: 2.2845 | Acc18: 0.3640, F1_18: 0.3156 | Acc_bin: 0.8015, F1_bin: 0.8014 | Acc_9: 0.3964, F1_9: 0.3392 | F1_avg_2+9: 0.5703
Epoch   2 | TrainLoss: 1.7440 | Acc18: 0.4259, F1_18: 0.4078 | Acc_bin: 0.8591, F1_bin: 0.8590 | Acc_9: 0.4553, F1_9: 0.4460 | F1_avg_2+9: 0.6525
Epoch   3 | TrainLoss: 1.5148 | Acc18: 0.4467, F1_18: 0.4364 | Acc_bin: 0.8683, F1_bin: 0.8682 | Acc_9: 0.4749, F1_9: 0.4610 | F1_avg_2+9: 0.6646
Epoch   4 | TrainLoss: 1.3739 | Acc18: 0.4602, F1_18: 0.4744 | Acc_bin: 0.8493, F1_bin: 0.8492 | Acc_9: 0.4853, F1_9: 0.4915 | F1_avg_2+9: 0.6703
Epoch   5 | TrainLoss: 1.2905 | Acc18: 0.4773, F1_18: 0.4759 | Acc_bin: 0.8866, F1_bin: 0.8864 | Acc_9: 0.4975, F1_9: 0.4942 | F1_avg_2+9: 0.6903
Epoch   6 | TrainLoss: 1.2048 | Acc18: 0.4871, F1_18: 0.4810 | Acc_bin: 0.8879, F1_bin: 0.8875 | Acc_9: 0.5067, F1_9: 0.5139 | F1_avg_2+9: 0.7007
Epoch   7 | TrainLoss: 1.1453 | Acc18: 0.4767, F1_18: 0.4541 | Acc_bin: 0.8707, F1_bin: 0.8705 | Acc_9: 0.5031, F1_9: 0.4906 | F1_avg_2+9: 0.6805
Epoch   8 | TrainLoss: 1.0968 | Acc18: 0.5092, F1_18: 0.5104 | Acc_bin: 0.8836, F1_bin: 0.8835 | Acc_9: 0.5300, F1_9: 0.5305 | F1_avg_2+9: 0.7070
Epoch   9 | TrainLoss: 1.0370 | Acc18: 0.4975, F1_18: 0.5062 | Acc_bin: 0.8940, F1_bin: 0.8939 | Acc_9: 0.5214, F1_9: 0.5217 | F1_avg_2+9: 0.7078
Epoch  10 | TrainLoss: 1.0264 | Acc18: 0.5233, F1_18: 0.5175 | Acc_bin: 0.8848, F1_bin: 0.8847 | Acc_9: 0.5490, F1_9: 0.5432 | F1_avg_2+9: 0.7140
Epoch  11 | TrainLoss: 0.9774 | Acc18: 0.5123, F1_18: 0.5219 | Acc_bin: 0.8787, F1_bin: 0.8787 | Acc_9: 0.5380, F1_9: 0.5416 | F1_avg_2+9: 0.7101
Epoch  12 | TrainLoss: 0.9403 | Acc18: 0.5398, F1_18: 0.5468 | Acc_bin: 0.8866, F1_bin: 0.8866 | Acc_9: 0.5619, F1_9: 0.5633 | F1_avg_2+9: 0.7249
Epoch  13 | TrainLoss: 0.8891 | Acc18: 0.5098, F1_18: 0.5212 | Acc_bin: 0.8621, F1_bin: 0.8621 | Acc_9: 0.5417, F1_9: 0.5514 | F1_avg_2+9: 0.7068
Epoch  14 | TrainLoss: 0.8774 | Acc18: 0.5276, F1_18: 0.5288 | Acc_bin: 0.8873, F1_bin: 0.8872 | Acc_9: 0.5515, F1_9: 0.5498 | F1_avg_2+9: 0.7185
Epoch  15 | TrainLoss: 0.8474 | Acc18: 0.5386, F1_18: 0.5542 | Acc_bin: 0.8873, F1_bin: 0.8870 | Acc_9: 0.5637, F1_9: 0.5688 | F1_avg_2+9: 0.7279
Epoch  16 | TrainLoss: 0.8098 | Acc18: 0.5404, F1_18: 0.5314 | Acc_bin: 0.8915, F1_bin: 0.8912 | Acc_9: 0.5680, F1_9: 0.5553 | F1_avg_2+9: 0.7232
Epoch  17 | TrainLoss: 0.7732 | Acc18: 0.5594, F1_18: 0.5584 | Acc_bin: 0.9013, F1_bin: 0.9012 | Acc_9: 0.5815, F1_9: 0.5801 | F1_avg_2+9: 0.7407
Epoch  18 | TrainLoss: 0.7733 | Acc18: 0.5551, F1_18: 0.5466 | Acc_bin: 0.9026, F1_bin: 0.9024 | Acc_9: 0.5827, F1_9: 0.5810 | F1_avg_2+9: 0.7417
Epoch  19 | TrainLoss: 0.7479 | Acc18: 0.5582, F1_18: 0.5605 | Acc_bin: 0.8824, F1_bin: 0.8823 | Acc_9: 0.5803, F1_9: 0.5700 | F1_avg_2+9: 0.7262
Epoch  20 | TrainLoss: 0.7459 | Acc18: 0.5417, F1_18: 0.5324 | Acc_bin: 0.8824, F1_bin: 0.8823 | Acc_9: 0.5637, F1_9: 0.5543 | F1_avg_2+9: 0.7183
Epoch  21 | TrainLoss: 0.7447 | Acc18: 0.5515, F1_18: 0.5703 | Acc_bin: 0.8756, F1_bin: 0.8756 | Acc_9: 0.5839, F1_9: 0.5878 | F1_avg_2+9: 0.7317
Epoch  22 | TrainLoss: 0.7094 | Acc18: 0.5594, F1_18: 0.5665 | Acc_bin: 0.9001, F1_bin: 0.8997 | Acc_9: 0.5858, F1_9: 0.5800 | F1_avg_2+9: 0.7399
Epoch  23 | TrainLoss: 0.6974 | Acc18: 0.5502, F1_18: 0.5605 | Acc_bin: 0.8854, F1_bin: 0.8854 | Acc_9: 0.5680, F1_9: 0.5629 | F1_avg_2+9: 0.7242
Epoch  24 | TrainLoss: 0.6686 | Acc18: 0.5594, F1_18: 0.5839 | Acc_bin: 0.8995, F1_bin: 0.8994 | Acc_9: 0.5809, F1_9: 0.5925 | F1_avg_2+9: 0.7460
Epoch  25 | TrainLoss: 0.6430 | Acc18: 0.5594, F1_18: 0.5555 | Acc_bin: 0.8971, F1_bin: 0.8970 | Acc_9: 0.5772, F1_9: 0.5821 | F1_avg_2+9: 0.7396
Epoch  26 | TrainLoss: 0.6533 | Acc18: 0.5741, F1_18: 0.5752 | Acc_bin: 0.9112, F1_bin: 0.9109 | Acc_9: 0.5931, F1_9: 0.5983 | F1_avg_2+9: 0.7546
Epoch  27 | TrainLoss: 0.6329 | Acc18: 0.5662, F1_18: 0.5864 | Acc_bin: 0.8989, F1_bin: 0.8986 | Acc_9: 0.5931, F1_9: 0.5996 | F1_avg_2+9: 0.7491
Epoch  28 | TrainLoss: 0.6166 | Acc18: 0.5607, F1_18: 0.5674 | Acc_bin: 0.8958, F1_bin: 0.8956 | Acc_9: 0.5748, F1_9: 0.5830 | F1_avg_2+9: 0.7393
Epoch  29 | TrainLoss: 0.6122 | Acc18: 0.5674, F1_18: 0.5813 | Acc_bin: 0.9069, F1_bin: 0.9064 | Acc_9: 0.5870, F1_9: 0.5914 | F1_avg_2+9: 0.7489
Epoch  30 | TrainLoss: 0.6273 | Acc18: 0.5312, F1_18: 0.5475 | Acc_bin: 0.8781, F1_bin: 0.8777 | Acc_9: 0.5558, F1_9: 0.5678 | F1_avg_2+9: 0.7228
Epoch  31 | TrainLoss: 0.6219 | Acc18: 0.5692, F1_18: 0.5681 | Acc_bin: 0.9013, F1_bin: 0.9007 | Acc_9: 0.5931, F1_9: 0.5940 | F1_avg_2+9: 0.7473
Epoch  32 | TrainLoss: 0.5787 | Acc18: 0.5613, F1_18: 0.5736 | Acc_bin: 0.8971, F1_bin: 0.8968 | Acc_9: 0.5846, F1_9: 0.5875 | F1_avg_2+9: 0.7422
Epoch  33 | TrainLoss: 0.5800 | Acc18: 0.5705, F1_18: 0.5792 | Acc_bin: 0.8983, F1_bin: 0.8978 | Acc_9: 0.5974, F1_9: 0.6034 | F1_avg_2+9: 0.7506
Epoch  34 | TrainLoss: 0.5642 | Acc18: 0.5699, F1_18: 0.5773 | Acc_bin: 0.8989, F1_bin: 0.8987 | Acc_9: 0.5925, F1_9: 0.5965 | F1_avg_2+9: 0.7476
Epoch  35 | TrainLoss: 0.5767 | Acc18: 0.5631, F1_18: 0.5690 | Acc_bin: 0.8934, F1_bin: 0.8921 | Acc_9: 0.5858, F1_9: 0.5895 | F1_avg_2+9: 0.7408
Epoch  36 | TrainLoss: 0.5717 | Acc18: 0.5392, F1_18: 0.5500 | Acc_bin: 0.8817, F1_bin: 0.8814 | Acc_9: 0.5613, F1_9: 0.5608 | F1_avg_2+9: 0.7211
Epoch  37 | TrainLoss: 0.5402 | Acc18: 0.5772, F1_18: 0.5842 | Acc_bin: 0.9050, F1_bin: 0.9049 | Acc_9: 0.6011, F1_9: 0.5956 | F1_avg_2+9: 0.7502
Epoch  38 | TrainLoss: 0.5451 | Acc18: 0.5478, F1_18: 0.5515 | Acc_bin: 0.8977, F1_bin: 0.8976 | Acc_9: 0.5668, F1_9: 0.5668 | F1_avg_2+9: 0.7322
Epoch  39 | TrainLoss: 0.5414 | Acc18: 0.5711, F1_18: 0.5693 | Acc_bin: 0.9099, F1_bin: 0.9097 | Acc_9: 0.5895, F1_9: 0.5897 | F1_avg_2+9: 0.7497
Epoch  40 | TrainLoss: 0.5268 | Acc18: 0.5368, F1_18: 0.5411 | Acc_bin: 0.8915, F1_bin: 0.8915 | Acc_9: 0.5637, F1_9: 0.5705 | F1_avg_2+9: 0.7310
Epoch  41 | TrainLoss: 0.5245 | Acc18: 0.5594, F1_18: 0.5628 | Acc_bin: 0.8977, F1_bin: 0.8976 | Acc_9: 0.5809, F1_9: 0.5783 | F1_avg_2+9: 0.7379
Epoch  42 | TrainLoss: 0.5101 | Acc18: 0.5772, F1_18: 0.5899 | Acc_bin: 0.8977, F1_bin: 0.8976 | Acc_9: 0.6005, F1_9: 0.6012 | F1_avg_2+9: 0.7494
Epoch  43 | TrainLoss: 0.5011 | Acc18: 0.5711, F1_18: 0.5843 | Acc_bin: 0.8928, F1_bin: 0.8927 | Acc_9: 0.5938, F1_9: 0.5971 | F1_avg_2+9: 0.7449
Epoch  44 | TrainLoss: 0.5119 | Acc18: 0.5545, F1_18: 0.5572 | Acc_bin: 0.8989, F1_bin: 0.8988 | Acc_9: 0.5699, F1_9: 0.5687 | F1_avg_2+9: 0.7338
Epoch  45 | TrainLoss: 0.5214 | Acc18: 0.5600, F1_18: 0.5602 | Acc_bin: 0.9001, F1_bin: 0.8998 | Acc_9: 0.5846, F1_9: 0.5837 | F1_avg_2+9: 0.7417
Epoch  46 | TrainLoss: 0.4820 | Acc18: 0.5735, F1_18: 0.5736 | Acc_bin: 0.9020, F1_bin: 0.9018 | Acc_9: 0.5944, F1_9: 0.5876 | F1_avg_2+9: 0.7447
Epoch  47 | TrainLoss: 0.4739 | Acc18: 0.5637, F1_18: 0.5748 | Acc_bin: 0.9062, F1_bin: 0.9060 | Acc_9: 0.5846, F1_9: 0.5856 | F1_avg_2+9: 0.7458
Epoch  48 | TrainLoss: 0.4722 | Acc18: 0.5631, F1_18: 0.5725 | Acc_bin: 0.8928, F1_bin: 0.8926 | Acc_9: 0.5876, F1_9: 0.5964 | F1_avg_2+9: 0.7445
Epoch  49 | TrainLoss: 0.4534 | Acc18: 0.5772, F1_18: 0.5807 | Acc_bin: 0.8866, F1_bin: 0.8866 | Acc_9: 0.5987, F1_9: 0.5984 | F1_avg_2+9: 0.7425
Epoch  50 | TrainLoss: 0.4907 | Acc18: 0.5833, F1_18: 0.5880 | Acc_bin: 0.8995, F1_bin: 0.8990 | Acc_9: 0.6023, F1_9: 0.6062 | F1_avg_2+9: 0.7526
Epoch  51 | TrainLoss: 0.4717 | Acc18: 0.5686, F1_18: 0.5815 | Acc_bin: 0.8946, F1_bin: 0.8943 | Acc_9: 0.5907, F1_9: 0.6001 | F1_avg_2+9: 0.7472
Epoch  52 | TrainLoss: 0.4578 | Acc18: 0.5594, F1_18: 0.5660 | Acc_bin: 0.9026, F1_bin: 0.9021 | Acc_9: 0.5852, F1_9: 0.5890 | F1_avg_2+9: 0.7455
Epoch  53 | TrainLoss: 0.4443 | Acc18: 0.5484, F1_18: 0.5580 | Acc_bin: 0.8879, F1_bin: 0.8877 | Acc_9: 0.5766, F1_9: 0.5776 | F1_avg_2+9: 0.7327
Epoch  54 | TrainLoss: 0.4633 | Acc18: 0.5741, F1_18: 0.5744 | Acc_bin: 0.9142, F1_bin: 0.9137 | Acc_9: 0.5925, F1_9: 0.5937 | F1_avg_2+9: 0.7537
Epoch  55 | TrainLoss: 0.4436 | Acc18: 0.5686, F1_18: 0.5775 | Acc_bin: 0.8946, F1_bin: 0.8945 | Acc_9: 0.5919, F1_9: 0.5885 | F1_avg_2+9: 0.7415
Epoch  56 | TrainLoss: 0.4402 | Acc18: 0.5570, F1_18: 0.5722 | Acc_bin: 0.8922, F1_bin: 0.8921 | Acc_9: 0.5852, F1_9: 0.5858 | F1_avg_2+9: 0.7390
Epoch  57 | TrainLoss: 0.4390 | Acc18: 0.5656, F1_18: 0.5699 | Acc_bin: 0.8922, F1_bin: 0.8921 | Acc_9: 0.5888, F1_9: 0.5909 | F1_avg_2+9: 0.7415
Epoch  58 | TrainLoss: 0.4335 | Acc18: 0.5741, F1_18: 0.5774 | Acc_bin: 0.8995, F1_bin: 0.8990 | Acc_9: 0.5962, F1_9: 0.5913 | F1_avg_2+9: 0.7452
Epoch  59 | TrainLoss: 0.4528 | Acc18: 0.5656, F1_18: 0.5764 | Acc_bin: 0.9013, F1_bin: 0.9008 | Acc_9: 0.5882, F1_9: 0.5970 | F1_avg_2+9: 0.7489
Epoch  60 | TrainLoss: 0.4234 | Acc18: 0.5631, F1_18: 0.5632 | Acc_bin: 0.8842, F1_bin: 0.8841 | Acc_9: 0.5925, F1_9: 0.5912 | F1_avg_2+9: 0.7376
Epoch  61 | TrainLoss: 0.4250 | Acc18: 0.5790, F1_18: 0.5748 | Acc_bin: 0.9105, F1_bin: 0.9101 | Acc_9: 0.5950, F1_9: 0.5935 | F1_avg_2+9: 0.7518
Epoch  62 | TrainLoss: 0.4096 | Acc18: 0.5735, F1_18: 0.5836 | Acc_bin: 0.9069, F1_bin: 0.9065 | Acc_9: 0.5968, F1_9: 0.5966 | F1_avg_2+9: 0.7516
Early stopping
=== Fold 3 ===
Epoch   1 | TrainLoss: 2.2914 | Acc18: 0.3982, F1_18: 0.3372 | Acc_bin: 0.8098, F1_bin: 0.8094 | Acc_9: 0.4258, F1_9: 0.3700 | F1_avg_2+9: 0.5897
Epoch   2 | TrainLoss: 1.7379 | Acc18: 0.4400, F1_18: 0.4339 | Acc_bin: 0.8357, F1_bin: 0.8357 | Acc_9: 0.4683, F1_9: 0.4501 | F1_avg_2+9: 0.6429
Epoch   3 | TrainLoss: 1.5029 | Acc18: 0.4523, F1_18: 0.4509 | Acc_bin: 0.8572, F1_bin: 0.8571 | Acc_9: 0.4726, F1_9: 0.4738 | F1_avg_2+9: 0.6654
Epoch   4 | TrainLoss: 1.3662 | Acc18: 0.4960, F1_18: 0.4941 | Acc_bin: 0.8683, F1_bin: 0.8682 | Acc_9: 0.5243, F1_9: 0.5265 | F1_avg_2+9: 0.6973
Epoch   5 | TrainLoss: 1.2824 | Acc18: 0.5102, F1_18: 0.4971 | Acc_bin: 0.8763, F1_bin: 0.8762 | Acc_9: 0.5323, F1_9: 0.5336 | F1_avg_2+9: 0.7049
Epoch   6 | TrainLoss: 1.2097 | Acc18: 0.4929, F1_18: 0.4960 | Acc_bin: 0.8825, F1_bin: 0.8822 | Acc_9: 0.5188, F1_9: 0.5360 | F1_avg_2+9: 0.7091
Epoch   7 | TrainLoss: 1.1645 | Acc18: 0.5175, F1_18: 0.5149 | Acc_bin: 0.8732, F1_bin: 0.8731 | Acc_9: 0.5446, F1_9: 0.5500 | F1_avg_2+9: 0.7115
Epoch   8 | TrainLoss: 1.0998 | Acc18: 0.5372, F1_18: 0.5392 | Acc_bin: 0.8874, F1_bin: 0.8873 | Acc_9: 0.5643, F1_9: 0.5662 | F1_avg_2+9: 0.7268
Epoch   9 | TrainLoss: 1.0486 | Acc18: 0.4898, F1_18: 0.4972 | Acc_bin: 0.8880, F1_bin: 0.8879 | Acc_9: 0.5114, F1_9: 0.5200 | F1_avg_2+9: 0.7040
Epoch  10 | TrainLoss: 1.0193 | Acc18: 0.5249, F1_18: 0.5154 | Acc_bin: 0.8837, F1_bin: 0.8837 | Acc_9: 0.5508, F1_9: 0.5494 | F1_avg_2+9: 0.7165
Epoch  11 | TrainLoss: 0.9873 | Acc18: 0.5366, F1_18: 0.5284 | Acc_bin: 0.8695, F1_bin: 0.8695 | Acc_9: 0.5668, F1_9: 0.5652 | F1_avg_2+9: 0.7174
Epoch  12 | TrainLoss: 0.9374 | Acc18: 0.5686, F1_18: 0.5735 | Acc_bin: 0.8806, F1_bin: 0.8801 | Acc_9: 0.5982, F1_9: 0.6076 | F1_avg_2+9: 0.7438
Epoch  13 | TrainLoss: 0.9101 | Acc18: 0.5538, F1_18: 0.5480 | Acc_bin: 0.8757, F1_bin: 0.8757 | Acc_9: 0.5803, F1_9: 0.5768 | F1_avg_2+9: 0.7263
Epoch  14 | TrainLoss: 0.8627 | Acc18: 0.5588, F1_18: 0.5579 | Acc_bin: 0.8825, F1_bin: 0.8823 | Acc_9: 0.5871, F1_9: 0.5920 | F1_avg_2+9: 0.7372
Epoch  15 | TrainLoss: 0.8356 | Acc18: 0.5705, F1_18: 0.5672 | Acc_bin: 0.8794, F1_bin: 0.8794 | Acc_9: 0.5945, F1_9: 0.5945 | F1_avg_2+9: 0.7369
Epoch  16 | TrainLoss: 0.8178 | Acc18: 0.5662, F1_18: 0.5659 | Acc_bin: 0.8880, F1_bin: 0.8877 | Acc_9: 0.5945, F1_9: 0.5980 | F1_avg_2+9: 0.7429
Epoch  17 | TrainLoss: 0.7975 | Acc18: 0.5618, F1_18: 0.5704 | Acc_bin: 0.8702, F1_bin: 0.8699 | Acc_9: 0.5963, F1_9: 0.5999 | F1_avg_2+9: 0.7349
Epoch  18 | TrainLoss: 0.7776 | Acc18: 0.5471, F1_18: 0.5399 | Acc_bin: 0.8843, F1_bin: 0.8837 | Acc_9: 0.5723, F1_9: 0.5851 | F1_avg_2+9: 0.7344
Epoch  19 | TrainLoss: 0.7465 | Acc18: 0.5908, F1_18: 0.5726 | Acc_bin: 0.8954, F1_bin: 0.8953 | Acc_9: 0.6135, F1_9: 0.6048 | F1_avg_2+9: 0.7500
Epoch  20 | TrainLoss: 0.7385 | Acc18: 0.5397, F1_18: 0.5467 | Acc_bin: 0.8585, F1_bin: 0.8576 | Acc_9: 0.5766, F1_9: 0.5852 | F1_avg_2+9: 0.7214
Epoch  21 | TrainLoss: 0.7225 | Acc18: 0.5040, F1_18: 0.5083 | Acc_bin: 0.8566, F1_bin: 0.8554 | Acc_9: 0.5354, F1_9: 0.5498 | F1_avg_2+9: 0.7026
Epoch  22 | TrainLoss: 0.6971 | Acc18: 0.5760, F1_18: 0.5748 | Acc_bin: 0.8714, F1_bin: 0.8713 | Acc_9: 0.6068, F1_9: 0.6110 | F1_avg_2+9: 0.7411
Epoch  23 | TrainLoss: 0.6588 | Acc18: 0.5852, F1_18: 0.5727 | Acc_bin: 0.8818, F1_bin: 0.8818 | Acc_9: 0.6142, F1_9: 0.6092 | F1_avg_2+9: 0.7455
Epoch  24 | TrainLoss: 0.6742 | Acc18: 0.5465, F1_18: 0.5403 | Acc_bin: 0.8812, F1_bin: 0.8812 | Acc_9: 0.5791, F1_9: 0.5807 | F1_avg_2+9: 0.7310
Epoch  25 | TrainLoss: 0.6427 | Acc18: 0.5938, F1_18: 0.5848 | Acc_bin: 0.8997, F1_bin: 0.8996 | Acc_9: 0.6129, F1_9: 0.6116 | F1_avg_2+9: 0.7556
Epoch  26 | TrainLoss: 0.6307 | Acc18: 0.5723, F1_18: 0.5802 | Acc_bin: 0.8831, F1_bin: 0.8830 | Acc_9: 0.5945, F1_9: 0.5977 | F1_avg_2+9: 0.7404
Epoch  27 | TrainLoss: 0.6501 | Acc18: 0.5846, F1_18: 0.5822 | Acc_bin: 0.8855, F1_bin: 0.8853 | Acc_9: 0.6129, F1_9: 0.6117 | F1_avg_2+9: 0.7485
Epoch  28 | TrainLoss: 0.6081 | Acc18: 0.6000, F1_18: 0.5839 | Acc_bin: 0.8972, F1_bin: 0.8970 | Acc_9: 0.6271, F1_9: 0.6161 | F1_avg_2+9: 0.7566
Epoch  29 | TrainLoss: 0.5998 | Acc18: 0.5834, F1_18: 0.5892 | Acc_bin: 0.8855, F1_bin: 0.8855 | Acc_9: 0.6080, F1_9: 0.6075 | F1_avg_2+9: 0.7465
Epoch  30 | TrainLoss: 0.5937 | Acc18: 0.5520, F1_18: 0.5607 | Acc_bin: 0.8695, F1_bin: 0.8693 | Acc_9: 0.5729, F1_9: 0.5812 | F1_avg_2+9: 0.7252
Epoch  31 | TrainLoss: 0.5859 | Acc18: 0.5711, F1_18: 0.5611 | Acc_bin: 0.8849, F1_bin: 0.8843 | Acc_9: 0.5938, F1_9: 0.5924 | F1_avg_2+9: 0.7383
Epoch  32 | TrainLoss: 0.6038 | Acc18: 0.5748, F1_18: 0.5793 | Acc_bin: 0.8886, F1_bin: 0.8885 | Acc_9: 0.6012, F1_9: 0.6017 | F1_avg_2+9: 0.7451
Epoch  33 | TrainLoss: 0.5615 | Acc18: 0.5840, F1_18: 0.5748 | Acc_bin: 0.8702, F1_bin: 0.8700 | Acc_9: 0.6117, F1_9: 0.6063 | F1_avg_2+9: 0.7381
Epoch  34 | TrainLoss: 0.5609 | Acc18: 0.5938, F1_18: 0.5884 | Acc_bin: 0.8831, F1_bin: 0.8828 | Acc_9: 0.6228, F1_9: 0.6253 | F1_avg_2+9: 0.7541
Epoch  35 | TrainLoss: 0.5492 | Acc18: 0.5994, F1_18: 0.5948 | Acc_bin: 0.8837, F1_bin: 0.8834 | Acc_9: 0.6258, F1_9: 0.6272 | F1_avg_2+9: 0.7553
Epoch  36 | TrainLoss: 0.5399 | Acc18: 0.5889, F1_18: 0.5913 | Acc_bin: 0.8738, F1_bin: 0.8736 | Acc_9: 0.6148, F1_9: 0.6114 | F1_avg_2+9: 0.7425
Epoch  37 | TrainLoss: 0.5514 | Acc18: 0.5822, F1_18: 0.5819 | Acc_bin: 0.8732, F1_bin: 0.8732 | Acc_9: 0.6080, F1_9: 0.6066 | F1_avg_2+9: 0.7399
Epoch  38 | TrainLoss: 0.5437 | Acc18: 0.5938, F1_18: 0.5709 | Acc_bin: 0.8886, F1_bin: 0.8886 | Acc_9: 0.6178, F1_9: 0.6023 | F1_avg_2+9: 0.7455
Epoch  39 | TrainLoss: 0.5245 | Acc18: 0.5828, F1_18: 0.5837 | Acc_bin: 0.8855, F1_bin: 0.8855 | Acc_9: 0.6037, F1_9: 0.6046 | F1_avg_2+9: 0.7451
Epoch  40 | TrainLoss: 0.5216 | Acc18: 0.5791, F1_18: 0.5792 | Acc_bin: 0.8818, F1_bin: 0.8816 | Acc_9: 0.6006, F1_9: 0.6040 | F1_avg_2+9: 0.7428
Epoch  41 | TrainLoss: 0.5084 | Acc18: 0.5723, F1_18: 0.5674 | Acc_bin: 0.8794, F1_bin: 0.8791 | Acc_9: 0.5988, F1_9: 0.6032 | F1_avg_2+9: 0.7411
Epoch  42 | TrainLoss: 0.5070 | Acc18: 0.5785, F1_18: 0.5807 | Acc_bin: 0.8726, F1_bin: 0.8723 | Acc_9: 0.6080, F1_9: 0.6101 | F1_avg_2+9: 0.7412
Epoch  43 | TrainLoss: 0.4874 | Acc18: 0.5705, F1_18: 0.5762 | Acc_bin: 0.8788, F1_bin: 0.8788 | Acc_9: 0.5914, F1_9: 0.5952 | F1_avg_2+9: 0.7370
Epoch  44 | TrainLoss: 0.4938 | Acc18: 0.5686, F1_18: 0.5686 | Acc_bin: 0.8874, F1_bin: 0.8873 | Acc_9: 0.5951, F1_9: 0.5969 | F1_avg_2+9: 0.7421
Epoch  45 | TrainLoss: 0.4658 | Acc18: 0.5618, F1_18: 0.5707 | Acc_bin: 0.8757, F1_bin: 0.8755 | Acc_9: 0.5877, F1_9: 0.5901 | F1_avg_2+9: 0.7328
Epoch  46 | TrainLoss: 0.4911 | Acc18: 0.5797, F1_18: 0.5739 | Acc_bin: 0.8831, F1_bin: 0.8830 | Acc_9: 0.6000, F1_9: 0.5980 | F1_avg_2+9: 0.7405
Epoch  47 | TrainLoss: 0.4638 | Acc18: 0.5803, F1_18: 0.5943 | Acc_bin: 0.8769, F1_bin: 0.8767 | Acc_9: 0.6037, F1_9: 0.6189 | F1_avg_2+9: 0.7478
Epoch  48 | TrainLoss: 0.4745 | Acc18: 0.5871, F1_18: 0.5741 | Acc_bin: 0.8831, F1_bin: 0.8828 | Acc_9: 0.6160, F1_9: 0.6185 | F1_avg_2+9: 0.7507
Epoch  49 | TrainLoss: 0.4427 | Acc18: 0.5778, F1_18: 0.5636 | Acc_bin: 0.8763, F1_bin: 0.8763 | Acc_9: 0.6074, F1_9: 0.5980 | F1_avg_2+9: 0.7371
Epoch  50 | TrainLoss: 0.4593 | Acc18: 0.5748, F1_18: 0.5778 | Acc_bin: 0.8788, F1_bin: 0.8788 | Acc_9: 0.6000, F1_9: 0.6007 | F1_avg_2+9: 0.7397
Epoch  51 | TrainLoss: 0.4479 | Acc18: 0.5600, F1_18: 0.5531 | Acc_bin: 0.8812, F1_bin: 0.8811 | Acc_9: 0.5858, F1_9: 0.5872 | F1_avg_2+9: 0.7341
Epoch  52 | TrainLoss: 0.4813 | Acc18: 0.5902, F1_18: 0.5986 | Acc_bin: 0.8788, F1_bin: 0.8787 | Acc_9: 0.6142, F1_9: 0.6251 | F1_avg_2+9: 0.7519
Epoch  53 | TrainLoss: 0.4295 | Acc18: 0.5975, F1_18: 0.5925 | Acc_bin: 0.8880, F1_bin: 0.8879 | Acc_9: 0.6209, F1_9: 0.6194 | F1_avg_2+9: 0.7536
Epoch  54 | TrainLoss: 0.4383 | Acc18: 0.5742, F1_18: 0.5728 | Acc_bin: 0.8874, F1_bin: 0.8873 | Acc_9: 0.5982, F1_9: 0.5877 | F1_avg_2+9: 0.7375
Epoch  55 | TrainLoss: 0.4331 | Acc18: 0.5889, F1_18: 0.5765 | Acc_bin: 0.8751, F1_bin: 0.8750 | Acc_9: 0.6191, F1_9: 0.6079 | F1_avg_2+9: 0.7415
Epoch  56 | TrainLoss: 0.4053 | Acc18: 0.5895, F1_18: 0.5931 | Acc_bin: 0.8782, F1_bin: 0.8781 | Acc_9: 0.6111, F1_9: 0.6085 | F1_avg_2+9: 0.7433
Epoch  57 | TrainLoss: 0.4162 | Acc18: 0.5705, F1_18: 0.5567 | Acc_bin: 0.8658, F1_bin: 0.8657 | Acc_9: 0.5994, F1_9: 0.6005 | F1_avg_2+9: 0.7331
Epoch  58 | TrainLoss: 0.4194 | Acc18: 0.5895, F1_18: 0.5831 | Acc_bin: 0.8831, F1_bin: 0.8830 | Acc_9: 0.6148, F1_9: 0.5994 | F1_avg_2+9: 0.7412
Epoch  59 | TrainLoss: 0.4220 | Acc18: 0.5822, F1_18: 0.5873 | Acc_bin: 0.8775, F1_bin: 0.8775 | Acc_9: 0.6049, F1_9: 0.6091 | F1_avg_2+9: 0.7433
Epoch  60 | TrainLoss: 0.3893 | Acc18: 0.5674, F1_18: 0.5771 | Acc_bin: 0.8806, F1_bin: 0.8806 | Acc_9: 0.5926, F1_9: 0.5971 | F1_avg_2+9: 0.7388
Epoch  61 | TrainLoss: 0.3972 | Acc18: 0.5582, F1_18: 0.5742 | Acc_bin: 0.8800, F1_bin: 0.8800 | Acc_9: 0.5772, F1_9: 0.5824 | F1_avg_2+9: 0.7312
Epoch  62 | TrainLoss: 0.4142 | Acc18: 0.5625, F1_18: 0.5591 | Acc_bin: 0.8751, F1_bin: 0.8749 | Acc_9: 0.5926, F1_9: 0.5916 | F1_avg_2+9: 0.7332
Epoch  63 | TrainLoss: 0.3970 | Acc18: 0.5489, F1_18: 0.5562 | Acc_bin: 0.8683, F1_bin: 0.8683 | Acc_9: 0.5766, F1_9: 0.5691 | F1_avg_2+9: 0.7187
Epoch  64 | TrainLoss: 0.3912 | Acc18: 0.5698, F1_18: 0.5753 | Acc_bin: 0.8677, F1_bin: 0.8671 | Acc_9: 0.6006, F1_9: 0.6034 | F1_avg_2+9: 0.7353
Epoch  65 | TrainLoss: 0.3860 | Acc18: 0.5883, F1_18: 0.5931 | Acc_bin: 0.8868, F1_bin: 0.8867 | Acc_9: 0.6074, F1_9: 0.6053 | F1_avg_2+9: 0.7460
Epoch  66 | TrainLoss: 0.3906 | Acc18: 0.5711, F1_18: 0.5788 | Acc_bin: 0.8782, F1_bin: 0.8778 | Acc_9: 0.5963, F1_9: 0.6056 | F1_avg_2+9: 0.7417
Epoch  67 | TrainLoss: 0.3846 | Acc18: 0.5729, F1_18: 0.5813 | Acc_bin: 0.8622, F1_bin: 0.8619 | Acc_9: 0.6068, F1_9: 0.6109 | F1_avg_2+9: 0.7364
Epoch  68 | TrainLoss: 0.3860 | Acc18: 0.5649, F1_18: 0.5740 | Acc_bin: 0.8769, F1_bin: 0.8769 | Acc_9: 0.5883, F1_9: 0.5884 | F1_avg_2+9: 0.7326
Epoch  69 | TrainLoss: 0.3862 | Acc18: 0.5840, F1_18: 0.5781 | Acc_bin: 0.8769, F1_bin: 0.8764 | Acc_9: 0.6123, F1_9: 0.6136 | F1_avg_2+9: 0.7450
Epoch  70 | TrainLoss: 0.3738 | Acc18: 0.5692, F1_18: 0.5820 | Acc_bin: 0.8702, F1_bin: 0.8701 | Acc_9: 0.6031, F1_9: 0.6076 | F1_avg_2+9: 0.7389
Epoch  71 | TrainLoss: 0.3579 | Acc18: 0.5772, F1_18: 0.5926 | Acc_bin: 0.8849, F1_bin: 0.8848 | Acc_9: 0.5982, F1_9: 0.6057 | F1_avg_2+9: 0.7452
Epoch  72 | TrainLoss: 0.3681 | Acc18: 0.5791, F1_18: 0.5771 | Acc_bin: 0.8782, F1_bin: 0.8782 | Acc_9: 0.5969, F1_9: 0.5954 | F1_avg_2+9: 0.7368
Early stopping
=== Fold 4 ===
Epoch   1 | TrainLoss: 2.2500 | Acc18: 0.3376, F1_18: 0.2610 | Acc_bin: 0.7739, F1_bin: 0.7738 | Acc_9: 0.3805, F1_9: 0.3151 | F1_avg_2+9: 0.5445
Epoch   2 | TrainLoss: 1.6875 | Acc18: 0.4001, F1_18: 0.3624 | Acc_bin: 0.8002, F1_bin: 0.8002 | Acc_9: 0.4400, F1_9: 0.3953 | F1_avg_2+9: 0.5978
Epoch   3 | TrainLoss: 1.4602 | Acc18: 0.4026, F1_18: 0.3859 | Acc_bin: 0.8260, F1_bin: 0.8258 | Acc_9: 0.4412, F1_9: 0.4285 | F1_avg_2+9: 0.6271
Epoch   4 | TrainLoss: 1.3313 | Acc18: 0.4185, F1_18: 0.4137 | Acc_bin: 0.8290, F1_bin: 0.8284 | Acc_9: 0.4577, F1_9: 0.4501 | F1_avg_2+9: 0.6393
Epoch   5 | TrainLoss: 1.2296 | Acc18: 0.4491, F1_18: 0.4332 | Acc_bin: 0.8352, F1_bin: 0.8349 | Acc_9: 0.4859, F1_9: 0.4675 | F1_avg_2+9: 0.6512
Epoch   6 | TrainLoss: 1.1449 | Acc18: 0.4571, F1_18: 0.4423 | Acc_bin: 0.8425, F1_bin: 0.8417 | Acc_9: 0.4908, F1_9: 0.4809 | F1_avg_2+9: 0.6613
Epoch   7 | TrainLoss: 1.1104 | Acc18: 0.4577, F1_18: 0.4413 | Acc_bin: 0.8168, F1_bin: 0.8167 | Acc_9: 0.4884, F1_9: 0.4721 | F1_avg_2+9: 0.6444
Epoch   8 | TrainLoss: 1.0400 | Acc18: 0.4755, F1_18: 0.4527 | Acc_bin: 0.8431, F1_bin: 0.8431 | Acc_9: 0.5104, F1_9: 0.4859 | F1_avg_2+9: 0.6645
Epoch   9 | TrainLoss: 1.0158 | Acc18: 0.4657, F1_18: 0.4602 | Acc_bin: 0.8536, F1_bin: 0.8535 | Acc_9: 0.5037, F1_9: 0.5009 | F1_avg_2+9: 0.6772
Epoch  10 | TrainLoss: 0.9683 | Acc18: 0.4792, F1_18: 0.4750 | Acc_bin: 0.8529, F1_bin: 0.8526 | Acc_9: 0.5165, F1_9: 0.5060 | F1_avg_2+9: 0.6793
Epoch  11 | TrainLoss: 0.9332 | Acc18: 0.5159, F1_18: 0.5049 | Acc_bin: 0.8585, F1_bin: 0.8585 | Acc_9: 0.5478, F1_9: 0.5329 | F1_avg_2+9: 0.6957
Epoch  12 | TrainLoss: 0.8805 | Acc18: 0.4847, F1_18: 0.5003 | Acc_bin: 0.8431, F1_bin: 0.8430 | Acc_9: 0.5282, F1_9: 0.5229 | F1_avg_2+9: 0.6830
Epoch  13 | TrainLoss: 0.8527 | Acc18: 0.4951, F1_18: 0.4879 | Acc_bin: 0.8438, F1_bin: 0.8431 | Acc_9: 0.5263, F1_9: 0.5063 | F1_avg_2+9: 0.6747
Epoch  14 | TrainLoss: 0.8295 | Acc18: 0.5049, F1_18: 0.5088 | Acc_bin: 0.8499, F1_bin: 0.8496 | Acc_9: 0.5392, F1_9: 0.5272 | F1_avg_2+9: 0.6884
Epoch  15 | TrainLoss: 0.7946 | Acc18: 0.5092, F1_18: 0.5124 | Acc_bin: 0.8578, F1_bin: 0.8578 | Acc_9: 0.5429, F1_9: 0.5335 | F1_avg_2+9: 0.6956
Epoch  16 | TrainLoss: 0.7876 | Acc18: 0.5227, F1_18: 0.5189 | Acc_bin: 0.8536, F1_bin: 0.8530 | Acc_9: 0.5551, F1_9: 0.5434 | F1_avg_2+9: 0.6982
Epoch  17 | TrainLoss: 0.7688 | Acc18: 0.4975, F1_18: 0.4913 | Acc_bin: 0.8223, F1_bin: 0.8223 | Acc_9: 0.5349, F1_9: 0.5118 | F1_avg_2+9: 0.6670
Epoch  18 | TrainLoss: 0.7574 | Acc18: 0.5165, F1_18: 0.5157 | Acc_bin: 0.8505, F1_bin: 0.8505 | Acc_9: 0.5484, F1_9: 0.5330 | F1_avg_2+9: 0.6917
Epoch  19 | TrainLoss: 0.7277 | Acc18: 0.5294, F1_18: 0.5357 | Acc_bin: 0.8615, F1_bin: 0.8611 | Acc_9: 0.5600, F1_9: 0.5458 | F1_avg_2+9: 0.7035
Epoch  20 | TrainLoss: 0.7163 | Acc18: 0.5123, F1_18: 0.5230 | Acc_bin: 0.8578, F1_bin: 0.8568 | Acc_9: 0.5435, F1_9: 0.5373 | F1_avg_2+9: 0.6970
Epoch  21 | TrainLoss: 0.6965 | Acc18: 0.4822, F1_18: 0.4579 | Acc_bin: 0.8450, F1_bin: 0.8435 | Acc_9: 0.5184, F1_9: 0.4958 | F1_avg_2+9: 0.6696
Epoch  22 | TrainLoss: 0.6883 | Acc18: 0.5055, F1_18: 0.5190 | Acc_bin: 0.8658, F1_bin: 0.8655 | Acc_9: 0.5331, F1_9: 0.5289 | F1_avg_2+9: 0.6972
Epoch  23 | TrainLoss: 0.6689 | Acc18: 0.5141, F1_18: 0.5097 | Acc_bin: 0.8401, F1_bin: 0.8400 | Acc_9: 0.5533, F1_9: 0.5364 | F1_avg_2+9: 0.6882
Epoch  24 | TrainLoss: 0.6722 | Acc18: 0.5331, F1_18: 0.5442 | Acc_bin: 0.8615, F1_bin: 0.8611 | Acc_9: 0.5607, F1_9: 0.5575 | F1_avg_2+9: 0.7093
Epoch  25 | TrainLoss: 0.6482 | Acc18: 0.5080, F1_18: 0.5192 | Acc_bin: 0.8450, F1_bin: 0.8445 | Acc_9: 0.5386, F1_9: 0.5260 | F1_avg_2+9: 0.6853
Epoch  26 | TrainLoss: 0.6606 | Acc18: 0.5227, F1_18: 0.5251 | Acc_bin: 0.8591, F1_bin: 0.8591 | Acc_9: 0.5527, F1_9: 0.5471 | F1_avg_2+9: 0.7031
Epoch  27 | TrainLoss: 0.6267 | Acc18: 0.5067, F1_18: 0.5143 | Acc_bin: 0.8413, F1_bin: 0.8413 | Acc_9: 0.5472, F1_9: 0.5301 | F1_avg_2+9: 0.6857
Epoch  28 | TrainLoss: 0.6148 | Acc18: 0.5214, F1_18: 0.5225 | Acc_bin: 0.8431, F1_bin: 0.8430 | Acc_9: 0.5527, F1_9: 0.5448 | F1_avg_2+9: 0.6939
Epoch  29 | TrainLoss: 0.5966 | Acc18: 0.5386, F1_18: 0.5327 | Acc_bin: 0.8474, F1_bin: 0.8474 | Acc_9: 0.5735, F1_9: 0.5587 | F1_avg_2+9: 0.7030
Epoch  30 | TrainLoss: 0.5971 | Acc18: 0.5172, F1_18: 0.5217 | Acc_bin: 0.8474, F1_bin: 0.8466 | Acc_9: 0.5551, F1_9: 0.5536 | F1_avg_2+9: 0.7001
Epoch  31 | TrainLoss: 0.5845 | Acc18: 0.5037, F1_18: 0.5009 | Acc_bin: 0.8438, F1_bin: 0.8437 | Acc_9: 0.5460, F1_9: 0.5384 | F1_avg_2+9: 0.6911
Epoch  32 | TrainLoss: 0.5769 | Acc18: 0.5306, F1_18: 0.5318 | Acc_bin: 0.8370, F1_bin: 0.8369 | Acc_9: 0.5699, F1_9: 0.5541 | F1_avg_2+9: 0.6955
Epoch  33 | TrainLoss: 0.5709 | Acc18: 0.5196, F1_18: 0.5261 | Acc_bin: 0.8572, F1_bin: 0.8571 | Acc_9: 0.5558, F1_9: 0.5502 | F1_avg_2+9: 0.7036
Epoch  34 | TrainLoss: 0.5592 | Acc18: 0.5184, F1_18: 0.5260 | Acc_bin: 0.8560, F1_bin: 0.8557 | Acc_9: 0.5484, F1_9: 0.5392 | F1_avg_2+9: 0.6975
Epoch  35 | TrainLoss: 0.5414 | Acc18: 0.5227, F1_18: 0.5276 | Acc_bin: 0.8290, F1_bin: 0.8290 | Acc_9: 0.5625, F1_9: 0.5548 | F1_avg_2+9: 0.6919
Epoch  36 | TrainLoss: 0.5427 | Acc18: 0.5110, F1_18: 0.5162 | Acc_bin: 0.8370, F1_bin: 0.8370 | Acc_9: 0.5472, F1_9: 0.5347 | F1_avg_2+9: 0.6858
Epoch  37 | TrainLoss: 0.5381 | Acc18: 0.5233, F1_18: 0.5394 | Acc_bin: 0.8474, F1_bin: 0.8471 | Acc_9: 0.5527, F1_9: 0.5501 | F1_avg_2+9: 0.6986
Epoch  38 | TrainLoss: 0.5415 | Acc18: 0.5325, F1_18: 0.5361 | Acc_bin: 0.8585, F1_bin: 0.8581 | Acc_9: 0.5625, F1_9: 0.5548 | F1_avg_2+9: 0.7064
Epoch  39 | TrainLoss: 0.5317 | Acc18: 0.5012, F1_18: 0.5240 | Acc_bin: 0.8401, F1_bin: 0.8399 | Acc_9: 0.5349, F1_9: 0.5387 | F1_avg_2+9: 0.6893
Epoch  40 | TrainLoss: 0.5253 | Acc18: 0.5368, F1_18: 0.5315 | Acc_bin: 0.8529, F1_bin: 0.8524 | Acc_9: 0.5674, F1_9: 0.5529 | F1_avg_2+9: 0.7027
Epoch  41 | TrainLoss: 0.5087 | Acc18: 0.5202, F1_18: 0.5157 | Acc_bin: 0.8358, F1_bin: 0.8358 | Acc_9: 0.5521, F1_9: 0.5315 | F1_avg_2+9: 0.6837
Epoch  42 | TrainLoss: 0.5260 | Acc18: 0.5288, F1_18: 0.5297 | Acc_bin: 0.8529, F1_bin: 0.8528 | Acc_9: 0.5576, F1_9: 0.5501 | F1_avg_2+9: 0.7015
Epoch  43 | TrainLoss: 0.5184 | Acc18: 0.5319, F1_18: 0.5301 | Acc_bin: 0.8572, F1_bin: 0.8572 | Acc_9: 0.5705, F1_9: 0.5580 | F1_avg_2+9: 0.7076
Epoch  44 | TrainLoss: 0.4978 | Acc18: 0.5282, F1_18: 0.5417 | Acc_bin: 0.8505, F1_bin: 0.8504 | Acc_9: 0.5625, F1_9: 0.5595 | F1_avg_2+9: 0.7050
Early stopping
=== Fold 5 ===
Epoch   1 | TrainLoss: 2.3081 | Acc18: 0.4008, F1_18: 0.3485 | Acc_bin: 0.8080, F1_bin: 0.8078 | Acc_9: 0.4331, F1_9: 0.3742 | F1_avg_2+9: 0.5910
Epoch   2 | TrainLoss: 1.7268 | Acc18: 0.4406, F1_18: 0.4227 | Acc_bin: 0.8691, F1_bin: 0.8691 | Acc_9: 0.4642, F1_9: 0.4478 | F1_avg_2+9: 0.6584
Epoch   3 | TrainLoss: 1.5262 | Acc18: 0.4666, F1_18: 0.4497 | Acc_bin: 0.8927, F1_bin: 0.8925 | Acc_9: 0.4850, F1_9: 0.4768 | F1_avg_2+9: 0.6847
Epoch   4 | TrainLoss: 1.3910 | Acc18: 0.4544, F1_18: 0.4541 | Acc_bin: 0.8691, F1_bin: 0.8688 | Acc_9: 0.4752, F1_9: 0.4732 | F1_avg_2+9: 0.6710
Epoch   5 | TrainLoss: 1.2888 | Acc18: 0.4850, F1_18: 0.4840 | Acc_bin: 0.8818, F1_bin: 0.8812 | Acc_9: 0.5156, F1_9: 0.5062 | F1_avg_2+9: 0.6937
Epoch   6 | TrainLoss: 1.2167 | Acc18: 0.4735, F1_18: 0.4600 | Acc_bin: 0.8916, F1_bin: 0.8911 | Acc_9: 0.4960, F1_9: 0.4962 | F1_avg_2+9: 0.6936
Epoch   7 | TrainLoss: 1.1626 | Acc18: 0.4792, F1_18: 0.4695 | Acc_bin: 0.8904, F1_bin: 0.8902 | Acc_9: 0.5000, F1_9: 0.4964 | F1_avg_2+9: 0.6933
Epoch   8 | TrainLoss: 1.1269 | Acc18: 0.5017, F1_18: 0.5030 | Acc_bin: 0.8962, F1_bin: 0.8961 | Acc_9: 0.5271, F1_9: 0.5236 | F1_avg_2+9: 0.7098
Epoch   9 | TrainLoss: 1.0993 | Acc18: 0.5040, F1_18: 0.4918 | Acc_bin: 0.9014, F1_bin: 0.9013 | Acc_9: 0.5231, F1_9: 0.5091 | F1_avg_2+9: 0.7052
Epoch  10 | TrainLoss: 1.0334 | Acc18: 0.4879, F1_18: 0.4923 | Acc_bin: 0.8697, F1_bin: 0.8681 | Acc_9: 0.5254, F1_9: 0.5146 | F1_avg_2+9: 0.6914
Epoch  11 | TrainLoss: 0.9842 | Acc18: 0.5127, F1_18: 0.5066 | Acc_bin: 0.9031, F1_bin: 0.9026 | Acc_9: 0.5340, F1_9: 0.5322 | F1_avg_2+9: 0.7174
Epoch  12 | TrainLoss: 0.9612 | Acc18: 0.5392, F1_18: 0.5425 | Acc_bin: 0.8950, F1_bin: 0.8948 | Acc_9: 0.5588, F1_9: 0.5552 | F1_avg_2+9: 0.7250
Epoch  13 | TrainLoss: 0.9008 | Acc18: 0.5582, F1_18: 0.5448 | Acc_bin: 0.9002, F1_bin: 0.9000 | Acc_9: 0.5813, F1_9: 0.5639 | F1_avg_2+9: 0.7320
Epoch  14 | TrainLoss: 0.8898 | Acc18: 0.5646, F1_18: 0.5724 | Acc_bin: 0.9089, F1_bin: 0.9085 | Acc_9: 0.5905, F1_9: 0.5851 | F1_avg_2+9: 0.7468
Epoch  15 | TrainLoss: 0.8538 | Acc18: 0.5231, F1_18: 0.5244 | Acc_bin: 0.8956, F1_bin: 0.8956 | Acc_9: 0.5456, F1_9: 0.5404 | F1_avg_2+9: 0.7180
Epoch  16 | TrainLoss: 0.8074 | Acc18: 0.5248, F1_18: 0.5294 | Acc_bin: 0.8991, F1_bin: 0.8987 | Acc_9: 0.5507, F1_9: 0.5492 | F1_avg_2+9: 0.7240
Epoch  17 | TrainLoss: 0.8027 | Acc18: 0.5461, F1_18: 0.5389 | Acc_bin: 0.8973, F1_bin: 0.8971 | Acc_9: 0.5681, F1_9: 0.5477 | F1_avg_2+9: 0.7224
Epoch  18 | TrainLoss: 0.8048 | Acc18: 0.5415, F1_18: 0.5598 | Acc_bin: 0.8858, F1_bin: 0.8857 | Acc_9: 0.5738, F1_9: 0.5773 | F1_avg_2+9: 0.7315
Epoch  19 | TrainLoss: 0.7581 | Acc18: 0.5554, F1_18: 0.5577 | Acc_bin: 0.8899, F1_bin: 0.8898 | Acc_9: 0.5767, F1_9: 0.5733 | F1_avg_2+9: 0.7316
Epoch  20 | TrainLoss: 0.7275 | Acc18: 0.5657, F1_18: 0.5667 | Acc_bin: 0.9175, F1_bin: 0.9174 | Acc_9: 0.5773, F1_9: 0.5749 | F1_avg_2+9: 0.7461
Epoch  21 | TrainLoss: 0.7277 | Acc18: 0.5709, F1_18: 0.5779 | Acc_bin: 0.9198, F1_bin: 0.9197 | Acc_9: 0.5882, F1_9: 0.5864 | F1_avg_2+9: 0.7530
Epoch  22 | TrainLoss: 0.6901 | Acc18: 0.5767, F1_18: 0.5767 | Acc_bin: 0.9170, F1_bin: 0.9166 | Acc_9: 0.5980, F1_9: 0.5894 | F1_avg_2+9: 0.7530
Epoch  23 | TrainLoss: 0.6819 | Acc18: 0.5536, F1_18: 0.5384 | Acc_bin: 0.9123, F1_bin: 0.9122 | Acc_9: 0.5681, F1_9: 0.5497 | F1_avg_2+9: 0.7310
Epoch  24 | TrainLoss: 0.6953 | Acc18: 0.5859, F1_18: 0.6034 | Acc_bin: 0.9146, F1_bin: 0.9145 | Acc_9: 0.6027, F1_9: 0.6044 | F1_avg_2+9: 0.7594
Epoch  25 | TrainLoss: 0.6717 | Acc18: 0.5669, F1_18: 0.5552 | Acc_bin: 0.9043, F1_bin: 0.9041 | Acc_9: 0.5928, F1_9: 0.5840 | F1_avg_2+9: 0.7441
Epoch  26 | TrainLoss: 0.6915 | Acc18: 0.5709, F1_18: 0.5693 | Acc_bin: 0.9037, F1_bin: 0.9031 | Acc_9: 0.5969, F1_9: 0.5873 | F1_avg_2+9: 0.7452
Epoch  27 | TrainLoss: 0.6334 | Acc18: 0.5830, F1_18: 0.5877 | Acc_bin: 0.9198, F1_bin: 0.9197 | Acc_9: 0.6003, F1_9: 0.5878 | F1_avg_2+9: 0.7538
Epoch  28 | TrainLoss: 0.6244 | Acc18: 0.5502, F1_18: 0.5717 | Acc_bin: 0.9077, F1_bin: 0.9071 | Acc_9: 0.5732, F1_9: 0.5780 | F1_avg_2+9: 0.7425
Epoch  29 | TrainLoss: 0.6193 | Acc18: 0.5813, F1_18: 0.5916 | Acc_bin: 0.9100, F1_bin: 0.9099 | Acc_9: 0.6044, F1_9: 0.5988 | F1_avg_2+9: 0.7544
Epoch  30 | TrainLoss: 0.6083 | Acc18: 0.5484, F1_18: 0.5609 | Acc_bin: 0.9100, F1_bin: 0.9093 | Acc_9: 0.5698, F1_9: 0.5743 | F1_avg_2+9: 0.7418
Epoch  31 | TrainLoss: 0.6165 | Acc18: 0.5704, F1_18: 0.5680 | Acc_bin: 0.9164, F1_bin: 0.9160 | Acc_9: 0.5940, F1_9: 0.5869 | F1_avg_2+9: 0.7515
Epoch  32 | TrainLoss: 0.5926 | Acc18: 0.5721, F1_18: 0.5831 | Acc_bin: 0.9083, F1_bin: 0.9081 | Acc_9: 0.5969, F1_9: 0.5910 | F1_avg_2+9: 0.7495
Epoch  33 | TrainLoss: 0.5858 | Acc18: 0.5669, F1_18: 0.5794 | Acc_bin: 0.9187, F1_bin: 0.9185 | Acc_9: 0.5848, F1_9: 0.5885 | F1_avg_2+9: 0.7535
Epoch  34 | TrainLoss: 0.5630 | Acc18: 0.5617, F1_18: 0.5725 | Acc_bin: 0.8979, F1_bin: 0.8977 | Acc_9: 0.5790, F1_9: 0.5786 | F1_avg_2+9: 0.7382
Epoch  35 | TrainLoss: 0.5923 | Acc18: 0.5692, F1_18: 0.5726 | Acc_bin: 0.9031, F1_bin: 0.9025 | Acc_9: 0.5905, F1_9: 0.5859 | F1_avg_2+9: 0.7442
Epoch  36 | TrainLoss: 0.5487 | Acc18: 0.5761, F1_18: 0.5843 | Acc_bin: 0.9170, F1_bin: 0.9168 | Acc_9: 0.6009, F1_9: 0.6024 | F1_avg_2+9: 0.7596
Epoch  37 | TrainLoss: 0.5379 | Acc18: 0.5761, F1_18: 0.5857 | Acc_bin: 0.8991, F1_bin: 0.8990 | Acc_9: 0.6021, F1_9: 0.5993 | F1_avg_2+9: 0.7492
Epoch  38 | TrainLoss: 0.5701 | Acc18: 0.5542, F1_18: 0.5517 | Acc_bin: 0.8962, F1_bin: 0.8959 | Acc_9: 0.5784, F1_9: 0.5705 | F1_avg_2+9: 0.7332
Epoch  39 | TrainLoss: 0.5263 | Acc18: 0.5750, F1_18: 0.5763 | Acc_bin: 0.9123, F1_bin: 0.9119 | Acc_9: 0.5952, F1_9: 0.5957 | F1_avg_2+9: 0.7538
Epoch  40 | TrainLoss: 0.5260 | Acc18: 0.5681, F1_18: 0.5838 | Acc_bin: 0.9054, F1_bin: 0.9051 | Acc_9: 0.5900, F1_9: 0.5862 | F1_avg_2+9: 0.7456
Epoch  41 | TrainLoss: 0.5410 | Acc18: 0.5704, F1_18: 0.5770 | Acc_bin: 0.9089, F1_bin: 0.9086 | Acc_9: 0.5911, F1_9: 0.5900 | F1_avg_2+9: 0.7493
Epoch  42 | TrainLoss: 0.5001 | Acc18: 0.5646, F1_18: 0.5773 | Acc_bin: 0.9112, F1_bin: 0.9109 | Acc_9: 0.5836, F1_9: 0.5778 | F1_avg_2+9: 0.7443
Epoch  43 | TrainLoss: 0.5229 | Acc18: 0.5606, F1_18: 0.5691 | Acc_bin: 0.8950, F1_bin: 0.8940 | Acc_9: 0.5871, F1_9: 0.5879 | F1_avg_2+9: 0.7409
Epoch  44 | TrainLoss: 0.5248 | Acc18: 0.5646, F1_18: 0.5703 | Acc_bin: 0.8956, F1_bin: 0.8954 | Acc_9: 0.5871, F1_9: 0.5701 | F1_avg_2+9: 0.7328
Early stopping

(venv) C:\dev\sensor_project>train_test_18class_22.py

(venv) C:\dev\sensor_project>train_test_18class_22.py

(venv) C:\dev\sensor_project>train_test_18class_22.py
=== Fold 1 ===
Traceback (most recent call last):
  File "C:\dev\sensor_project\train_test_18class_22.py", line 106, in <module>
    model = TwoBranchModel(imu_dim, tof_dim, pad_len, n_classes=len(le.classes_)).to(device)
            ^^^^^^^^^^^^^^
NameError: name 'TwoBranchModel' is not defined

(venv) C:\dev\sensor_project>train_test_18class_22.py
=== Fold 1 ===
Traceback (most recent call last):
  File "C:\dev\sensor_project\train_test_18class_22.py", line 104, in <module>
    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
                              ^^^^^^^^^^^^^^^
NameError: name 'SequenceDataset' is not defined

(venv) C:\dev\sensor_project>train_test_18class_22.py
=== Fold 1 ===
Epoch   1 | TrainLoss: 2.2772 | Acc18: 0.4555, F1_18: 0.3672 | Acc_bin: 0.8370, F1_bin: 0.8370 | Acc_9: 0.4810, F1_9: 0.3971 | F1_avg_2+9: 0.6171
Epoch   2 | TrainLoss: 1.7337 | Acc18: 0.5445, F1_18: 0.4966 | Acc_bin: 0.8815, F1_bin: 0.8812 | Acc_9: 0.5838, F1_9: 0.5482 | F1_avg_2+9: 0.7147
Epoch   3 | TrainLoss: 1.4967 | Acc18: 0.5596, F1_18: 0.5411 | Acc_bin: 0.8940, F1_bin: 0.8939 | Acc_9: 0.5864, F1_9: 0.5693 | F1_avg_2+9: 0.7316
Epoch   4 | TrainLoss: 1.3500 | Acc18: 0.5818, F1_18: 0.5888 | Acc_bin: 0.9123, F1_bin: 0.9120 | Acc_9: 0.6113, F1_9: 0.6156 | F1_avg_2+9: 0.7638
Epoch   5 | TrainLoss: 1.2407 | Acc18: 0.5746, F1_18: 0.5531 | Acc_bin: 0.9182, F1_bin: 0.9178 | Acc_9: 0.6021, F1_9: 0.5896 | F1_avg_2+9: 0.7537
Epoch   6 | TrainLoss: 1.1688 | Acc18: 0.6139, F1_18: 0.6138 | Acc_bin: 0.9208, F1_bin: 0.9203 | Acc_9: 0.6381, F1_9: 0.6366 | F1_avg_2+9: 0.7785
Epoch   7 | TrainLoss: 1.1038 | Acc18: 0.5982, F1_18: 0.5873 | Acc_bin: 0.9130, F1_bin: 0.9118 | Acc_9: 0.6217, F1_9: 0.6196 | F1_avg_2+9: 0.7657
Epoch   8 | TrainLoss: 1.0448 | Acc18: 0.6250, F1_18: 0.6330 | Acc_bin: 0.9208, F1_bin: 0.9200 | Acc_9: 0.6492, F1_9: 0.6558 | F1_avg_2+9: 0.7879
Epoch   9 | TrainLoss: 0.9896 | Acc18: 0.6178, F1_18: 0.6083 | Acc_bin: 0.9116, F1_bin: 0.9110 | Acc_9: 0.6394, F1_9: 0.6355 | F1_avg_2+9: 0.7732
Epoch  10 | TrainLoss: 0.9563 | Acc18: 0.6178, F1_18: 0.6249 | Acc_bin: 0.9169, F1_bin: 0.9159 | Acc_9: 0.6433, F1_9: 0.6543 | F1_avg_2+9: 0.7851
Epoch  11 | TrainLoss: 0.9288 | Acc18: 0.6152, F1_18: 0.6082 | Acc_bin: 0.9149, F1_bin: 0.9147 | Acc_9: 0.6322, F1_9: 0.6394 | F1_avg_2+9: 0.7770
Epoch  12 | TrainLoss: 0.9101 | Acc18: 0.6296, F1_18: 0.6351 | Acc_bin: 0.9228, F1_bin: 0.9221 | Acc_9: 0.6577, F1_9: 0.6541 | F1_avg_2+9: 0.7881
Epoch  13 | TrainLoss: 0.8660 | Acc18: 0.6217, F1_18: 0.6228 | Acc_bin: 0.9116, F1_bin: 0.9103 | Acc_9: 0.6440, F1_9: 0.6454 | F1_avg_2+9: 0.7778
Epoch  14 | TrainLoss: 0.8466 | Acc18: 0.6348, F1_18: 0.6280 | Acc_bin: 0.9280, F1_bin: 0.9278 | Acc_9: 0.6551, F1_9: 0.6531 | F1_avg_2+9: 0.7904
Epoch  15 | TrainLoss: 0.8409 | Acc18: 0.6230, F1_18: 0.6222 | Acc_bin: 0.9247, F1_bin: 0.9245 | Acc_9: 0.6433, F1_9: 0.6515 | F1_avg_2+9: 0.7880
Epoch  16 | TrainLoss: 0.8175 | Acc18: 0.5818, F1_18: 0.5824 | Acc_bin: 0.9247, F1_bin: 0.9239 | Acc_9: 0.6021, F1_9: 0.6133 | F1_avg_2+9: 0.7686
Epoch  17 | TrainLoss: 0.7793 | Acc18: 0.6217, F1_18: 0.6283 | Acc_bin: 0.9260, F1_bin: 0.9259 | Acc_9: 0.6427, F1_9: 0.6477 | F1_avg_2+9: 0.7868
Epoch  18 | TrainLoss: 0.7696 | Acc18: 0.6263, F1_18: 0.6391 | Acc_bin: 0.9149, F1_bin: 0.9139 | Acc_9: 0.6545, F1_9: 0.6574 | F1_avg_2+9: 0.7856
Epoch  19 | TrainLoss: 0.7371 | Acc18: 0.6126, F1_18: 0.6246 | Acc_bin: 0.9247, F1_bin: 0.9244 | Acc_9: 0.6329, F1_9: 0.6446 | F1_avg_2+9: 0.7845
Epoch  20 | TrainLoss: 0.7329 | Acc18: 0.6420, F1_18: 0.6523 | Acc_bin: 0.9267, F1_bin: 0.9263 | Acc_9: 0.6682, F1_9: 0.6755 | F1_avg_2+9: 0.8009
Epoch  21 | TrainLoss: 0.7144 | Acc18: 0.6270, F1_18: 0.6395 | Acc_bin: 0.9254, F1_bin: 0.9252 | Acc_9: 0.6545, F1_9: 0.6633 | F1_avg_2+9: 0.7942
Epoch  22 | TrainLoss: 0.7054 | Acc18: 0.6139, F1_18: 0.6157 | Acc_bin: 0.9306, F1_bin: 0.9303 | Acc_9: 0.6368, F1_9: 0.6418 | F1_avg_2+9: 0.7861
Epoch  23 | TrainLoss: 0.6848 | Acc18: 0.6230, F1_18: 0.6176 | Acc_bin: 0.9195, F1_bin: 0.9189 | Acc_9: 0.6492, F1_9: 0.6457 | F1_avg_2+9: 0.7823
Epoch  24 | TrainLoss: 0.6909 | Acc18: 0.6270, F1_18: 0.6340 | Acc_bin: 0.9188, F1_bin: 0.9186 | Acc_9: 0.6433, F1_9: 0.6484 | F1_avg_2+9: 0.7835
Epoch  25 | TrainLoss: 0.6713 | Acc18: 0.6414, F1_18: 0.6526 | Acc_bin: 0.9254, F1_bin: 0.9250 | Acc_9: 0.6603, F1_9: 0.6683 | F1_avg_2+9: 0.7967
Epoch  26 | TrainLoss: 0.6589 | Acc18: 0.6329, F1_18: 0.6452 | Acc_bin: 0.9287, F1_bin: 0.9283 | Acc_9: 0.6590, F1_9: 0.6611 | F1_avg_2+9: 0.7947
Epoch  27 | TrainLoss: 0.6304 | Acc18: 0.6198, F1_18: 0.6296 | Acc_bin: 0.9254, F1_bin: 0.9252 | Acc_9: 0.6453, F1_9: 0.6480 | F1_avg_2+9: 0.7866
Epoch  28 | TrainLoss: 0.6425 | Acc18: 0.6368, F1_18: 0.6497 | Acc_bin: 0.9208, F1_bin: 0.9202 | Acc_9: 0.6636, F1_9: 0.6713 | F1_avg_2+9: 0.7958
Epoch  29 | TrainLoss: 0.5984 | Acc18: 0.6099, F1_18: 0.6280 | Acc_bin: 0.9025, F1_bin: 0.9009 | Acc_9: 0.6466, F1_9: 0.6535 | F1_avg_2+9: 0.7772
Epoch  30 | TrainLoss: 0.6146 | Acc18: 0.6368, F1_18: 0.6500 | Acc_bin: 0.9241, F1_bin: 0.9240 | Acc_9: 0.6564, F1_9: 0.6587 | F1_avg_2+9: 0.7913
Epoch  31 | TrainLoss: 0.5917 | Acc18: 0.6394, F1_18: 0.6408 | Acc_bin: 0.9267, F1_bin: 0.9264 | Acc_9: 0.6682, F1_9: 0.6623 | F1_avg_2+9: 0.7944
Epoch  32 | TrainLoss: 0.5875 | Acc18: 0.6414, F1_18: 0.6483 | Acc_bin: 0.9306, F1_bin: 0.9303 | Acc_9: 0.6630, F1_9: 0.6601 | F1_avg_2+9: 0.7952
Epoch  33 | TrainLoss: 0.5728 | Acc18: 0.6361, F1_18: 0.6460 | Acc_bin: 0.9234, F1_bin: 0.9230 | Acc_9: 0.6616, F1_9: 0.6629 | F1_avg_2+9: 0.7930
Epoch  34 | TrainLoss: 0.5639 | Acc18: 0.6276, F1_18: 0.6343 | Acc_bin: 0.9208, F1_bin: 0.9200 | Acc_9: 0.6499, F1_9: 0.6578 | F1_avg_2+9: 0.7889
Epoch  35 | TrainLoss: 0.5695 | Acc18: 0.6263, F1_18: 0.6362 | Acc_bin: 0.9267, F1_bin: 0.9260 | Acc_9: 0.6440, F1_9: 0.6529 | F1_avg_2+9: 0.7895
Epoch  36 | TrainLoss: 0.5550 | Acc18: 0.6165, F1_18: 0.6426 | Acc_bin: 0.9241, F1_bin: 0.9236 | Acc_9: 0.6414, F1_9: 0.6543 | F1_avg_2+9: 0.7889
Epoch  37 | TrainLoss: 0.5423 | Acc18: 0.6466, F1_18: 0.6418 | Acc_bin: 0.9228, F1_bin: 0.9227 | Acc_9: 0.6682, F1_9: 0.6620 | F1_avg_2+9: 0.7924
Epoch  38 | TrainLoss: 0.5123 | Acc18: 0.6440, F1_18: 0.6574 | Acc_bin: 0.9274, F1_bin: 0.9270 | Acc_9: 0.6695, F1_9: 0.6806 | F1_avg_2+9: 0.8038
Epoch  39 | TrainLoss: 0.5004 | Acc18: 0.6257, F1_18: 0.6153 | Acc_bin: 0.9169, F1_bin: 0.9163 | Acc_9: 0.6518, F1_9: 0.6456 | F1_avg_2+9: 0.7809
Epoch  40 | TrainLoss: 0.5318 | Acc18: 0.6394, F1_18: 0.6597 | Acc_bin: 0.9208, F1_bin: 0.9202 | Acc_9: 0.6636, F1_9: 0.6707 | F1_avg_2+9: 0.7955
Epoch  41 | TrainLoss: 0.5106 | Acc18: 0.6119, F1_18: 0.6104 | Acc_bin: 0.9195, F1_bin: 0.9192 | Acc_9: 0.6342, F1_9: 0.6244 | F1_avg_2+9: 0.7718
Epoch  42 | TrainLoss: 0.5219 | Acc18: 0.6027, F1_18: 0.5935 | Acc_bin: 0.9130, F1_bin: 0.9129 | Acc_9: 0.6309, F1_9: 0.6371 | F1_avg_2+9: 0.7750
Epoch  43 | TrainLoss: 0.4844 | Acc18: 0.6139, F1_18: 0.6309 | Acc_bin: 0.9221, F1_bin: 0.9211 | Acc_9: 0.6374, F1_9: 0.6434 | F1_avg_2+9: 0.7822
Epoch  44 | TrainLoss: 0.5042 | Acc18: 0.6531, F1_18: 0.6595 | Acc_bin: 0.9313, F1_bin: 0.9309 | Acc_9: 0.6774, F1_9: 0.6730 | F1_avg_2+9: 0.8020
Epoch  45 | TrainLoss: 0.4768 | Acc18: 0.6113, F1_18: 0.6338 | Acc_bin: 0.9215, F1_bin: 0.9208 | Acc_9: 0.6335, F1_9: 0.6506 | F1_avg_2+9: 0.7857
Epoch  46 | TrainLoss: 0.4851 | Acc18: 0.6355, F1_18: 0.6519 | Acc_bin: 0.9247, F1_bin: 0.9242 | Acc_9: 0.6610, F1_9: 0.6578 | F1_avg_2+9: 0.7910
Epoch  47 | TrainLoss: 0.4724 | Acc18: 0.6459, F1_18: 0.6504 | Acc_bin: 0.9254, F1_bin: 0.9248 | Acc_9: 0.6649, F1_9: 0.6574 | F1_avg_2+9: 0.7911
Epoch  48 | TrainLoss: 0.4620 | Acc18: 0.6512, F1_18: 0.6524 | Acc_bin: 0.9319, F1_bin: 0.9316 | Acc_9: 0.6728, F1_9: 0.6652 | F1_avg_2+9: 0.7984
Epoch  49 | TrainLoss: 0.4648 | Acc18: 0.6414, F1_18: 0.6501 | Acc_bin: 0.9215, F1_bin: 0.9209 | Acc_9: 0.6584, F1_9: 0.6616 | F1_avg_2+9: 0.7912
Epoch  50 | TrainLoss: 0.4660 | Acc18: 0.6453, F1_18: 0.6485 | Acc_bin: 0.9202, F1_bin: 0.9199 | Acc_9: 0.6741, F1_9: 0.6774 | F1_avg_2+9: 0.7987
Epoch  51 | TrainLoss: 0.4751 | Acc18: 0.6283, F1_18: 0.6318 | Acc_bin: 0.9306, F1_bin: 0.9303 | Acc_9: 0.6453, F1_9: 0.6466 | F1_avg_2+9: 0.7884
Epoch  52 | TrainLoss: 0.4330 | Acc18: 0.6420, F1_18: 0.6434 | Acc_bin: 0.9175, F1_bin: 0.9164 | Acc_9: 0.6623, F1_9: 0.6615 | F1_avg_2+9: 0.7890
Epoch  53 | TrainLoss: 0.4266 | Acc18: 0.6368, F1_18: 0.6498 | Acc_bin: 0.9234, F1_bin: 0.9231 | Acc_9: 0.6630, F1_9: 0.6681 | F1_avg_2+9: 0.7956
Epoch  54 | TrainLoss: 0.4370 | Acc18: 0.6453, F1_18: 0.6559 | Acc_bin: 0.9228, F1_bin: 0.9222 | Acc_9: 0.6688, F1_9: 0.6630 | F1_avg_2+9: 0.7926
Epoch  55 | TrainLoss: 0.4097 | Acc18: 0.6603, F1_18: 0.6736 | Acc_bin: 0.9260, F1_bin: 0.9257 | Acc_9: 0.6832, F1_9: 0.6872 | F1_avg_2+9: 0.8064
Epoch  56 | TrainLoss: 0.4503 | Acc18: 0.6577, F1_18: 0.6428 | Acc_bin: 0.9346, F1_bin: 0.9342 | Acc_9: 0.6741, F1_9: 0.6654 | F1_avg_2+9: 0.7998
Epoch  57 | TrainLoss: 0.4205 | Acc18: 0.6374, F1_18: 0.6417 | Acc_bin: 0.9149, F1_bin: 0.9145 | Acc_9: 0.6630, F1_9: 0.6575 | F1_avg_2+9: 0.7860
Epoch  58 | TrainLoss: 0.4353 | Acc18: 0.6486, F1_18: 0.6714 | Acc_bin: 0.9274, F1_bin: 0.9270 | Acc_9: 0.6669, F1_9: 0.6788 | F1_avg_2+9: 0.8029
Epoch  59 | TrainLoss: 0.4050 | Acc18: 0.6551, F1_18: 0.6853 | Acc_bin: 0.9247, F1_bin: 0.9242 | Acc_9: 0.6774, F1_9: 0.6968 | F1_avg_2+9: 0.8105
Epoch  60 | TrainLoss: 0.4179 | Acc18: 0.6518, F1_18: 0.6740 | Acc_bin: 0.9280, F1_bin: 0.9273 | Acc_9: 0.6734, F1_9: 0.6866 | F1_avg_2+9: 0.8069
Epoch  61 | TrainLoss: 0.4256 | Acc18: 0.6479, F1_18: 0.6622 | Acc_bin: 0.9215, F1_bin: 0.9210 | Acc_9: 0.6728, F1_9: 0.6795 | F1_avg_2+9: 0.8003
Epoch  62 | TrainLoss: 0.4092 | Acc18: 0.6420, F1_18: 0.6667 | Acc_bin: 0.9241, F1_bin: 0.9237 | Acc_9: 0.6636, F1_9: 0.6715 | F1_avg_2+9: 0.7976
Epoch  63 | TrainLoss: 0.4254 | Acc18: 0.6597, F1_18: 0.6800 | Acc_bin: 0.9326, F1_bin: 0.9321 | Acc_9: 0.6819, F1_9: 0.6904 | F1_avg_2+9: 0.8112
Epoch  64 | TrainLoss: 0.3995 | Acc18: 0.6446, F1_18: 0.6467 | Acc_bin: 0.9300, F1_bin: 0.9297 | Acc_9: 0.6662, F1_9: 0.6664 | F1_avg_2+9: 0.7980
Epoch  65 | TrainLoss: 0.3904 | Acc18: 0.6381, F1_18: 0.6563 | Acc_bin: 0.9280, F1_bin: 0.9278 | Acc_9: 0.6616, F1_9: 0.6689 | F1_avg_2+9: 0.7983
Epoch  66 | TrainLoss: 0.3606 | Acc18: 0.6545, F1_18: 0.6661 | Acc_bin: 0.9215, F1_bin: 0.9210 | Acc_9: 0.6767, F1_9: 0.6760 | F1_avg_2+9: 0.7985
Epoch  67 | TrainLoss: 0.3726 | Acc18: 0.6505, F1_18: 0.6590 | Acc_bin: 0.9319, F1_bin: 0.9316 | Acc_9: 0.6669, F1_9: 0.6675 | F1_avg_2+9: 0.7996
Epoch  68 | TrainLoss: 0.3815 | Acc18: 0.6603, F1_18: 0.6765 | Acc_bin: 0.9339, F1_bin: 0.9333 | Acc_9: 0.6806, F1_9: 0.6838 | F1_avg_2+9: 0.8086
Epoch  69 | TrainLoss: 0.3879 | Acc18: 0.6427, F1_18: 0.6469 | Acc_bin: 0.9162, F1_bin: 0.9159 | Acc_9: 0.6675, F1_9: 0.6676 | F1_avg_2+9: 0.7917
Epoch  70 | TrainLoss: 0.3732 | Acc18: 0.6545, F1_18: 0.6757 | Acc_bin: 0.9306, F1_bin: 0.9301 | Acc_9: 0.6715, F1_9: 0.6842 | F1_avg_2+9: 0.8071
Epoch  71 | TrainLoss: 0.3703 | Acc18: 0.6459, F1_18: 0.6532 | Acc_bin: 0.9156, F1_bin: 0.9152 | Acc_9: 0.6669, F1_9: 0.6659 | F1_avg_2+9: 0.7906
Epoch  72 | TrainLoss: 0.3868 | Acc18: 0.6577, F1_18: 0.6656 | Acc_bin: 0.9267, F1_bin: 0.9264 | Acc_9: 0.6780, F1_9: 0.6743 | F1_avg_2+9: 0.8004
Epoch  73 | TrainLoss: 0.3864 | Acc18: 0.6512, F1_18: 0.6558 | Acc_bin: 0.9267, F1_bin: 0.9264 | Acc_9: 0.6715, F1_9: 0.6675 | F1_avg_2+9: 0.7969
Epoch  74 | TrainLoss: 0.3551 | Acc18: 0.6473, F1_18: 0.6444 | Acc_bin: 0.9188, F1_bin: 0.9184 | Acc_9: 0.6682, F1_9: 0.6611 | F1_avg_2+9: 0.7897
Epoch  75 | TrainLoss: 0.3495 | Acc18: 0.6459, F1_18: 0.6463 | Acc_bin: 0.9215, F1_bin: 0.9212 | Acc_9: 0.6708, F1_9: 0.6666 | F1_avg_2+9: 0.7939
Epoch  76 | TrainLoss: 0.3773 | Acc18: 0.6610, F1_18: 0.6712 | Acc_bin: 0.9300, F1_bin: 0.9295 | Acc_9: 0.6819, F1_9: 0.6843 | F1_avg_2+9: 0.8069
Epoch  77 | TrainLoss: 0.3466 | Acc18: 0.6531, F1_18: 0.6637 | Acc_bin: 0.9280, F1_bin: 0.9276 | Acc_9: 0.6793, F1_9: 0.6774 | F1_avg_2+9: 0.8025
Epoch  78 | TrainLoss: 0.3469 | Acc18: 0.6616, F1_18: 0.6807 | Acc_bin: 0.9326, F1_bin: 0.9323 | Acc_9: 0.6839, F1_9: 0.6892 | F1_avg_2+9: 0.8107
Epoch  79 | TrainLoss: 0.3256 | Acc18: 0.6440, F1_18: 0.6522 | Acc_bin: 0.9254, F1_bin: 0.9251 | Acc_9: 0.6675, F1_9: 0.6635 | F1_avg_2+9: 0.7943
Early stopping
=== Fold 2 ===
Epoch   1 | TrainLoss: 2.2673 | Acc18: 0.3958, F1_18: 0.3495 | Acc_bin: 0.7782, F1_bin: 0.7768 | Acc_9: 0.4344, F1_9: 0.3887 | F1_avg_2+9: 0.5827
Epoch   2 | TrainLoss: 1.6893 | Acc18: 0.4638, F1_18: 0.4347 | Acc_bin: 0.8248, F1_bin: 0.8246 | Acc_9: 0.5006, F1_9: 0.4774 | F1_avg_2+9: 0.6510
Epoch   3 | TrainLoss: 1.4295 | Acc18: 0.5086, F1_18: 0.5199 | Acc_bin: 0.8701, F1_bin: 0.8700 | Acc_9: 0.5362, F1_9: 0.5385 | F1_avg_2+9: 0.7043
Epoch   4 | TrainLoss: 1.2860 | Acc18: 0.5447, F1_18: 0.5371 | Acc_bin: 0.8695, F1_bin: 0.8695 | Acc_9: 0.5772, F1_9: 0.5728 | F1_avg_2+9: 0.7211
Epoch   5 | TrainLoss: 1.1928 | Acc18: 0.5521, F1_18: 0.5545 | Acc_bin: 0.8860, F1_bin: 0.8859 | Acc_9: 0.5729, F1_9: 0.5760 | F1_avg_2+9: 0.7309
Epoch   6 | TrainLoss: 1.1133 | Acc18: 0.5588, F1_18: 0.5639 | Acc_bin: 0.8817, F1_bin: 0.8816 | Acc_9: 0.5809, F1_9: 0.5866 | F1_avg_2+9: 0.7341
Epoch   7 | TrainLoss: 1.0721 | Acc18: 0.5619, F1_18: 0.5642 | Acc_bin: 0.8732, F1_bin: 0.8731 | Acc_9: 0.5968, F1_9: 0.5968 | F1_avg_2+9: 0.7350
Epoch   8 | TrainLoss: 1.0237 | Acc18: 0.5558, F1_18: 0.5551 | Acc_bin: 0.8873, F1_bin: 0.8872 | Acc_9: 0.5790, F1_9: 0.5857 | F1_avg_2+9: 0.7364
Epoch   9 | TrainLoss: 0.9657 | Acc18: 0.5631, F1_18: 0.5588 | Acc_bin: 0.9026, F1_bin: 0.9025 | Acc_9: 0.5846, F1_9: 0.5849 | F1_avg_2+9: 0.7437
Epoch  10 | TrainLoss: 0.9268 | Acc18: 0.5656, F1_18: 0.5692 | Acc_bin: 0.8958, F1_bin: 0.8957 | Acc_9: 0.5895, F1_9: 0.5939 | F1_avg_2+9: 0.7448
Epoch  11 | TrainLoss: 0.8958 | Acc18: 0.5760, F1_18: 0.5932 | Acc_bin: 0.9093, F1_bin: 0.9089 | Acc_9: 0.6029, F1_9: 0.6149 | F1_avg_2+9: 0.7619
Epoch  12 | TrainLoss: 0.8576 | Acc18: 0.5766, F1_18: 0.5829 | Acc_bin: 0.8940, F1_bin: 0.8940 | Acc_9: 0.6060, F1_9: 0.6061 | F1_avg_2+9: 0.7500
Epoch  13 | TrainLoss: 0.8484 | Acc18: 0.5699, F1_18: 0.5701 | Acc_bin: 0.8860, F1_bin: 0.8851 | Acc_9: 0.5950, F1_9: 0.5994 | F1_avg_2+9: 0.7422
Epoch  14 | TrainLoss: 0.8115 | Acc18: 0.5809, F1_18: 0.5732 | Acc_bin: 0.9062, F1_bin: 0.9062 | Acc_9: 0.5999, F1_9: 0.5980 | F1_avg_2+9: 0.7521
Epoch  15 | TrainLoss: 0.7981 | Acc18: 0.5564, F1_18: 0.5615 | Acc_bin: 0.8983, F1_bin: 0.8978 | Acc_9: 0.5839, F1_9: 0.5776 | F1_avg_2+9: 0.7377
Epoch  16 | TrainLoss: 0.8079 | Acc18: 0.5846, F1_18: 0.5829 | Acc_bin: 0.8971, F1_bin: 0.8971 | Acc_9: 0.6109, F1_9: 0.6111 | F1_avg_2+9: 0.7541
Epoch  17 | TrainLoss: 0.7667 | Acc18: 0.5907, F1_18: 0.6000 | Acc_bin: 0.9087, F1_bin: 0.9086 | Acc_9: 0.6146, F1_9: 0.6167 | F1_avg_2+9: 0.7626
Epoch  18 | TrainLoss: 0.7329 | Acc18: 0.5699, F1_18: 0.5652 | Acc_bin: 0.9081, F1_bin: 0.9079 | Acc_9: 0.5925, F1_9: 0.5801 | F1_avg_2+9: 0.7440
Epoch  19 | TrainLoss: 0.7368 | Acc18: 0.5888, F1_18: 0.5822 | Acc_bin: 0.9118, F1_bin: 0.9113 | Acc_9: 0.6115, F1_9: 0.6114 | F1_avg_2+9: 0.7614
Epoch  20 | TrainLoss: 0.7166 | Acc18: 0.5864, F1_18: 0.5961 | Acc_bin: 0.9173, F1_bin: 0.9169 | Acc_9: 0.6036, F1_9: 0.6096 | F1_avg_2+9: 0.7632
Epoch  21 | TrainLoss: 0.6891 | Acc18: 0.6011, F1_18: 0.5983 | Acc_bin: 0.9112, F1_bin: 0.9110 | Acc_9: 0.6213, F1_9: 0.6191 | F1_avg_2+9: 0.7650
Epoch  22 | TrainLoss: 0.6577 | Acc18: 0.6078, F1_18: 0.6045 | Acc_bin: 0.9056, F1_bin: 0.9049 | Acc_9: 0.6324, F1_9: 0.6317 | F1_avg_2+9: 0.7683
Epoch  23 | TrainLoss: 0.6700 | Acc18: 0.5821, F1_18: 0.5926 | Acc_bin: 0.8934, F1_bin: 0.8933 | Acc_9: 0.6176, F1_9: 0.6213 | F1_avg_2+9: 0.7573
Epoch  24 | TrainLoss: 0.6426 | Acc18: 0.5748, F1_18: 0.5858 | Acc_bin: 0.9044, F1_bin: 0.9043 | Acc_9: 0.5999, F1_9: 0.6053 | F1_avg_2+9: 0.7548
Epoch  25 | TrainLoss: 0.6392 | Acc18: 0.5974, F1_18: 0.6031 | Acc_bin: 0.9032, F1_bin: 0.9029 | Acc_9: 0.6213, F1_9: 0.6278 | F1_avg_2+9: 0.7654
Epoch  26 | TrainLoss: 0.6200 | Acc18: 0.5827, F1_18: 0.5945 | Acc_bin: 0.9087, F1_bin: 0.9083 | Acc_9: 0.6091, F1_9: 0.6027 | F1_avg_2+9: 0.7555
Epoch  27 | TrainLoss: 0.6286 | Acc18: 0.5717, F1_18: 0.5786 | Acc_bin: 0.8915, F1_bin: 0.8915 | Acc_9: 0.5980, F1_9: 0.5947 | F1_avg_2+9: 0.7431
Epoch  28 | TrainLoss: 0.6146 | Acc18: 0.6060, F1_18: 0.6095 | Acc_bin: 0.9142, F1_bin: 0.9140 | Acc_9: 0.6287, F1_9: 0.6224 | F1_avg_2+9: 0.7682
Epoch  29 | TrainLoss: 0.6087 | Acc18: 0.5607, F1_18: 0.5737 | Acc_bin: 0.9081, F1_bin: 0.9078 | Acc_9: 0.5790, F1_9: 0.5808 | F1_avg_2+9: 0.7443
Epoch  30 | TrainLoss: 0.5840 | Acc18: 0.5864, F1_18: 0.5957 | Acc_bin: 0.8958, F1_bin: 0.8958 | Acc_9: 0.6127, F1_9: 0.6153 | F1_avg_2+9: 0.7556
Epoch  31 | TrainLoss: 0.5679 | Acc18: 0.5398, F1_18: 0.5545 | Acc_bin: 0.8977, F1_bin: 0.8975 | Acc_9: 0.5637, F1_9: 0.5803 | F1_avg_2+9: 0.7389
Epoch  32 | TrainLoss: 0.5647 | Acc18: 0.5827, F1_18: 0.5911 | Acc_bin: 0.9007, F1_bin: 0.9007 | Acc_9: 0.6085, F1_9: 0.6003 | F1_avg_2+9: 0.7505
Epoch  33 | TrainLoss: 0.5788 | Acc18: 0.5735, F1_18: 0.5781 | Acc_bin: 0.8928, F1_bin: 0.8925 | Acc_9: 0.6036, F1_9: 0.6068 | F1_avg_2+9: 0.7497
Epoch  34 | TrainLoss: 0.5608 | Acc18: 0.5974, F1_18: 0.6049 | Acc_bin: 0.9069, F1_bin: 0.9064 | Acc_9: 0.6140, F1_9: 0.6228 | F1_avg_2+9: 0.7646
Epoch  35 | TrainLoss: 0.5346 | Acc18: 0.6078, F1_18: 0.6070 | Acc_bin: 0.9118, F1_bin: 0.9110 | Acc_9: 0.6305, F1_9: 0.6329 | F1_avg_2+9: 0.7720
Epoch  36 | TrainLoss: 0.5327 | Acc18: 0.5741, F1_18: 0.5817 | Acc_bin: 0.9099, F1_bin: 0.9097 | Acc_9: 0.6017, F1_9: 0.6092 | F1_avg_2+9: 0.7595
Epoch  37 | TrainLoss: 0.5297 | Acc18: 0.6066, F1_18: 0.6082 | Acc_bin: 0.9148, F1_bin: 0.9145 | Acc_9: 0.6256, F1_9: 0.6262 | F1_avg_2+9: 0.7704
Epoch  38 | TrainLoss: 0.5112 | Acc18: 0.6072, F1_18: 0.6170 | Acc_bin: 0.9007, F1_bin: 0.9005 | Acc_9: 0.6299, F1_9: 0.6344 | F1_avg_2+9: 0.7675
Epoch  39 | TrainLoss: 0.5067 | Acc18: 0.5809, F1_18: 0.5992 | Acc_bin: 0.8830, F1_bin: 0.8829 | Acc_9: 0.6103, F1_9: 0.6131 | F1_avg_2+9: 0.7480
Epoch  40 | TrainLoss: 0.5074 | Acc18: 0.5968, F1_18: 0.5953 | Acc_bin: 0.8995, F1_bin: 0.8993 | Acc_9: 0.6207, F1_9: 0.6180 | F1_avg_2+9: 0.7586
Epoch  41 | TrainLoss: 0.4955 | Acc18: 0.5386, F1_18: 0.5306 | Acc_bin: 0.9020, F1_bin: 0.9017 | Acc_9: 0.5619, F1_9: 0.5476 | F1_avg_2+9: 0.7247
Epoch  42 | TrainLoss: 0.4855 | Acc18: 0.5882, F1_18: 0.5910 | Acc_bin: 0.8866, F1_bin: 0.8857 | Acc_9: 0.6103, F1_9: 0.6140 | F1_avg_2+9: 0.7498
Epoch  43 | TrainLoss: 0.4794 | Acc18: 0.5827, F1_18: 0.5880 | Acc_bin: 0.8964, F1_bin: 0.8956 | Acc_9: 0.6048, F1_9: 0.6099 | F1_avg_2+9: 0.7528
Epoch  44 | TrainLoss: 0.5086 | Acc18: 0.6085, F1_18: 0.6172 | Acc_bin: 0.9001, F1_bin: 0.8995 | Acc_9: 0.6299, F1_9: 0.6322 | F1_avg_2+9: 0.7658
Epoch  45 | TrainLoss: 0.4627 | Acc18: 0.6091, F1_18: 0.6064 | Acc_bin: 0.9013, F1_bin: 0.9007 | Acc_9: 0.6299, F1_9: 0.6278 | F1_avg_2+9: 0.7642
Epoch  46 | TrainLoss: 0.4871 | Acc18: 0.5956, F1_18: 0.6046 | Acc_bin: 0.9161, F1_bin: 0.9153 | Acc_9: 0.6176, F1_9: 0.6278 | F1_avg_2+9: 0.7716
Epoch  47 | TrainLoss: 0.4617 | Acc18: 0.5968, F1_18: 0.6056 | Acc_bin: 0.9167, F1_bin: 0.9164 | Acc_9: 0.6134, F1_9: 0.6159 | F1_avg_2+9: 0.7662
Epoch  48 | TrainLoss: 0.4532 | Acc18: 0.5913, F1_18: 0.6068 | Acc_bin: 0.8977, F1_bin: 0.8974 | Acc_9: 0.6152, F1_9: 0.6238 | F1_avg_2+9: 0.7606
Epoch  49 | TrainLoss: 0.4411 | Acc18: 0.6085, F1_18: 0.6026 | Acc_bin: 0.8977, F1_bin: 0.8970 | Acc_9: 0.6293, F1_9: 0.6218 | F1_avg_2+9: 0.7594
Epoch  50 | TrainLoss: 0.4204 | Acc18: 0.5999, F1_18: 0.6067 | Acc_bin: 0.9081, F1_bin: 0.9074 | Acc_9: 0.6213, F1_9: 0.6248 | F1_avg_2+9: 0.7661
Epoch  51 | TrainLoss: 0.4494 | Acc18: 0.6042, F1_18: 0.6157 | Acc_bin: 0.8897, F1_bin: 0.8896 | Acc_9: 0.6311, F1_9: 0.6341 | F1_avg_2+9: 0.7618
Epoch  52 | TrainLoss: 0.4242 | Acc18: 0.6036, F1_18: 0.6023 | Acc_bin: 0.9056, F1_bin: 0.9052 | Acc_9: 0.6195, F1_9: 0.6152 | F1_avg_2+9: 0.7602
Epoch  53 | TrainLoss: 0.4337 | Acc18: 0.6103, F1_18: 0.6240 | Acc_bin: 0.9112, F1_bin: 0.9111 | Acc_9: 0.6293, F1_9: 0.6383 | F1_avg_2+9: 0.7747
Epoch  54 | TrainLoss: 0.4332 | Acc18: 0.6078, F1_18: 0.6235 | Acc_bin: 0.9007, F1_bin: 0.9007 | Acc_9: 0.6287, F1_9: 0.6342 | F1_avg_2+9: 0.7674
Epoch  55 | TrainLoss: 0.4038 | Acc18: 0.6036, F1_18: 0.6156 | Acc_bin: 0.9130, F1_bin: 0.9128 | Acc_9: 0.6232, F1_9: 0.6301 | F1_avg_2+9: 0.7714
Epoch  56 | TrainLoss: 0.4332 | Acc18: 0.5999, F1_18: 0.6126 | Acc_bin: 0.9069, F1_bin: 0.9066 | Acc_9: 0.6213, F1_9: 0.6337 | F1_avg_2+9: 0.7702
Epoch  57 | TrainLoss: 0.4016 | Acc18: 0.6121, F1_18: 0.6115 | Acc_bin: 0.9136, F1_bin: 0.9132 | Acc_9: 0.6299, F1_9: 0.6298 | F1_avg_2+9: 0.7715
Epoch  58 | TrainLoss: 0.4216 | Acc18: 0.5956, F1_18: 0.6163 | Acc_bin: 0.9179, F1_bin: 0.9175 | Acc_9: 0.6134, F1_9: 0.6270 | F1_avg_2+9: 0.7723
Epoch  59 | TrainLoss: 0.4114 | Acc18: 0.6066, F1_18: 0.6239 | Acc_bin: 0.9093, F1_bin: 0.9090 | Acc_9: 0.6238, F1_9: 0.6307 | F1_avg_2+9: 0.7698
Epoch  60 | TrainLoss: 0.4090 | Acc18: 0.5956, F1_18: 0.6127 | Acc_bin: 0.9081, F1_bin: 0.9076 | Acc_9: 0.6115, F1_9: 0.6261 | F1_avg_2+9: 0.7668
Epoch  61 | TrainLoss: 0.4084 | Acc18: 0.6232, F1_18: 0.6339 | Acc_bin: 0.9210, F1_bin: 0.9207 | Acc_9: 0.6391, F1_9: 0.6431 | F1_avg_2+9: 0.7819
Epoch  62 | TrainLoss: 0.3751 | Acc18: 0.6232, F1_18: 0.6446 | Acc_bin: 0.9032, F1_bin: 0.9023 | Acc_9: 0.6452, F1_9: 0.6603 | F1_avg_2+9: 0.7813
Epoch  63 | TrainLoss: 0.4188 | Acc18: 0.6017, F1_18: 0.6103 | Acc_bin: 0.9167, F1_bin: 0.9165 | Acc_9: 0.6195, F1_9: 0.6289 | F1_avg_2+9: 0.7727
Epoch  64 | TrainLoss: 0.3793 | Acc18: 0.6023, F1_18: 0.6163 | Acc_bin: 0.8983, F1_bin: 0.8983 | Acc_9: 0.6250, F1_9: 0.6309 | F1_avg_2+9: 0.7646
Epoch  65 | TrainLoss: 0.3770 | Acc18: 0.6158, F1_18: 0.6325 | Acc_bin: 0.9075, F1_bin: 0.9072 | Acc_9: 0.6336, F1_9: 0.6469 | F1_avg_2+9: 0.7770
Epoch  66 | TrainLoss: 0.3677 | Acc18: 0.6011, F1_18: 0.6112 | Acc_bin: 0.9191, F1_bin: 0.9189 | Acc_9: 0.6164, F1_9: 0.6176 | F1_avg_2+9: 0.7683
Epoch  67 | TrainLoss: 0.3769 | Acc18: 0.5938, F1_18: 0.6171 | Acc_bin: 0.9093, F1_bin: 0.9093 | Acc_9: 0.6146, F1_9: 0.6225 | F1_avg_2+9: 0.7659
Epoch  68 | TrainLoss: 0.3554 | Acc18: 0.6170, F1_18: 0.6237 | Acc_bin: 0.9167, F1_bin: 0.9166 | Acc_9: 0.6379, F1_9: 0.6355 | F1_avg_2+9: 0.7761
Epoch  69 | TrainLoss: 0.3380 | Acc18: 0.5938, F1_18: 0.6106 | Acc_bin: 0.9124, F1_bin: 0.9119 | Acc_9: 0.6121, F1_9: 0.6212 | F1_avg_2+9: 0.7666
Epoch  70 | TrainLoss: 0.3444 | Acc18: 0.5833, F1_18: 0.5999 | Acc_bin: 0.9069, F1_bin: 0.9065 | Acc_9: 0.6054, F1_9: 0.6082 | F1_avg_2+9: 0.7574
Epoch  71 | TrainLoss: 0.3566 | Acc18: 0.6170, F1_18: 0.6095 | Acc_bin: 0.9118, F1_bin: 0.9117 | Acc_9: 0.6422, F1_9: 0.6400 | F1_avg_2+9: 0.7758
Epoch  72 | TrainLoss: 0.3697 | Acc18: 0.6146, F1_18: 0.6405 | Acc_bin: 0.9161, F1_bin: 0.9155 | Acc_9: 0.6317, F1_9: 0.6434 | F1_avg_2+9: 0.7795
Epoch  73 | TrainLoss: 0.3601 | Acc18: 0.6097, F1_18: 0.6320 | Acc_bin: 0.9142, F1_bin: 0.9140 | Acc_9: 0.6311, F1_9: 0.6417 | F1_avg_2+9: 0.7778
Epoch  74 | TrainLoss: 0.3737 | Acc18: 0.5925, F1_18: 0.6008 | Acc_bin: 0.9050, F1_bin: 0.9046 | Acc_9: 0.6103, F1_9: 0.6130 | F1_avg_2+9: 0.7588
Epoch  75 | TrainLoss: 0.3383 | Acc18: 0.5870, F1_18: 0.6051 | Acc_bin: 0.9056, F1_bin: 0.9051 | Acc_9: 0.6060, F1_9: 0.6113 | F1_avg_2+9: 0.7582
Epoch  76 | TrainLoss: 0.3455 | Acc18: 0.6127, F1_18: 0.6227 | Acc_bin: 0.9087, F1_bin: 0.9083 | Acc_9: 0.6317, F1_9: 0.6355 | F1_avg_2+9: 0.7719
Epoch  77 | TrainLoss: 0.3184 | Acc18: 0.6213, F1_18: 0.6368 | Acc_bin: 0.9093, F1_bin: 0.9091 | Acc_9: 0.6422, F1_9: 0.6493 | F1_avg_2+9: 0.7792
Epoch  78 | TrainLoss: 0.3277 | Acc18: 0.6066, F1_18: 0.6308 | Acc_bin: 0.9118, F1_bin: 0.9115 | Acc_9: 0.6268, F1_9: 0.6417 | F1_avg_2+9: 0.7766
Epoch  79 | TrainLoss: 0.3505 | Acc18: 0.6036, F1_18: 0.6253 | Acc_bin: 0.8891, F1_bin: 0.8890 | Acc_9: 0.6238, F1_9: 0.6379 | F1_avg_2+9: 0.7635
Epoch  80 | TrainLoss: 0.3119 | Acc18: 0.6072, F1_18: 0.6282 | Acc_bin: 0.9130, F1_bin: 0.9124 | Acc_9: 0.6299, F1_9: 0.6449 | F1_avg_2+9: 0.7787
Epoch  81 | TrainLoss: 0.3349 | Acc18: 0.6127, F1_18: 0.6294 | Acc_bin: 0.9136, F1_bin: 0.9131 | Acc_9: 0.6342, F1_9: 0.6404 | F1_avg_2+9: 0.7768
Epoch  82 | TrainLoss: 0.3194 | Acc18: 0.6048, F1_18: 0.6202 | Acc_bin: 0.8971, F1_bin: 0.8957 | Acc_9: 0.6281, F1_9: 0.6309 | F1_avg_2+9: 0.7633
Early stopping
=== Fold 3 ===
Epoch   1 | TrainLoss: 2.2857 | Acc18: 0.4302, F1_18: 0.3929 | Acc_bin: 0.8203, F1_bin: 0.8203 | Acc_9: 0.4658, F1_9: 0.4279 | F1_avg_2+9: 0.6241
Epoch   2 | TrainLoss: 1.7059 | Acc18: 0.4812, F1_18: 0.4835 | Acc_bin: 0.8468, F1_bin: 0.8467 | Acc_9: 0.5126, F1_9: 0.4935 | F1_avg_2+9: 0.6701
Epoch   3 | TrainLoss: 1.4622 | Acc18: 0.5286, F1_18: 0.5140 | Acc_bin: 0.8585, F1_bin: 0.8584 | Acc_9: 0.5606, F1_9: 0.5530 | F1_avg_2+9: 0.7057
Epoch   4 | TrainLoss: 1.3186 | Acc18: 0.5563, F1_18: 0.5519 | Acc_bin: 0.8732, F1_bin: 0.8730 | Acc_9: 0.5834, F1_9: 0.5798 | F1_avg_2+9: 0.7264
Epoch   5 | TrainLoss: 1.2096 | Acc18: 0.5625, F1_18: 0.5583 | Acc_bin: 0.8769, F1_bin: 0.8769 | Acc_9: 0.5883, F1_9: 0.5821 | F1_avg_2+9: 0.7295
Epoch   6 | TrainLoss: 1.1207 | Acc18: 0.5612, F1_18: 0.5442 | Acc_bin: 0.8782, F1_bin: 0.8780 | Acc_9: 0.5871, F1_9: 0.5721 | F1_avg_2+9: 0.7250
Epoch   7 | TrainLoss: 1.0684 | Acc18: 0.5440, F1_18: 0.5436 | Acc_bin: 0.8646, F1_bin: 0.8645 | Acc_9: 0.5754, F1_9: 0.5769 | F1_avg_2+9: 0.7207
Epoch   8 | TrainLoss: 1.0197 | Acc18: 0.5391, F1_18: 0.5354 | Acc_bin: 0.8683, F1_bin: 0.8683 | Acc_9: 0.5668, F1_9: 0.5652 | F1_avg_2+9: 0.7167
Epoch   9 | TrainLoss: 0.9691 | Acc18: 0.5729, F1_18: 0.5831 | Acc_bin: 0.8929, F1_bin: 0.8926 | Acc_9: 0.5902, F1_9: 0.6017 | F1_avg_2+9: 0.7472
Epoch  10 | TrainLoss: 0.9286 | Acc18: 0.5908, F1_18: 0.5720 | Acc_bin: 0.8917, F1_bin: 0.8912 | Acc_9: 0.6178, F1_9: 0.6173 | F1_avg_2+9: 0.7543
Epoch  11 | TrainLoss: 0.8976 | Acc18: 0.5969, F1_18: 0.6020 | Acc_bin: 0.8775, F1_bin: 0.8770 | Acc_9: 0.6178, F1_9: 0.6247 | F1_avg_2+9: 0.7509
Epoch  12 | TrainLoss: 0.8814 | Acc18: 0.5945, F1_18: 0.5834 | Acc_bin: 0.8898, F1_bin: 0.8897 | Acc_9: 0.6160, F1_9: 0.6126 | F1_avg_2+9: 0.7512
Epoch  13 | TrainLoss: 0.8239 | Acc18: 0.5772, F1_18: 0.5698 | Acc_bin: 0.8862, F1_bin: 0.8860 | Acc_9: 0.6012, F1_9: 0.6041 | F1_avg_2+9: 0.7450
Epoch  14 | TrainLoss: 0.8154 | Acc18: 0.5680, F1_18: 0.5738 | Acc_bin: 0.8892, F1_bin: 0.8892 | Acc_9: 0.5889, F1_9: 0.5943 | F1_avg_2+9: 0.7418
Epoch  15 | TrainLoss: 0.8100 | Acc18: 0.5686, F1_18: 0.5632 | Acc_bin: 0.8825, F1_bin: 0.8825 | Acc_9: 0.5926, F1_9: 0.5970 | F1_avg_2+9: 0.7397
Epoch  16 | TrainLoss: 0.7755 | Acc18: 0.5908, F1_18: 0.5802 | Acc_bin: 0.8874, F1_bin: 0.8873 | Acc_9: 0.6166, F1_9: 0.6148 | F1_avg_2+9: 0.7511
Epoch  17 | TrainLoss: 0.7763 | Acc18: 0.5871, F1_18: 0.5773 | Acc_bin: 0.8985, F1_bin: 0.8983 | Acc_9: 0.6092, F1_9: 0.5970 | F1_avg_2+9: 0.7476
Epoch  18 | TrainLoss: 0.7395 | Acc18: 0.5975, F1_18: 0.5903 | Acc_bin: 0.8794, F1_bin: 0.8793 | Acc_9: 0.6191, F1_9: 0.6268 | F1_avg_2+9: 0.7531
Epoch  19 | TrainLoss: 0.7181 | Acc18: 0.5963, F1_18: 0.5884 | Acc_bin: 0.8788, F1_bin: 0.8783 | Acc_9: 0.6228, F1_9: 0.6267 | F1_avg_2+9: 0.7525
Epoch  20 | TrainLoss: 0.7077 | Acc18: 0.5858, F1_18: 0.5772 | Acc_bin: 0.8874, F1_bin: 0.8872 | Acc_9: 0.6055, F1_9: 0.6001 | F1_avg_2+9: 0.7437
Epoch  21 | TrainLoss: 0.7144 | Acc18: 0.5754, F1_18: 0.5908 | Acc_bin: 0.8745, F1_bin: 0.8742 | Acc_9: 0.6006, F1_9: 0.6093 | F1_avg_2+9: 0.7417
Epoch  22 | TrainLoss: 0.6793 | Acc18: 0.5563, F1_18: 0.5569 | Acc_bin: 0.8868, F1_bin: 0.8867 | Acc_9: 0.5772, F1_9: 0.5887 | F1_avg_2+9: 0.7377
Epoch  23 | TrainLoss: 0.6781 | Acc18: 0.5865, F1_18: 0.5922 | Acc_bin: 0.8818, F1_bin: 0.8812 | Acc_9: 0.6129, F1_9: 0.6194 | F1_avg_2+9: 0.7503
Epoch  24 | TrainLoss: 0.6315 | Acc18: 0.5951, F1_18: 0.5828 | Acc_bin: 0.8831, F1_bin: 0.8829 | Acc_9: 0.6240, F1_9: 0.6138 | F1_avg_2+9: 0.7484
Epoch  25 | TrainLoss: 0.6263 | Acc18: 0.5822, F1_18: 0.5843 | Acc_bin: 0.8794, F1_bin: 0.8793 | Acc_9: 0.6031, F1_9: 0.6057 | F1_avg_2+9: 0.7425
Epoch  26 | TrainLoss: 0.6350 | Acc18: 0.5994, F1_18: 0.5949 | Acc_bin: 0.8763, F1_bin: 0.8762 | Acc_9: 0.6289, F1_9: 0.6239 | F1_avg_2+9: 0.7501
Epoch  27 | TrainLoss: 0.6211 | Acc18: 0.5963, F1_18: 0.5981 | Acc_bin: 0.8671, F1_bin: 0.8670 | Acc_9: 0.6265, F1_9: 0.6257 | F1_avg_2+9: 0.7463
Epoch  28 | TrainLoss: 0.6038 | Acc18: 0.5969, F1_18: 0.5890 | Acc_bin: 0.8788, F1_bin: 0.8786 | Acc_9: 0.6240, F1_9: 0.6192 | F1_avg_2+9: 0.7489
Epoch  29 | TrainLoss: 0.5942 | Acc18: 0.5791, F1_18: 0.5693 | Acc_bin: 0.8640, F1_bin: 0.8640 | Acc_9: 0.6049, F1_9: 0.5966 | F1_avg_2+9: 0.7303
Epoch  30 | TrainLoss: 0.5702 | Acc18: 0.5871, F1_18: 0.5910 | Acc_bin: 0.8683, F1_bin: 0.8682 | Acc_9: 0.6160, F1_9: 0.6193 | F1_avg_2+9: 0.7437
Epoch  31 | TrainLoss: 0.5530 | Acc18: 0.6043, F1_18: 0.6043 | Acc_bin: 0.8800, F1_bin: 0.8799 | Acc_9: 0.6332, F1_9: 0.6314 | F1_avg_2+9: 0.7556
Epoch  32 | TrainLoss: 0.5522 | Acc18: 0.5908, F1_18: 0.5912 | Acc_bin: 0.8658, F1_bin: 0.8658 | Acc_9: 0.6178, F1_9: 0.6156 | F1_avg_2+9: 0.7407
Epoch  33 | TrainLoss: 0.5384 | Acc18: 0.5883, F1_18: 0.5997 | Acc_bin: 0.8874, F1_bin: 0.8873 | Acc_9: 0.6068, F1_9: 0.6166 | F1_avg_2+9: 0.7519
Epoch  34 | TrainLoss: 0.5409 | Acc18: 0.5858, F1_18: 0.5884 | Acc_bin: 0.8585, F1_bin: 0.8584 | Acc_9: 0.6142, F1_9: 0.6196 | F1_avg_2+9: 0.7390
Epoch  35 | TrainLoss: 0.5354 | Acc18: 0.5994, F1_18: 0.5974 | Acc_bin: 0.8849, F1_bin: 0.8846 | Acc_9: 0.6289, F1_9: 0.6288 | F1_avg_2+9: 0.7567
Epoch  36 | TrainLoss: 0.5293 | Acc18: 0.6111, F1_18: 0.6091 | Acc_bin: 0.9015, F1_bin: 0.9014 | Acc_9: 0.6326, F1_9: 0.6302 | F1_avg_2+9: 0.7658
Epoch  37 | TrainLoss: 0.5188 | Acc18: 0.5914, F1_18: 0.5987 | Acc_bin: 0.8825, F1_bin: 0.8824 | Acc_9: 0.6086, F1_9: 0.6206 | F1_avg_2+9: 0.7515
Epoch  38 | TrainLoss: 0.4893 | Acc18: 0.6055, F1_18: 0.6003 | Acc_bin: 0.8855, F1_bin: 0.8854 | Acc_9: 0.6277, F1_9: 0.6291 | F1_avg_2+9: 0.7572
Epoch  39 | TrainLoss: 0.5032 | Acc18: 0.5994, F1_18: 0.5987 | Acc_bin: 0.8849, F1_bin: 0.8847 | Acc_9: 0.6154, F1_9: 0.6230 | F1_avg_2+9: 0.7539
Epoch  40 | TrainLoss: 0.4883 | Acc18: 0.5994, F1_18: 0.6082 | Acc_bin: 0.8929, F1_bin: 0.8928 | Acc_9: 0.6197, F1_9: 0.6280 | F1_avg_2+9: 0.7604
Epoch  41 | TrainLoss: 0.4917 | Acc18: 0.6271, F1_18: 0.6227 | Acc_bin: 0.9003, F1_bin: 0.9003 | Acc_9: 0.6443, F1_9: 0.6478 | F1_avg_2+9: 0.7741
Epoch  42 | TrainLoss: 0.4755 | Acc18: 0.6000, F1_18: 0.5934 | Acc_bin: 0.8855, F1_bin: 0.8855 | Acc_9: 0.6228, F1_9: 0.6142 | F1_avg_2+9: 0.7498
Epoch  43 | TrainLoss: 0.4473 | Acc18: 0.5988, F1_18: 0.6073 | Acc_bin: 0.8825, F1_bin: 0.8825 | Acc_9: 0.6209, F1_9: 0.6281 | F1_avg_2+9: 0.7553
Epoch  44 | TrainLoss: 0.4671 | Acc18: 0.5938, F1_18: 0.5999 | Acc_bin: 0.8849, F1_bin: 0.8847 | Acc_9: 0.6166, F1_9: 0.6194 | F1_avg_2+9: 0.7521
Epoch  45 | TrainLoss: 0.4549 | Acc18: 0.6092, F1_18: 0.6231 | Acc_bin: 0.8769, F1_bin: 0.8769 | Acc_9: 0.6302, F1_9: 0.6428 | F1_avg_2+9: 0.7599
Epoch  46 | TrainLoss: 0.4664 | Acc18: 0.5766, F1_18: 0.5748 | Acc_bin: 0.8720, F1_bin: 0.8718 | Acc_9: 0.6092, F1_9: 0.6097 | F1_avg_2+9: 0.7407
Epoch  47 | TrainLoss: 0.4513 | Acc18: 0.6043, F1_18: 0.6040 | Acc_bin: 0.8862, F1_bin: 0.8858 | Acc_9: 0.6375, F1_9: 0.6347 | F1_avg_2+9: 0.7602
Epoch  48 | TrainLoss: 0.4375 | Acc18: 0.6037, F1_18: 0.6113 | Acc_bin: 0.8923, F1_bin: 0.8922 | Acc_9: 0.6234, F1_9: 0.6343 | F1_avg_2+9: 0.7632
Epoch  49 | TrainLoss: 0.4347 | Acc18: 0.6111, F1_18: 0.6203 | Acc_bin: 0.8935, F1_bin: 0.8934 | Acc_9: 0.6308, F1_9: 0.6338 | F1_avg_2+9: 0.7636
Epoch  50 | TrainLoss: 0.4161 | Acc18: 0.5809, F1_18: 0.5834 | Acc_bin: 0.8806, F1_bin: 0.8805 | Acc_9: 0.6105, F1_9: 0.6052 | F1_avg_2+9: 0.7428
Epoch  51 | TrainLoss: 0.4246 | Acc18: 0.5988, F1_18: 0.5944 | Acc_bin: 0.8782, F1_bin: 0.8781 | Acc_9: 0.6228, F1_9: 0.6233 | F1_avg_2+9: 0.7507
Epoch  52 | TrainLoss: 0.4561 | Acc18: 0.6160, F1_18: 0.6141 | Acc_bin: 0.8831, F1_bin: 0.8831 | Acc_9: 0.6363, F1_9: 0.6305 | F1_avg_2+9: 0.7568
Epoch  53 | TrainLoss: 0.4161 | Acc18: 0.6135, F1_18: 0.6182 | Acc_bin: 0.8935, F1_bin: 0.8934 | Acc_9: 0.6326, F1_9: 0.6287 | F1_avg_2+9: 0.7610
Epoch  54 | TrainLoss: 0.4283 | Acc18: 0.5988, F1_18: 0.6013 | Acc_bin: 0.8880, F1_bin: 0.8879 | Acc_9: 0.6142, F1_9: 0.6200 | F1_avg_2+9: 0.7540
Epoch  55 | TrainLoss: 0.3883 | Acc18: 0.6068, F1_18: 0.6083 | Acc_bin: 0.8886, F1_bin: 0.8882 | Acc_9: 0.6302, F1_9: 0.6347 | F1_avg_2+9: 0.7614
Epoch  56 | TrainLoss: 0.3818 | Acc18: 0.6012, F1_18: 0.6010 | Acc_bin: 0.8837, F1_bin: 0.8835 | Acc_9: 0.6246, F1_9: 0.6249 | F1_avg_2+9: 0.7542
Epoch  57 | TrainLoss: 0.4068 | Acc18: 0.6080, F1_18: 0.6177 | Acc_bin: 0.8960, F1_bin: 0.8958 | Acc_9: 0.6295, F1_9: 0.6348 | F1_avg_2+9: 0.7653
Epoch  58 | TrainLoss: 0.4111 | Acc18: 0.6135, F1_18: 0.6149 | Acc_bin: 0.8800, F1_bin: 0.8798 | Acc_9: 0.6320, F1_9: 0.6280 | F1_avg_2+9: 0.7539
Epoch  59 | TrainLoss: 0.3966 | Acc18: 0.6086, F1_18: 0.6001 | Acc_bin: 0.8837, F1_bin: 0.8834 | Acc_9: 0.6314, F1_9: 0.6220 | F1_avg_2+9: 0.7527
Epoch  60 | TrainLoss: 0.3752 | Acc18: 0.6209, F1_18: 0.6217 | Acc_bin: 0.8972, F1_bin: 0.8972 | Acc_9: 0.6382, F1_9: 0.6361 | F1_avg_2+9: 0.7667
Epoch  61 | TrainLoss: 0.3808 | Acc18: 0.5969, F1_18: 0.6119 | Acc_bin: 0.8708, F1_bin: 0.8703 | Acc_9: 0.6215, F1_9: 0.6276 | F1_avg_2+9: 0.7490
Epoch  62 | TrainLoss: 0.3651 | Acc18: 0.6062, F1_18: 0.6155 | Acc_bin: 0.8757, F1_bin: 0.8756 | Acc_9: 0.6332, F1_9: 0.6302 | F1_avg_2+9: 0.7529
Epoch  63 | TrainLoss: 0.3336 | Acc18: 0.6117, F1_18: 0.6102 | Acc_bin: 0.8880, F1_bin: 0.8876 | Acc_9: 0.6302, F1_9: 0.6371 | F1_avg_2+9: 0.7624
Epoch  64 | TrainLoss: 0.3453 | Acc18: 0.6209, F1_18: 0.6253 | Acc_bin: 0.8849, F1_bin: 0.8849 | Acc_9: 0.6480, F1_9: 0.6446 | F1_avg_2+9: 0.7647
Epoch  65 | TrainLoss: 0.3302 | Acc18: 0.6068, F1_18: 0.6037 | Acc_bin: 0.8665, F1_bin: 0.8662 | Acc_9: 0.6375, F1_9: 0.6361 | F1_avg_2+9: 0.7512
Epoch  66 | TrainLoss: 0.3543 | Acc18: 0.6148, F1_18: 0.6143 | Acc_bin: 0.8917, F1_bin: 0.8916 | Acc_9: 0.6338, F1_9: 0.6358 | F1_avg_2+9: 0.7637
Epoch  67 | TrainLoss: 0.3654 | Acc18: 0.5858, F1_18: 0.5924 | Acc_bin: 0.8812, F1_bin: 0.8809 | Acc_9: 0.6043, F1_9: 0.6065 | F1_avg_2+9: 0.7437
Epoch  68 | TrainLoss: 0.3522 | Acc18: 0.6172, F1_18: 0.6156 | Acc_bin: 0.8862, F1_bin: 0.8860 | Acc_9: 0.6400, F1_9: 0.6381 | F1_avg_2+9: 0.7620
Epoch  69 | TrainLoss: 0.3624 | Acc18: 0.6246, F1_18: 0.6206 | Acc_bin: 0.8825, F1_bin: 0.8824 | Acc_9: 0.6474, F1_9: 0.6440 | F1_avg_2+9: 0.7632
Epoch  70 | TrainLoss: 0.3396 | Acc18: 0.6129, F1_18: 0.6134 | Acc_bin: 0.8714, F1_bin: 0.8712 | Acc_9: 0.6388, F1_9: 0.6409 | F1_avg_2+9: 0.7561
Epoch  71 | TrainLoss: 0.3345 | Acc18: 0.6055, F1_18: 0.6027 | Acc_bin: 0.8732, F1_bin: 0.8732 | Acc_9: 0.6363, F1_9: 0.6244 | F1_avg_2+9: 0.7488
Epoch  72 | TrainLoss: 0.3528 | Acc18: 0.6025, F1_18: 0.6134 | Acc_bin: 0.8818, F1_bin: 0.8814 | Acc_9: 0.6277, F1_9: 0.6344 | F1_avg_2+9: 0.7579
Epoch  73 | TrainLoss: 0.3362 | Acc18: 0.6185, F1_18: 0.6240 | Acc_bin: 0.8929, F1_bin: 0.8927 | Acc_9: 0.6375, F1_9: 0.6426 | F1_avg_2+9: 0.7676
Epoch  74 | TrainLoss: 0.3505 | Acc18: 0.6068, F1_18: 0.6086 | Acc_bin: 0.8880, F1_bin: 0.8880 | Acc_9: 0.6234, F1_9: 0.6234 | F1_avg_2+9: 0.7557
Epoch  75 | TrainLoss: 0.3168 | Acc18: 0.6111, F1_18: 0.6190 | Acc_bin: 0.8886, F1_bin: 0.8885 | Acc_9: 0.6326, F1_9: 0.6326 | F1_avg_2+9: 0.7605
Epoch  76 | TrainLoss: 0.2985 | Acc18: 0.6105, F1_18: 0.6168 | Acc_bin: 0.8646, F1_bin: 0.8645 | Acc_9: 0.6363, F1_9: 0.6387 | F1_avg_2+9: 0.7516
Epoch  77 | TrainLoss: 0.3394 | Acc18: 0.5908, F1_18: 0.5976 | Acc_bin: 0.8732, F1_bin: 0.8732 | Acc_9: 0.6148, F1_9: 0.6225 | F1_avg_2+9: 0.7478
Epoch  78 | TrainLoss: 0.3230 | Acc18: 0.5957, F1_18: 0.5933 | Acc_bin: 0.8812, F1_bin: 0.8812 | Acc_9: 0.6178, F1_9: 0.6168 | F1_avg_2+9: 0.7490
Epoch  79 | TrainLoss: 0.3034 | Acc18: 0.6185, F1_18: 0.6231 | Acc_bin: 0.8782, F1_bin: 0.8779 | Acc_9: 0.6418, F1_9: 0.6429 | F1_avg_2+9: 0.7604
Epoch  80 | TrainLoss: 0.3264 | Acc18: 0.5908, F1_18: 0.6081 | Acc_bin: 0.8775, F1_bin: 0.8767 | Acc_9: 0.6148, F1_9: 0.6283 | F1_avg_2+9: 0.7525
Epoch  81 | TrainLoss: 0.2975 | Acc18: 0.6148, F1_18: 0.6137 | Acc_bin: 0.8788, F1_bin: 0.8787 | Acc_9: 0.6351, F1_9: 0.6255 | F1_avg_2+9: 0.7521
Epoch  82 | TrainLoss: 0.3109 | Acc18: 0.6117, F1_18: 0.6157 | Acc_bin: 0.8892, F1_bin: 0.8889 | Acc_9: 0.6338, F1_9: 0.6317 | F1_avg_2+9: 0.7603
Epoch  83 | TrainLoss: 0.2761 | Acc18: 0.6086, F1_18: 0.6176 | Acc_bin: 0.8825, F1_bin: 0.8824 | Acc_9: 0.6283, F1_9: 0.6288 | F1_avg_2+9: 0.7556
Epoch  84 | TrainLoss: 0.2902 | Acc18: 0.6166, F1_18: 0.6258 | Acc_bin: 0.8683, F1_bin: 0.8683 | Acc_9: 0.6443, F1_9: 0.6442 | F1_avg_2+9: 0.7563
Epoch  85 | TrainLoss: 0.2790 | Acc18: 0.6012, F1_18: 0.6105 | Acc_bin: 0.8886, F1_bin: 0.8886 | Acc_9: 0.6178, F1_9: 0.6256 | F1_avg_2+9: 0.7571
Epoch  86 | TrainLoss: 0.2916 | Acc18: 0.6172, F1_18: 0.6242 | Acc_bin: 0.8911, F1_bin: 0.8909 | Acc_9: 0.6388, F1_9: 0.6428 | F1_avg_2+9: 0.7669
Epoch  87 | TrainLoss: 0.2719 | Acc18: 0.6142, F1_18: 0.6207 | Acc_bin: 0.8757, F1_bin: 0.8757 | Acc_9: 0.6462, F1_9: 0.6425 | F1_avg_2+9: 0.7591
Epoch  88 | TrainLoss: 0.2946 | Acc18: 0.5963, F1_18: 0.5985 | Acc_bin: 0.8837, F1_bin: 0.8836 | Acc_9: 0.6166, F1_9: 0.6199 | F1_avg_2+9: 0.7518
Epoch  89 | TrainLoss: 0.3027 | Acc18: 0.6068, F1_18: 0.6125 | Acc_bin: 0.8794, F1_bin: 0.8791 | Acc_9: 0.6265, F1_9: 0.6216 | F1_avg_2+9: 0.7503
Epoch  90 | TrainLoss: 0.2954 | Acc18: 0.5926, F1_18: 0.5926 | Acc_bin: 0.8874, F1_bin: 0.8872 | Acc_9: 0.6105, F1_9: 0.6096 | F1_avg_2+9: 0.7484
Epoch  91 | TrainLoss: 0.2978 | Acc18: 0.6098, F1_18: 0.6148 | Acc_bin: 0.8874, F1_bin: 0.8873 | Acc_9: 0.6314, F1_9: 0.6318 | F1_avg_2+9: 0.7595
Epoch  92 | TrainLoss: 0.3161 | Acc18: 0.6031, F1_18: 0.6063 | Acc_bin: 0.8794, F1_bin: 0.8794 | Acc_9: 0.6228, F1_9: 0.6222 | F1_avg_2+9: 0.7508
Epoch  93 | TrainLoss: 0.2797 | Acc18: 0.6092, F1_18: 0.6126 | Acc_bin: 0.8800, F1_bin: 0.8796 | Acc_9: 0.6314, F1_9: 0.6344 | F1_avg_2+9: 0.7570
Epoch  94 | TrainLoss: 0.2469 | Acc18: 0.5975, F1_18: 0.6112 | Acc_bin: 0.8812, F1_bin: 0.8811 | Acc_9: 0.6172, F1_9: 0.6208 | F1_avg_2+9: 0.7510
Epoch  95 | TrainLoss: 0.2801 | Acc18: 0.6129, F1_18: 0.6189 | Acc_bin: 0.8757, F1_bin: 0.8754 | Acc_9: 0.6388, F1_9: 0.6413 | F1_avg_2+9: 0.7583
Epoch  96 | TrainLoss: 0.2794 | Acc18: 0.6006, F1_18: 0.6079 | Acc_bin: 0.8640, F1_bin: 0.8640 | Acc_9: 0.6289, F1_9: 0.6305 | F1_avg_2+9: 0.7472
Epoch  97 | TrainLoss: 0.2334 | Acc18: 0.5957, F1_18: 0.6085 | Acc_bin: 0.8769, F1_bin: 0.8769 | Acc_9: 0.6209, F1_9: 0.6283 | F1_avg_2+9: 0.7526
Epoch  98 | TrainLoss: 0.2507 | Acc18: 0.6203, F1_18: 0.6221 | Acc_bin: 0.8825, F1_bin: 0.8824 | Acc_9: 0.6462, F1_9: 0.6407 | F1_avg_2+9: 0.7615
Epoch  99 | TrainLoss: 0.2532 | Acc18: 0.5815, F1_18: 0.6070 | Acc_bin: 0.8929, F1_bin: 0.8926 | Acc_9: 0.5957, F1_9: 0.6143 | F1_avg_2+9: 0.7534
Epoch 100 | TrainLoss: 0.2719 | Acc18: 0.6105, F1_18: 0.6203 | Acc_bin: 0.8911, F1_bin: 0.8910 | Acc_9: 0.6332, F1_9: 0.6405 | F1_avg_2+9: 0.7658
=== Fold 4 ===
Epoch   1 | TrainLoss: 2.2315 | Acc18: 0.3940, F1_18: 0.3624 | Acc_bin: 0.7653, F1_bin: 0.7653 | Acc_9: 0.4430, F1_9: 0.4002 | F1_avg_2+9: 0.5827
Epoch   2 | TrainLoss: 1.6404 | Acc18: 0.4516, F1_18: 0.4417 | Acc_bin: 0.8064, F1_bin: 0.8063 | Acc_9: 0.4957, F1_9: 0.4684 | F1_avg_2+9: 0.6373
Epoch   3 | TrainLoss: 1.3870 | Acc18: 0.4547, F1_18: 0.4537 | Acc_bin: 0.8137, F1_bin: 0.8137 | Acc_9: 0.4884, F1_9: 0.4734 | F1_avg_2+9: 0.6436
Epoch   4 | TrainLoss: 1.2645 | Acc18: 0.4657, F1_18: 0.4697 | Acc_bin: 0.8419, F1_bin: 0.8419 | Acc_9: 0.5080, F1_9: 0.4956 | F1_avg_2+9: 0.6688
Epoch   5 | TrainLoss: 1.1406 | Acc18: 0.4841, F1_18: 0.4861 | Acc_bin: 0.8468, F1_bin: 0.8464 | Acc_9: 0.5141, F1_9: 0.5091 | F1_avg_2+9: 0.6777
Epoch   6 | TrainLoss: 1.0846 | Acc18: 0.4890, F1_18: 0.4912 | Acc_bin: 0.8278, F1_bin: 0.8277 | Acc_9: 0.5276, F1_9: 0.5154 | F1_avg_2+9: 0.6716
Epoch   7 | TrainLoss: 1.0243 | Acc18: 0.5086, F1_18: 0.5208 | Acc_bin: 0.8431, F1_bin: 0.8430 | Acc_9: 0.5453, F1_9: 0.5370 | F1_avg_2+9: 0.6900
Epoch   8 | TrainLoss: 0.9679 | Acc18: 0.5006, F1_18: 0.5190 | Acc_bin: 0.8499, F1_bin: 0.8499 | Acc_9: 0.5392, F1_9: 0.5438 | F1_avg_2+9: 0.6968
Epoch   9 | TrainLoss: 0.9558 | Acc18: 0.5043, F1_18: 0.5165 | Acc_bin: 0.8554, F1_bin: 0.8554 | Acc_9: 0.5368, F1_9: 0.5258 | F1_avg_2+9: 0.6906
Epoch  10 | TrainLoss: 0.9023 | Acc18: 0.5012, F1_18: 0.5067 | Acc_bin: 0.8456, F1_bin: 0.8445 | Acc_9: 0.5362, F1_9: 0.5359 | F1_avg_2+9: 0.6902
Epoch  11 | TrainLoss: 0.8778 | Acc18: 0.5159, F1_18: 0.5058 | Acc_bin: 0.8346, F1_bin: 0.8344 | Acc_9: 0.5453, F1_9: 0.5364 | F1_avg_2+9: 0.6854
Epoch  12 | TrainLoss: 0.8495 | Acc18: 0.5116, F1_18: 0.5102 | Acc_bin: 0.8578, F1_bin: 0.8578 | Acc_9: 0.5435, F1_9: 0.5427 | F1_avg_2+9: 0.7003
Epoch  13 | TrainLoss: 0.8153 | Acc18: 0.5153, F1_18: 0.5396 | Acc_bin: 0.8597, F1_bin: 0.8597 | Acc_9: 0.5496, F1_9: 0.5551 | F1_avg_2+9: 0.7074
Epoch  14 | TrainLoss: 0.8081 | Acc18: 0.5184, F1_18: 0.5268 | Acc_bin: 0.8474, F1_bin: 0.8473 | Acc_9: 0.5564, F1_9: 0.5486 | F1_avg_2+9: 0.6980
Epoch  15 | TrainLoss: 0.7866 | Acc18: 0.5190, F1_18: 0.5361 | Acc_bin: 0.8658, F1_bin: 0.8658 | Acc_9: 0.5472, F1_9: 0.5512 | F1_avg_2+9: 0.7085
Epoch  16 | TrainLoss: 0.7439 | Acc18: 0.5221, F1_18: 0.5395 | Acc_bin: 0.8609, F1_bin: 0.8607 | Acc_9: 0.5582, F1_9: 0.5595 | F1_avg_2+9: 0.7101
Epoch  17 | TrainLoss: 0.7362 | Acc18: 0.5331, F1_18: 0.5537 | Acc_bin: 0.8713, F1_bin: 0.8710 | Acc_9: 0.5607, F1_9: 0.5613 | F1_avg_2+9: 0.7161
Epoch  18 | TrainLoss: 0.7302 | Acc18: 0.5018, F1_18: 0.5137 | Acc_bin: 0.8646, F1_bin: 0.8638 | Acc_9: 0.5343, F1_9: 0.5352 | F1_avg_2+9: 0.6995
Epoch  19 | TrainLoss: 0.7319 | Acc18: 0.5214, F1_18: 0.5209 | Acc_bin: 0.8652, F1_bin: 0.8650 | Acc_9: 0.5521, F1_9: 0.5426 | F1_avg_2+9: 0.7038
Epoch  20 | TrainLoss: 0.7015 | Acc18: 0.5116, F1_18: 0.5265 | Acc_bin: 0.8738, F1_bin: 0.8734 | Acc_9: 0.5429, F1_9: 0.5404 | F1_avg_2+9: 0.7069
Epoch  21 | TrainLoss: 0.6823 | Acc18: 0.5288, F1_18: 0.5528 | Acc_bin: 0.8493, F1_bin: 0.8479 | Acc_9: 0.5650, F1_9: 0.5751 | F1_avg_2+9: 0.7115
Epoch  22 | TrainLoss: 0.6605 | Acc18: 0.5263, F1_18: 0.5218 | Acc_bin: 0.8511, F1_bin: 0.8511 | Acc_9: 0.5600, F1_9: 0.5558 | F1_avg_2+9: 0.7034
Epoch  23 | TrainLoss: 0.6661 | Acc18: 0.5263, F1_18: 0.5435 | Acc_bin: 0.8566, F1_bin: 0.8564 | Acc_9: 0.5607, F1_9: 0.5738 | F1_avg_2+9: 0.7151
Epoch  24 | TrainLoss: 0.6398 | Acc18: 0.5257, F1_18: 0.5286 | Acc_bin: 0.8652, F1_bin: 0.8650 | Acc_9: 0.5533, F1_9: 0.5495 | F1_avg_2+9: 0.7073
Epoch  25 | TrainLoss: 0.6293 | Acc18: 0.5239, F1_18: 0.5269 | Acc_bin: 0.8542, F1_bin: 0.8533 | Acc_9: 0.5588, F1_9: 0.5565 | F1_avg_2+9: 0.7049
Epoch  26 | TrainLoss: 0.6156 | Acc18: 0.5355, F1_18: 0.5496 | Acc_bin: 0.8585, F1_bin: 0.8584 | Acc_9: 0.5680, F1_9: 0.5690 | F1_avg_2+9: 0.7137
Epoch  27 | TrainLoss: 0.5753 | Acc18: 0.5423, F1_18: 0.5467 | Acc_bin: 0.8658, F1_bin: 0.8655 | Acc_9: 0.5754, F1_9: 0.5724 | F1_avg_2+9: 0.7190
Epoch  28 | TrainLoss: 0.5871 | Acc18: 0.5496, F1_18: 0.5409 | Acc_bin: 0.8701, F1_bin: 0.8700 | Acc_9: 0.5913, F1_9: 0.5828 | F1_avg_2+9: 0.7264
Epoch  29 | TrainLoss: 0.5796 | Acc18: 0.5263, F1_18: 0.5366 | Acc_bin: 0.8652, F1_bin: 0.8643 | Acc_9: 0.5619, F1_9: 0.5653 | F1_avg_2+9: 0.7148
Epoch  30 | TrainLoss: 0.5815 | Acc18: 0.5447, F1_18: 0.5557 | Acc_bin: 0.8597, F1_bin: 0.8596 | Acc_9: 0.5778, F1_9: 0.5713 | F1_avg_2+9: 0.7155
Epoch  31 | TrainLoss: 0.5776 | Acc18: 0.5343, F1_18: 0.5705 | Acc_bin: 0.8548, F1_bin: 0.8545 | Acc_9: 0.5803, F1_9: 0.5929 | F1_avg_2+9: 0.7237
Epoch  32 | TrainLoss: 0.5568 | Acc18: 0.5172, F1_18: 0.5360 | Acc_bin: 0.8444, F1_bin: 0.8435 | Acc_9: 0.5564, F1_9: 0.5624 | F1_avg_2+9: 0.7029
Epoch  33 | TrainLoss: 0.5798 | Acc18: 0.5270, F1_18: 0.5268 | Acc_bin: 0.8738, F1_bin: 0.8732 | Acc_9: 0.5539, F1_9: 0.5444 | F1_avg_2+9: 0.7088
Epoch  34 | TrainLoss: 0.5449 | Acc18: 0.5386, F1_18: 0.5522 | Acc_bin: 0.8627, F1_bin: 0.8627 | Acc_9: 0.5772, F1_9: 0.5780 | F1_avg_2+9: 0.7204
Epoch  35 | TrainLoss: 0.5527 | Acc18: 0.5000, F1_18: 0.4997 | Acc_bin: 0.8474, F1_bin: 0.8472 | Acc_9: 0.5466, F1_9: 0.5346 | F1_avg_2+9: 0.6909
Epoch  36 | TrainLoss: 0.5510 | Acc18: 0.5245, F1_18: 0.5334 | Acc_bin: 0.8615, F1_bin: 0.8613 | Acc_9: 0.5570, F1_9: 0.5575 | F1_avg_2+9: 0.7094
Epoch  37 | TrainLoss: 0.5366 | Acc18: 0.5551, F1_18: 0.5655 | Acc_bin: 0.8591, F1_bin: 0.8591 | Acc_9: 0.5913, F1_9: 0.5863 | F1_avg_2+9: 0.7227
Epoch  38 | TrainLoss: 0.5079 | Acc18: 0.5490, F1_18: 0.5589 | Acc_bin: 0.8713, F1_bin: 0.8712 | Acc_9: 0.5852, F1_9: 0.5800 | F1_avg_2+9: 0.7256
Epoch  39 | TrainLoss: 0.5372 | Acc18: 0.5441, F1_18: 0.5643 | Acc_bin: 0.8683, F1_bin: 0.8681 | Acc_9: 0.5803, F1_9: 0.5868 | F1_avg_2+9: 0.7274
Epoch  40 | TrainLoss: 0.5048 | Acc18: 0.5404, F1_18: 0.5555 | Acc_bin: 0.8701, F1_bin: 0.8700 | Acc_9: 0.5760, F1_9: 0.5762 | F1_avg_2+9: 0.7231
Epoch  41 | TrainLoss: 0.4914 | Acc18: 0.5257, F1_18: 0.5391 | Acc_bin: 0.8523, F1_bin: 0.8521 | Acc_9: 0.5650, F1_9: 0.5657 | F1_avg_2+9: 0.7089
Epoch  42 | TrainLoss: 0.5063 | Acc18: 0.5239, F1_18: 0.5446 | Acc_bin: 0.8499, F1_bin: 0.8498 | Acc_9: 0.5576, F1_9: 0.5565 | F1_avg_2+9: 0.7031
Epoch  43 | TrainLoss: 0.4753 | Acc18: 0.5233, F1_18: 0.5499 | Acc_bin: 0.8591, F1_bin: 0.8590 | Acc_9: 0.5607, F1_9: 0.5597 | F1_avg_2+9: 0.7094
Epoch  44 | TrainLoss: 0.4555 | Acc18: 0.5429, F1_18: 0.5481 | Acc_bin: 0.8854, F1_bin: 0.8850 | Acc_9: 0.5748, F1_9: 0.5732 | F1_avg_2+9: 0.7291
Epoch  45 | TrainLoss: 0.4630 | Acc18: 0.5429, F1_18: 0.5455 | Acc_bin: 0.8719, F1_bin: 0.8717 | Acc_9: 0.5766, F1_9: 0.5699 | F1_avg_2+9: 0.7208
Epoch  46 | TrainLoss: 0.4587 | Acc18: 0.5588, F1_18: 0.5735 | Acc_bin: 0.8744, F1_bin: 0.8743 | Acc_9: 0.5968, F1_9: 0.5919 | F1_avg_2+9: 0.7331
Epoch  47 | TrainLoss: 0.4648 | Acc18: 0.5478, F1_18: 0.5655 | Acc_bin: 0.8689, F1_bin: 0.8689 | Acc_9: 0.5858, F1_9: 0.5878 | F1_avg_2+9: 0.7283
Epoch  48 | TrainLoss: 0.4611 | Acc18: 0.5392, F1_18: 0.5561 | Acc_bin: 0.8542, F1_bin: 0.8542 | Acc_9: 0.5827, F1_9: 0.5899 | F1_avg_2+9: 0.7220
Epoch  49 | TrainLoss: 0.4332 | Acc18: 0.5656, F1_18: 0.5803 | Acc_bin: 0.8701, F1_bin: 0.8699 | Acc_9: 0.5974, F1_9: 0.6043 | F1_avg_2+9: 0.7371
Epoch  50 | TrainLoss: 0.4489 | Acc18: 0.5435, F1_18: 0.5584 | Acc_bin: 0.8732, F1_bin: 0.8727 | Acc_9: 0.5797, F1_9: 0.5796 | F1_avg_2+9: 0.7262
Epoch  51 | TrainLoss: 0.4651 | Acc18: 0.5441, F1_18: 0.5536 | Acc_bin: 0.8676, F1_bin: 0.8673 | Acc_9: 0.5846, F1_9: 0.5833 | F1_avg_2+9: 0.7253
Epoch  52 | TrainLoss: 0.4408 | Acc18: 0.5196, F1_18: 0.5350 | Acc_bin: 0.8548, F1_bin: 0.8544 | Acc_9: 0.5613, F1_9: 0.5591 | F1_avg_2+9: 0.7068
Epoch  53 | TrainLoss: 0.4352 | Acc18: 0.5411, F1_18: 0.5624 | Acc_bin: 0.8676, F1_bin: 0.8676 | Acc_9: 0.5748, F1_9: 0.5837 | F1_avg_2+9: 0.7257
Epoch  54 | TrainLoss: 0.4278 | Acc18: 0.5466, F1_18: 0.5718 | Acc_bin: 0.8640, F1_bin: 0.8639 | Acc_9: 0.5839, F1_9: 0.5907 | F1_avg_2+9: 0.7273
Epoch  55 | TrainLoss: 0.4124 | Acc18: 0.5312, F1_18: 0.5485 | Acc_bin: 0.8560, F1_bin: 0.8558 | Acc_9: 0.5699, F1_9: 0.5716 | F1_avg_2+9: 0.7137
Epoch  56 | TrainLoss: 0.4164 | Acc18: 0.5472, F1_18: 0.5665 | Acc_bin: 0.8658, F1_bin: 0.8657 | Acc_9: 0.5790, F1_9: 0.5835 | F1_avg_2+9: 0.7246
Epoch  57 | TrainLoss: 0.4336 | Acc18: 0.5533, F1_18: 0.5775 | Acc_bin: 0.8615, F1_bin: 0.8612 | Acc_9: 0.5987, F1_9: 0.6048 | F1_avg_2+9: 0.7330
Epoch  58 | TrainLoss: 0.4189 | Acc18: 0.5178, F1_18: 0.5270 | Acc_bin: 0.8480, F1_bin: 0.8480 | Acc_9: 0.5502, F1_9: 0.5459 | F1_avg_2+9: 0.6969
Epoch  59 | TrainLoss: 0.4021 | Acc18: 0.5337, F1_18: 0.5634 | Acc_bin: 0.8468, F1_bin: 0.8468 | Acc_9: 0.5797, F1_9: 0.5845 | F1_avg_2+9: 0.7156
Epoch  60 | TrainLoss: 0.4017 | Acc18: 0.5545, F1_18: 0.5806 | Acc_bin: 0.8548, F1_bin: 0.8547 | Acc_9: 0.5956, F1_9: 0.5993 | F1_avg_2+9: 0.7270
Epoch  61 | TrainLoss: 0.3886 | Acc18: 0.5398, F1_18: 0.5520 | Acc_bin: 0.8695, F1_bin: 0.8694 | Acc_9: 0.5778, F1_9: 0.5789 | F1_avg_2+9: 0.7241
Epoch  62 | TrainLoss: 0.3747 | Acc18: 0.5472, F1_18: 0.5616 | Acc_bin: 0.8713, F1_bin: 0.8713 | Acc_9: 0.5790, F1_9: 0.5735 | F1_avg_2+9: 0.7224
Epoch  63 | TrainLoss: 0.3723 | Acc18: 0.5398, F1_18: 0.5248 | Acc_bin: 0.8719, F1_bin: 0.8719 | Acc_9: 0.5778, F1_9: 0.5652 | F1_avg_2+9: 0.7186
Epoch  64 | TrainLoss: 0.3815 | Acc18: 0.5349, F1_18: 0.5531 | Acc_bin: 0.8468, F1_bin: 0.8467 | Acc_9: 0.5797, F1_9: 0.5794 | F1_avg_2+9: 0.7130
Epoch  65 | TrainLoss: 0.3977 | Acc18: 0.5570, F1_18: 0.5581 | Acc_bin: 0.8670, F1_bin: 0.8669 | Acc_9: 0.5962, F1_9: 0.5939 | F1_avg_2+9: 0.7304
Epoch  66 | TrainLoss: 0.3712 | Acc18: 0.5625, F1_18: 0.5770 | Acc_bin: 0.8683, F1_bin: 0.8676 | Acc_9: 0.5987, F1_9: 0.6021 | F1_avg_2+9: 0.7349
Epoch  67 | TrainLoss: 0.3724 | Acc18: 0.5478, F1_18: 0.5658 | Acc_bin: 0.8536, F1_bin: 0.8535 | Acc_9: 0.5803, F1_9: 0.5879 | F1_avg_2+9: 0.7207
Epoch  68 | TrainLoss: 0.3706 | Acc18: 0.5619, F1_18: 0.5791 | Acc_bin: 0.8848, F1_bin: 0.8846 | Acc_9: 0.5888, F1_9: 0.5958 | F1_avg_2+9: 0.7402
Epoch  69 | TrainLoss: 0.3731 | Acc18: 0.5619, F1_18: 0.5717 | Acc_bin: 0.8725, F1_bin: 0.8725 | Acc_9: 0.5925, F1_9: 0.5913 | F1_avg_2+9: 0.7319
Epoch  70 | TrainLoss: 0.3690 | Acc18: 0.5539, F1_18: 0.5571 | Acc_bin: 0.8683, F1_bin: 0.8677 | Acc_9: 0.5888, F1_9: 0.5799 | F1_avg_2+9: 0.7238
Epoch  71 | TrainLoss: 0.3789 | Acc18: 0.5643, F1_18: 0.5693 | Acc_bin: 0.8652, F1_bin: 0.8652 | Acc_9: 0.5938, F1_9: 0.5924 | F1_avg_2+9: 0.7288
Epoch  72 | TrainLoss: 0.3934 | Acc18: 0.5625, F1_18: 0.5665 | Acc_bin: 0.8744, F1_bin: 0.8741 | Acc_9: 0.5987, F1_9: 0.5883 | F1_avg_2+9: 0.7312
Epoch  73 | TrainLoss: 0.3510 | Acc18: 0.5674, F1_18: 0.5844 | Acc_bin: 0.8725, F1_bin: 0.8725 | Acc_9: 0.6036, F1_9: 0.6070 | F1_avg_2+9: 0.7398
Epoch  74 | TrainLoss: 0.3354 | Acc18: 0.5594, F1_18: 0.5702 | Acc_bin: 0.8719, F1_bin: 0.8719 | Acc_9: 0.5907, F1_9: 0.5925 | F1_avg_2+9: 0.7322
Epoch  75 | TrainLoss: 0.3482 | Acc18: 0.5588, F1_18: 0.5756 | Acc_bin: 0.8689, F1_bin: 0.8687 | Acc_9: 0.5962, F1_9: 0.6036 | F1_avg_2+9: 0.7361
Epoch  76 | TrainLoss: 0.3624 | Acc18: 0.5631, F1_18: 0.5813 | Acc_bin: 0.8664, F1_bin: 0.8664 | Acc_9: 0.6036, F1_9: 0.6034 | F1_avg_2+9: 0.7349
Epoch  77 | TrainLoss: 0.3489 | Acc18: 0.5441, F1_18: 0.5591 | Acc_bin: 0.8707, F1_bin: 0.8706 | Acc_9: 0.5797, F1_9: 0.5810 | F1_avg_2+9: 0.7258
Epoch  78 | TrainLoss: 0.3444 | Acc18: 0.5460, F1_18: 0.5540 | Acc_bin: 0.8689, F1_bin: 0.8688 | Acc_9: 0.5846, F1_9: 0.5834 | F1_avg_2+9: 0.7261
Epoch  79 | TrainLoss: 0.3101 | Acc18: 0.5551, F1_18: 0.5560 | Acc_bin: 0.8701, F1_bin: 0.8701 | Acc_9: 0.5925, F1_9: 0.5975 | F1_avg_2+9: 0.7338
Epoch  80 | TrainLoss: 0.3268 | Acc18: 0.5417, F1_18: 0.5506 | Acc_bin: 0.8425, F1_bin: 0.8424 | Acc_9: 0.5882, F1_9: 0.5859 | F1_avg_2+9: 0.7142
Epoch  81 | TrainLoss: 0.2979 | Acc18: 0.5502, F1_18: 0.5409 | Acc_bin: 0.8634, F1_bin: 0.8629 | Acc_9: 0.5901, F1_9: 0.5810 | F1_avg_2+9: 0.7220
Epoch  82 | TrainLoss: 0.3288 | Acc18: 0.5551, F1_18: 0.5645 | Acc_bin: 0.8634, F1_bin: 0.8626 | Acc_9: 0.5913, F1_9: 0.5901 | F1_avg_2+9: 0.7264
Epoch  83 | TrainLoss: 0.3383 | Acc18: 0.5435, F1_18: 0.5599 | Acc_bin: 0.8683, F1_bin: 0.8680 | Acc_9: 0.5723, F1_9: 0.5775 | F1_avg_2+9: 0.7228
Epoch  84 | TrainLoss: 0.3395 | Acc18: 0.5576, F1_18: 0.5669 | Acc_bin: 0.8664, F1_bin: 0.8659 | Acc_9: 0.5925, F1_9: 0.5984 | F1_avg_2+9: 0.7321
Epoch  85 | TrainLoss: 0.3252 | Acc18: 0.5404, F1_18: 0.5486 | Acc_bin: 0.8536, F1_bin: 0.8535 | Acc_9: 0.5754, F1_9: 0.5690 | F1_avg_2+9: 0.7113
Epoch  86 | TrainLoss: 0.3238 | Acc18: 0.5429, F1_18: 0.5717 | Acc_bin: 0.8640, F1_bin: 0.8639 | Acc_9: 0.5815, F1_9: 0.5915 | F1_avg_2+9: 0.7277
Epoch  87 | TrainLoss: 0.3256 | Acc18: 0.5368, F1_18: 0.5505 | Acc_bin: 0.8621, F1_bin: 0.8621 | Acc_9: 0.5680, F1_9: 0.5687 | F1_avg_2+9: 0.7154
Epoch  88 | TrainLoss: 0.3311 | Acc18: 0.5619, F1_18: 0.5644 | Acc_bin: 0.8676, F1_bin: 0.8676 | Acc_9: 0.5993, F1_9: 0.5921 | F1_avg_2+9: 0.7299
Epoch  89 | TrainLoss: 0.2947 | Acc18: 0.5411, F1_18: 0.5432 | Acc_bin: 0.8787, F1_bin: 0.8784 | Acc_9: 0.5741, F1_9: 0.5739 | F1_avg_2+9: 0.7262
Epoch  90 | TrainLoss: 0.3023 | Acc18: 0.5423, F1_18: 0.5381 | Acc_bin: 0.8591, F1_bin: 0.8591 | Acc_9: 0.5846, F1_9: 0.5703 | F1_avg_2+9: 0.7147
Epoch  91 | TrainLoss: 0.2945 | Acc18: 0.5447, F1_18: 0.5602 | Acc_bin: 0.8719, F1_bin: 0.8719 | Acc_9: 0.5833, F1_9: 0.5809 | F1_avg_2+9: 0.7264
Epoch  92 | TrainLoss: 0.3152 | Acc18: 0.5570, F1_18: 0.5671 | Acc_bin: 0.8732, F1_bin: 0.8730 | Acc_9: 0.5950, F1_9: 0.5966 | F1_avg_2+9: 0.7348
Epoch  93 | TrainLoss: 0.3089 | Acc18: 0.5478, F1_18: 0.5511 | Acc_bin: 0.8689, F1_bin: 0.8688 | Acc_9: 0.5797, F1_9: 0.5734 | F1_avg_2+9: 0.7211
Early stopping
=== Fold 5 ===
Epoch   1 | TrainLoss: 2.3323 | Acc18: 0.3973, F1_18: 0.3458 | Acc_bin: 0.8068, F1_bin: 0.8065 | Acc_9: 0.4262, F1_9: 0.3794 | F1_avg_2+9: 0.5930
Epoch   2 | TrainLoss: 1.7615 | Acc18: 0.4810, F1_18: 0.4516 | Acc_bin: 0.8685, F1_bin: 0.8684 | Acc_9: 0.5012, F1_9: 0.4716 | F1_avg_2+9: 0.6700
Epoch   3 | TrainLoss: 1.4927 | Acc18: 0.5035, F1_18: 0.4794 | Acc_bin: 0.8806, F1_bin: 0.8805 | Acc_9: 0.5277, F1_9: 0.4979 | F1_avg_2+9: 0.6892
Epoch   4 | TrainLoss: 1.3494 | Acc18: 0.5185, F1_18: 0.5174 | Acc_bin: 0.8916, F1_bin: 0.8909 | Acc_9: 0.5398, F1_9: 0.5336 | F1_avg_2+9: 0.7123
Epoch   5 | TrainLoss: 1.2328 | Acc18: 0.5490, F1_18: 0.5368 | Acc_bin: 0.8847, F1_bin: 0.8846 | Acc_9: 0.5744, F1_9: 0.5576 | F1_avg_2+9: 0.7211
Epoch   6 | TrainLoss: 1.1729 | Acc18: 0.5473, F1_18: 0.5461 | Acc_bin: 0.8922, F1_bin: 0.8920 | Acc_9: 0.5681, F1_9: 0.5627 | F1_avg_2+9: 0.7273
Epoch   7 | TrainLoss: 1.1097 | Acc18: 0.5496, F1_18: 0.5376 | Acc_bin: 0.8875, F1_bin: 0.8873 | Acc_9: 0.5761, F1_9: 0.5610 | F1_avg_2+9: 0.7242
Epoch   8 | TrainLoss: 1.0472 | Acc18: 0.5421, F1_18: 0.5600 | Acc_bin: 0.9014, F1_bin: 0.9009 | Acc_9: 0.5657, F1_9: 0.5673 | F1_avg_2+9: 0.7341
Epoch   9 | TrainLoss: 1.0021 | Acc18: 0.5669, F1_18: 0.5746 | Acc_bin: 0.9043, F1_bin: 0.9038 | Acc_9: 0.5928, F1_9: 0.5903 | F1_avg_2+9: 0.7470
Epoch  10 | TrainLoss: 0.9504 | Acc18: 0.5502, F1_18: 0.5435 | Acc_bin: 0.9008, F1_bin: 0.9006 | Acc_9: 0.5767, F1_9: 0.5697 | F1_avg_2+9: 0.7351
Epoch  11 | TrainLoss: 0.9277 | Acc18: 0.5657, F1_18: 0.5581 | Acc_bin: 0.8997, F1_bin: 0.8991 | Acc_9: 0.5877, F1_9: 0.5847 | F1_avg_2+9: 0.7419
Epoch  12 | TrainLoss: 0.9284 | Acc18: 0.5606, F1_18: 0.5641 | Acc_bin: 0.9031, F1_bin: 0.9028 | Acc_9: 0.5888, F1_9: 0.5868 | F1_avg_2+9: 0.7448
Epoch  13 | TrainLoss: 0.8659 | Acc18: 0.5317, F1_18: 0.5402 | Acc_bin: 0.9031, F1_bin: 0.9028 | Acc_9: 0.5559, F1_9: 0.5516 | F1_avg_2+9: 0.7272
Epoch  14 | TrainLoss: 0.8471 | Acc18: 0.5669, F1_18: 0.5812 | Acc_bin: 0.8933, F1_bin: 0.8919 | Acc_9: 0.5952, F1_9: 0.5977 | F1_avg_2+9: 0.7448
Epoch  15 | TrainLoss: 0.8215 | Acc18: 0.5675, F1_18: 0.5566 | Acc_bin: 0.9031, F1_bin: 0.9029 | Acc_9: 0.5923, F1_9: 0.5864 | F1_avg_2+9: 0.7446
Epoch  16 | TrainLoss: 0.8085 | Acc18: 0.5732, F1_18: 0.5822 | Acc_bin: 0.9095, F1_bin: 0.9090 | Acc_9: 0.5934, F1_9: 0.5901 | F1_avg_2+9: 0.7495
Epoch  17 | TrainLoss: 0.7694 | Acc18: 0.5675, F1_18: 0.5648 | Acc_bin: 0.9037, F1_bin: 0.9031 | Acc_9: 0.5900, F1_9: 0.5827 | F1_avg_2+9: 0.7429
Epoch  18 | TrainLoss: 0.7686 | Acc18: 0.5721, F1_18: 0.5875 | Acc_bin: 0.9031, F1_bin: 0.9022 | Acc_9: 0.5911, F1_9: 0.5900 | F1_avg_2+9: 0.7461
Epoch  19 | TrainLoss: 0.7415 | Acc18: 0.5755, F1_18: 0.5658 | Acc_bin: 0.9089, F1_bin: 0.9087 | Acc_9: 0.5946, F1_9: 0.5815 | F1_avg_2+9: 0.7451
Epoch  20 | TrainLoss: 0.7135 | Acc18: 0.5709, F1_18: 0.5807 | Acc_bin: 0.9072, F1_bin: 0.9070 | Acc_9: 0.5928, F1_9: 0.5910 | F1_avg_2+9: 0.7490
Epoch  21 | TrainLoss: 0.7146 | Acc18: 0.5727, F1_18: 0.5691 | Acc_bin: 0.9083, F1_bin: 0.9079 | Acc_9: 0.5905, F1_9: 0.5755 | F1_avg_2+9: 0.7417
Epoch  22 | TrainLoss: 0.7324 | Acc18: 0.5721, F1_18: 0.5733 | Acc_bin: 0.9066, F1_bin: 0.9059 | Acc_9: 0.5877, F1_9: 0.5821 | F1_avg_2+9: 0.7440
Epoch  23 | TrainLoss: 0.6645 | Acc18: 0.5790, F1_18: 0.5830 | Acc_bin: 0.9089, F1_bin: 0.9086 | Acc_9: 0.5998, F1_9: 0.5957 | F1_avg_2+9: 0.7522
Epoch  24 | TrainLoss: 0.6970 | Acc18: 0.5802, F1_18: 0.5871 | Acc_bin: 0.8962, F1_bin: 0.8960 | Acc_9: 0.6044, F1_9: 0.6063 | F1_avg_2+9: 0.7512
Epoch  25 | TrainLoss: 0.6598 | Acc18: 0.5548, F1_18: 0.5577 | Acc_bin: 0.8939, F1_bin: 0.8938 | Acc_9: 0.5767, F1_9: 0.5760 | F1_avg_2+9: 0.7349
Epoch  26 | TrainLoss: 0.6684 | Acc18: 0.5681, F1_18: 0.5771 | Acc_bin: 0.9043, F1_bin: 0.9033 | Acc_9: 0.5917, F1_9: 0.5956 | F1_avg_2+9: 0.7494
Epoch  27 | TrainLoss: 0.6385 | Acc18: 0.5854, F1_18: 0.5927 | Acc_bin: 0.9066, F1_bin: 0.9059 | Acc_9: 0.6055, F1_9: 0.6111 | F1_avg_2+9: 0.7585
Epoch  28 | TrainLoss: 0.6392 | Acc18: 0.5773, F1_18: 0.5646 | Acc_bin: 0.9020, F1_bin: 0.9016 | Acc_9: 0.5992, F1_9: 0.5827 | F1_avg_2+9: 0.7421
Epoch  29 | TrainLoss: 0.6322 | Acc18: 0.5744, F1_18: 0.5654 | Acc_bin: 0.9043, F1_bin: 0.9035 | Acc_9: 0.5969, F1_9: 0.5953 | F1_avg_2+9: 0.7494
Epoch  30 | TrainLoss: 0.6034 | Acc18: 0.5819, F1_18: 0.5964 | Acc_bin: 0.9031, F1_bin: 0.9023 | Acc_9: 0.6003, F1_9: 0.6051 | F1_avg_2+9: 0.7537
Epoch  31 | TrainLoss: 0.5886 | Acc18: 0.5865, F1_18: 0.5899 | Acc_bin: 0.9054, F1_bin: 0.9051 | Acc_9: 0.6073, F1_9: 0.6045 | F1_avg_2+9: 0.7548
Epoch  32 | TrainLoss: 0.6005 | Acc18: 0.5704, F1_18: 0.5710 | Acc_bin: 0.8973, F1_bin: 0.8972 | Acc_9: 0.5905, F1_9: 0.5846 | F1_avg_2+9: 0.7409
Epoch  33 | TrainLoss: 0.5997 | Acc18: 0.5802, F1_18: 0.6063 | Acc_bin: 0.8968, F1_bin: 0.8963 | Acc_9: 0.6050, F1_9: 0.6118 | F1_avg_2+9: 0.7541
Epoch  34 | TrainLoss: 0.5721 | Acc18: 0.5934, F1_18: 0.5972 | Acc_bin: 0.9100, F1_bin: 0.9097 | Acc_9: 0.6176, F1_9: 0.6092 | F1_avg_2+9: 0.7595
Epoch  35 | TrainLoss: 0.5485 | Acc18: 0.6003, F1_18: 0.6051 | Acc_bin: 0.9135, F1_bin: 0.9134 | Acc_9: 0.6200, F1_9: 0.6158 | F1_avg_2+9: 0.7646
Epoch  36 | TrainLoss: 0.5614 | Acc18: 0.6003, F1_18: 0.6105 | Acc_bin: 0.9164, F1_bin: 0.9161 | Acc_9: 0.6246, F1_9: 0.6241 | F1_avg_2+9: 0.7701
Epoch  37 | TrainLoss: 0.5735 | Acc18: 0.5819, F1_18: 0.5870 | Acc_bin: 0.9083, F1_bin: 0.9078 | Acc_9: 0.6027, F1_9: 0.6019 | F1_avg_2+9: 0.7549
Epoch  38 | TrainLoss: 0.5372 | Acc18: 0.5969, F1_18: 0.6048 | Acc_bin: 0.9008, F1_bin: 0.9008 | Acc_9: 0.6194, F1_9: 0.6054 | F1_avg_2+9: 0.7531
Epoch  39 | TrainLoss: 0.5554 | Acc18: 0.6021, F1_18: 0.6063 | Acc_bin: 0.9072, F1_bin: 0.9069 | Acc_9: 0.6223, F1_9: 0.6156 | F1_avg_2+9: 0.7612
Epoch  40 | TrainLoss: 0.5163 | Acc18: 0.5802, F1_18: 0.5850 | Acc_bin: 0.9008, F1_bin: 0.9001 | Acc_9: 0.6044, F1_9: 0.6002 | F1_avg_2+9: 0.7502
Epoch  41 | TrainLoss: 0.5339 | Acc18: 0.6073, F1_18: 0.6087 | Acc_bin: 0.9152, F1_bin: 0.9149 | Acc_9: 0.6303, F1_9: 0.6191 | F1_avg_2+9: 0.7670
Epoch  42 | TrainLoss: 0.5471 | Acc18: 0.5992, F1_18: 0.6076 | Acc_bin: 0.9118, F1_bin: 0.9117 | Acc_9: 0.6228, F1_9: 0.6101 | F1_avg_2+9: 0.7609
Epoch  43 | TrainLoss: 0.4791 | Acc18: 0.5917, F1_18: 0.5878 | Acc_bin: 0.9048, F1_bin: 0.9046 | Acc_9: 0.6096, F1_9: 0.6070 | F1_avg_2+9: 0.7558
Epoch  44 | TrainLoss: 0.5270 | Acc18: 0.6055, F1_18: 0.6129 | Acc_bin: 0.9135, F1_bin: 0.9131 | Acc_9: 0.6269, F1_9: 0.6188 | F1_avg_2+9: 0.7660
Epoch  45 | TrainLoss: 0.5037 | Acc18: 0.5773, F1_18: 0.5858 | Acc_bin: 0.9066, F1_bin: 0.9063 | Acc_9: 0.6038, F1_9: 0.6055 | F1_avg_2+9: 0.7559
Epoch  46 | TrainLoss: 0.5009 | Acc18: 0.5934, F1_18: 0.5963 | Acc_bin: 0.8933, F1_bin: 0.8931 | Acc_9: 0.6188, F1_9: 0.6108 | F1_avg_2+9: 0.7519
Epoch  47 | TrainLoss: 0.5052 | Acc18: 0.5934, F1_18: 0.5956 | Acc_bin: 0.9158, F1_bin: 0.9157 | Acc_9: 0.6078, F1_9: 0.5995 | F1_avg_2+9: 0.7576
Epoch  48 | TrainLoss: 0.4873 | Acc18: 0.5790, F1_18: 0.5813 | Acc_bin: 0.9037, F1_bin: 0.9032 | Acc_9: 0.6003, F1_9: 0.5903 | F1_avg_2+9: 0.7468
Epoch  49 | TrainLoss: 0.5146 | Acc18: 0.5871, F1_18: 0.5842 | Acc_bin: 0.9037, F1_bin: 0.9037 | Acc_9: 0.6119, F1_9: 0.5930 | F1_avg_2+9: 0.7483
Epoch  50 | TrainLoss: 0.4803 | Acc18: 0.5900, F1_18: 0.5912 | Acc_bin: 0.9089, F1_bin: 0.9087 | Acc_9: 0.6061, F1_9: 0.5972 | F1_avg_2+9: 0.7529
Epoch  51 | TrainLoss: 0.4661 | Acc18: 0.5888, F1_18: 0.5833 | Acc_bin: 0.9164, F1_bin: 0.9163 | Acc_9: 0.6044, F1_9: 0.5931 | F1_avg_2+9: 0.7547
Epoch  52 | TrainLoss: 0.4480 | Acc18: 0.6148, F1_18: 0.6258 | Acc_bin: 0.9048, F1_bin: 0.9046 | Acc_9: 0.6367, F1_9: 0.6281 | F1_avg_2+9: 0.7664
Epoch  53 | TrainLoss: 0.4594 | Acc18: 0.6176, F1_18: 0.6221 | Acc_bin: 0.9164, F1_bin: 0.9163 | Acc_9: 0.6361, F1_9: 0.6291 | F1_avg_2+9: 0.7727
Epoch  54 | TrainLoss: 0.4581 | Acc18: 0.6009, F1_18: 0.6004 | Acc_bin: 0.9083, F1_bin: 0.9080 | Acc_9: 0.6234, F1_9: 0.6130 | F1_avg_2+9: 0.7605
Epoch  55 | TrainLoss: 0.4544 | Acc18: 0.6159, F1_18: 0.6126 | Acc_bin: 0.9175, F1_bin: 0.9174 | Acc_9: 0.6373, F1_9: 0.6293 | F1_avg_2+9: 0.7734
Epoch  56 | TrainLoss: 0.4345 | Acc18: 0.6050, F1_18: 0.6153 | Acc_bin: 0.9031, F1_bin: 0.9025 | Acc_9: 0.6298, F1_9: 0.6269 | F1_avg_2+9: 0.7647
Epoch  57 | TrainLoss: 0.4534 | Acc18: 0.5888, F1_18: 0.5912 | Acc_bin: 0.9112, F1_bin: 0.9108 | Acc_9: 0.6090, F1_9: 0.6022 | F1_avg_2+9: 0.7565
Epoch  58 | TrainLoss: 0.4569 | Acc18: 0.6113, F1_18: 0.6207 | Acc_bin: 0.9193, F1_bin: 0.9191 | Acc_9: 0.6309, F1_9: 0.6325 | F1_avg_2+9: 0.7758
Epoch  59 | TrainLoss: 0.4448 | Acc18: 0.5738, F1_18: 0.5837 | Acc_bin: 0.9014, F1_bin: 0.9009 | Acc_9: 0.5946, F1_9: 0.5906 | F1_avg_2+9: 0.7458
Epoch  60 | TrainLoss: 0.3965 | Acc18: 0.5911, F1_18: 0.5915 | Acc_bin: 0.9141, F1_bin: 0.9137 | Acc_9: 0.6153, F1_9: 0.6079 | F1_avg_2+9: 0.7608
Epoch  61 | TrainLoss: 0.4491 | Acc18: 0.5830, F1_18: 0.6030 | Acc_bin: 0.9002, F1_bin: 0.8997 | Acc_9: 0.6148, F1_9: 0.6181 | F1_avg_2+9: 0.7589
Epoch  62 | TrainLoss: 0.4254 | Acc18: 0.6148, F1_18: 0.6321 | Acc_bin: 0.9083, F1_bin: 0.9079 | Acc_9: 0.6355, F1_9: 0.6366 | F1_avg_2+9: 0.7722
Epoch  63 | TrainLoss: 0.4114 | Acc18: 0.5975, F1_18: 0.6106 | Acc_bin: 0.9175, F1_bin: 0.9172 | Acc_9: 0.6200, F1_9: 0.6193 | F1_avg_2+9: 0.7682
Epoch  64 | TrainLoss: 0.4362 | Acc18: 0.5882, F1_18: 0.5955 | Acc_bin: 0.9043, F1_bin: 0.9035 | Acc_9: 0.6096, F1_9: 0.6048 | F1_avg_2+9: 0.7542
Epoch  65 | TrainLoss: 0.4555 | Acc18: 0.6188, F1_18: 0.6188 | Acc_bin: 0.9100, F1_bin: 0.9100 | Acc_9: 0.6407, F1_9: 0.6284 | F1_avg_2+9: 0.7692
Epoch  66 | TrainLoss: 0.3862 | Acc18: 0.6090, F1_18: 0.6160 | Acc_bin: 0.9014, F1_bin: 0.9014 | Acc_9: 0.6309, F1_9: 0.6259 | F1_avg_2+9: 0.7636
Epoch  67 | TrainLoss: 0.4000 | Acc18: 0.6073, F1_18: 0.6128 | Acc_bin: 0.9164, F1_bin: 0.9163 | Acc_9: 0.6269, F1_9: 0.6185 | F1_avg_2+9: 0.7674
Epoch  68 | TrainLoss: 0.4195 | Acc18: 0.6107, F1_18: 0.6209 | Acc_bin: 0.9025, F1_bin: 0.9017 | Acc_9: 0.6332, F1_9: 0.6306 | F1_avg_2+9: 0.7661
Epoch  69 | TrainLoss: 0.3976 | Acc18: 0.6073, F1_18: 0.6123 | Acc_bin: 0.9181, F1_bin: 0.9179 | Acc_9: 0.6292, F1_9: 0.6275 | F1_avg_2+9: 0.7727
Epoch  70 | TrainLoss: 0.4044 | Acc18: 0.5779, F1_18: 0.5995 | Acc_bin: 0.9054, F1_bin: 0.9050 | Acc_9: 0.6055, F1_9: 0.6166 | F1_avg_2+9: 0.7608
Epoch  71 | TrainLoss: 0.4128 | Acc18: 0.6038, F1_18: 0.6112 | Acc_bin: 0.9077, F1_bin: 0.9077 | Acc_9: 0.6240, F1_9: 0.6204 | F1_avg_2+9: 0.7640
Epoch  72 | TrainLoss: 0.3894 | Acc18: 0.6217, F1_18: 0.6305 | Acc_bin: 0.9135, F1_bin: 0.9133 | Acc_9: 0.6413, F1_9: 0.6370 | F1_avg_2+9: 0.7751
Epoch  73 | TrainLoss: 0.3669 | Acc18: 0.6205, F1_18: 0.6237 | Acc_bin: 0.9112, F1_bin: 0.9110 | Acc_9: 0.6419, F1_9: 0.6346 | F1_avg_2+9: 0.7728
Epoch  74 | TrainLoss: 0.3717 | Acc18: 0.6015, F1_18: 0.6112 | Acc_bin: 0.9083, F1_bin: 0.9077 | Acc_9: 0.6234, F1_9: 0.6221 | F1_avg_2+9: 0.7649
Epoch  75 | TrainLoss: 0.3772 | Acc18: 0.6136, F1_18: 0.6269 | Acc_bin: 0.9129, F1_bin: 0.9129 | Acc_9: 0.6344, F1_9: 0.6343 | F1_avg_2+9: 0.7736
Epoch  76 | TrainLoss: 0.3944 | Acc18: 0.5807, F1_18: 0.5918 | Acc_bin: 0.9020, F1_bin: 0.9009 | Acc_9: 0.6015, F1_9: 0.6074 | F1_avg_2+9: 0.7541
Epoch  77 | TrainLoss: 0.3706 | Acc18: 0.6119, F1_18: 0.6231 | Acc_bin: 0.9135, F1_bin: 0.9131 | Acc_9: 0.6315, F1_9: 0.6283 | F1_avg_2+9: 0.7707
Epoch  78 | TrainLoss: 0.3796 | Acc18: 0.6015, F1_18: 0.6175 | Acc_bin: 0.9008, F1_bin: 0.9005 | Acc_9: 0.6257, F1_9: 0.6268 | F1_avg_2+9: 0.7637
Epoch  79 | TrainLoss: 0.3604 | Acc18: 0.5928, F1_18: 0.5931 | Acc_bin: 0.9141, F1_bin: 0.9139 | Acc_9: 0.6153, F1_9: 0.6152 | F1_avg_2+9: 0.7645
Epoch  80 | TrainLoss: 0.3632 | Acc18: 0.6015, F1_18: 0.6085 | Acc_bin: 0.9072, F1_bin: 0.9066 | Acc_9: 0.6286, F1_9: 0.6283 | F1_avg_2+9: 0.7674
Epoch  81 | TrainLoss: 0.3871 | Acc18: 0.6113, F1_18: 0.6165 | Acc_bin: 0.9170, F1_bin: 0.9169 | Acc_9: 0.6292, F1_9: 0.6219 | F1_avg_2+9: 0.7694
Epoch  82 | TrainLoss: 0.3945 | Acc18: 0.6067, F1_18: 0.6151 | Acc_bin: 0.9060, F1_bin: 0.9058 | Acc_9: 0.6309, F1_9: 0.6234 | F1_avg_2+9: 0.7646
Early stopping

"""
