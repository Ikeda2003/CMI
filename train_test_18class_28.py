#TOFブランチを画像として追加 CV=0.824
#CMI 2025 デモ提出 バージョン62 LB=0.74
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ========= 設定 ==========
RAW_CSV = "train.csv"
SAVE_DIR = "train_test_18class_28"
PAD_PERCENTILE = 95
BATCH_SIZE = 64
EPOCHS = 100
LR_INIT = 1e-3
WD = 1e-4
PATIENCE = 20
os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= データ準備 ==========
def feature_eng(df):
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
    return df[cols_wo_insert[:insert_index] + insert_cols + cols_wo_insert[insert_index:]]

df = pd.read_csv(RAW_CSV)
df = feature_eng(df)
df["gesture"] = df["gesture"].fillna("unknown")
le = LabelEncoder()
df["gesture_class"] = le.fit_transform(df["gesture"])

# ========= 特徴量カラム =========
num_cols = df.select_dtypes(include=[np.number]).columns
imu_cols = [c for c in num_cols if c.startswith(('acc_', 'rot_', 'acc_mag', 'rot_angle', 'acc_mag_jerk', 'rot_angle_vel', 'thm_')) and c != 'gesture_class']
tof_cols = [c for c in num_cols if c.startswith('tof_') and c != 'gesture_class']
imu_dim, tof_dim = len(imu_cols), len(tof_cols)
lens = df.groupby("sequence_id").size().values
pad_len = int(np.percentile(lens, PAD_PERCENTILE))

def to_binary(y): return [0 if i<9 else 1 for i in y]
def to_9class(y): return [i%9 for i in y]

print("FEATURE COLS:", imu_cols + tof_cols)

# ========= データセット =========
class Img8x8Dataset(Dataset):
    def __init__(self, imu_seqs, tof_seqs, y):
        self.imu_seqs = torch.tensor(imu_seqs, dtype=torch.float32)
        self.tof_seqs = torch.tensor(tof_seqs, dtype=torch.float32)  # (N,T,8,8)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.imu_seqs)
    def __getitem__(self, idx): return self.imu_seqs[idx], self.tof_seqs[idx], self.y[idx]

# ========= モデル =========
class MultiScaleConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,5,7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, k, padding=k//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ) for k in kernel_sizes
        ])
    def forward(self, x):
        return torch.cat([conv(x) for conv in self.convs], dim=1)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction),
            nn.ReLU(),
            nn.Linear(channels//reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b,c,_ = x.size()
        y = self.pool(x).view(b,c)
        y = self.fc(y).view(b,c,1)
        return x * y

class ResidualSEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, pool_size=2, drop=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=k//2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=k//2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch!=out_ch else nn.Identity()
        self.bn_sc = nn.BatchNorm1d(out_ch) if in_ch!=out_ch else nn.Identity()
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

class TinyCNN(nn.Module):
    def __init__(self, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self,x):
        x = self.net(x)
        return x.view(x.size(0), -1)

class AttentionLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model,1)
    def forward(self, x):
        score = torch.tanh(self.fc(x)).squeeze(-1)
        weights = F.softmax(score, dim=1).unsqueeze(-1)
        return (x*weights).sum(dim=1)

class TwoBranchImageSeqModel(nn.Module):
    def __init__(self, imu_dim, num_classes):
        super().__init__()
        self.imu_branches = nn.ModuleList([
            nn.Sequential(
                MultiScaleConv1d(1,12),
                ResidualSEBlock(36,48),
                ResidualSEBlock(48,48)
            ) for _ in range(imu_dim)
        ])
        self.imu_gru = nn.GRU(48*imu_dim, 64, batch_first=True, bidirectional=True)
        self.tof_cnn = TinyCNN(32)
        self.tof_gru = nn.GRU(32, 64, batch_first=True, bidirectional=True)
        self.fuse_gru = nn.GRU(128+128, 128, batch_first=True, bidirectional=True)
        self.attn = AttentionLayer(256)
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    def forward(self, x_imu, x_tof):
        imu_feats = []
        for i in range(x_imu.shape[2]):
            xi = x_imu[:,:,i].unsqueeze(1)
            fi = self.imu_branches[i](xi).transpose(1,2)
            imu_feats.append(fi)
        imu_feat = torch.cat(imu_feats, dim=2)
        imu_out,_ = self.imu_gru(imu_feat)
        B,T,H,W = x_tof.shape
        x_flat = x_tof.view(B*T,1,H,W)
        tof_feats = self.tof_cnn(x_flat).view(B,T,-1)
        tof_out,_ = self.tof_gru(tof_feats)
        tof_out = F.interpolate(tof_out.permute(0,2,1), size=imu_out.size(1), mode='linear').permute(0,2,1)
        x = torch.cat([imu_out, tof_out], dim=2)
        x,_ = self.fuse_gru(x)
        x = self.attn(x)
        return self.head(x), None

# ========= 学習 =========
kf = GroupKFold(n_splits=5)
seq_ids = df["sequence_id"].unique()
subject_map = df.drop_duplicates("sequence_id").set_index("sequence_id")["subject"]
groups = [subject_map[sid] for sid in seq_ids]

for fold, (tr_idx, va_idx) in enumerate(kf.split(seq_ids, groups=groups)):
    print(f"\n=== Fold {fold+1} ===")
    fold_dir = os.path.join(SAVE_DIR, f"fold{fold+1}")
    os.makedirs(fold_dir, exist_ok=True)

    train_ids, val_ids = seq_ids[tr_idx], seq_ids[va_idx]
    train_df, val_df = df[df["sequence_id"].isin(train_ids)], df[df["sequence_id"].isin(val_ids)]
    scaler = StandardScaler().fit(train_df[imu_cols + tof_cols].fillna(0))
    joblib.dump(scaler, os.path.join(fold_dir, "scaler.pkl"))

    def prepare_data(ids, df):
        X_imu, X_tof, y = [], [], []
        for sid in ids:
            g = df[df["sequence_id"]==sid]
            m = scaler.transform(g[imu_cols + tof_cols].ffill().bfill().fillna(0))
            imu_m, tof_m = m[:,:imu_dim], m[:,imu_dim:]
            imu_m = np.pad(imu_m, ((0,max(0,pad_len-len(imu_m))), (0,0)))
            tof_img = tof_m.reshape(-1,5,8,8)
            tof_img = np.pad(tof_img, ((0,max(0,pad_len-len(tof_img))),(0,0),(0,0),(0,0)))
            X_imu.append(imu_m[:pad_len])
            X_tof.append(tof_img[:pad_len,0,:,:])
            y.append(g["gesture_class"].iloc[0])
        return X_imu, X_tof, y

    X_imu_train, X_tof_train, y_train = prepare_data(train_ids, train_df)
    X_imu_val, X_tof_val, y_val = prepare_data(val_ids, val_df)

    train_loader = DataLoader(Img8x8Dataset(X_imu_train,X_tof_train,y_train),batch_size=BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(Img8x8Dataset(X_imu_val,X_tof_val,y_val),batch_size=BATCH_SIZE)

    model = TwoBranchImageSeqModel(imu_dim, len(le.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    best_f1, patience_counter = 0, 0
    for epoch in range(EPOCHS):
        model.train(); total_loss = 0
        for xb_imu, xb_tof, yb in train_loader:
            xb_imu, xb_tof, yb = xb_imu.to(device), xb_tof.to(device), yb.to(device)
            optimizer.zero_grad()
            logits,_ = model(xb_imu, xb_tof)
            loss = F.cross_entropy(logits, yb)
            loss.backward(); optimizer.step()
            total_loss += loss.item()*len(xb_imu)

        model.eval(); preds,trues = [],[]
        with torch.no_grad():
            for xb_imu, xb_tof, yb in val_loader:
                xb_imu, xb_tof = xb_imu.to(device), xb_tof.to(device)
                logits,_ = model(xb_imu, xb_tof)
                preds.extend(logits.argmax(1).cpu().numpy())
                trues.extend(yb.numpy())

        acc_18 = accuracy_score(trues, preds)
        f1_18 = f1_score(trues, preds, average="macro")
        acc_bin = accuracy_score(to_binary(trues), to_binary(preds))
        f1_bin = f1_score(to_binary(trues), to_binary(preds), average="macro")
        acc_9 = accuracy_score(to_9class(trues), to_9class(preds))
        f1_9 = f1_score(to_9class(trues), to_9class(preds), average="macro")
        f1_avg = (f1_bin + f1_9) / 2

        scheduler.step(1-f1_18)
        print(f"Epoch {epoch+1} | Loss:{total_loss/len(train_loader.dataset):.4f} | "
              f"Acc18:{acc_18:.4f}, F1_18:{f1_18:.4f} | "
              f"F1_bin:{f1_bin:.4f}, F1_9:{f1_9:.4f}, F1_avg:{f1_avg:.4f}")

        if f1_18 > best_f1:
            best_f1, patience_counter = f1_18, 0
            torch.save({
                "model_state_dict":model.state_dict(),
                "imu_dim":imu_dim,"pad_len":pad_len,"classes":le.classes_
            }, os.path.join(fold_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping."); break


r"""
=== Fold 1 ===
C:\dev\sensor_project\train_test_18class_28.py:61: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at ..\torch\csrc\utils\tensor_new.cpp:264.)
  self.imu_seqs = torch.tensor(imu_seqs, dtype=torch.float32)
Epoch 1 | Loss:1.9004 | Acc18:0.5533, F1_18:0.5422 | F1_bin:0.8723, F1_9:0.5726, F1_avg:0.7224
Epoch 2 | Loss:1.3481 | Acc18:0.6054, F1_18:0.6113 | F1_bin:0.8421, F1_9:0.6329, F1_avg:0.7375
Epoch 3 | Loss:1.1740 | Acc18:0.6311, F1_18:0.6386 | F1_bin:0.9017, F1_9:0.6602, F1_avg:0.7810
Epoch 4 | Loss:1.0544 | Acc18:0.6612, F1_18:0.6588 | F1_bin:0.9122, F1_9:0.6791, F1_avg:0.7956
Epoch 5 | Loss:0.9594 | Acc18:0.6520, F1_18:0.6683 | F1_bin:0.8976, F1_9:0.6945, F1_avg:0.7961
Epoch 6 | Loss:0.9100 | Acc18:0.6936, F1_18:0.7136 | F1_bin:0.9128, F1_9:0.7298, F1_avg:0.8213
Epoch 7 | Loss:0.8507 | Acc18:0.6893, F1_18:0.6825 | F1_bin:0.9182, F1_9:0.7117, F1_avg:0.8150
Epoch 8 | Loss:0.8088 | Acc18:0.6771, F1_18:0.6743 | F1_bin:0.9269, F1_9:0.6970, F1_avg:0.8119
Epoch 9 | Loss:0.7568 | Acc18:0.7083, F1_18:0.7154 | F1_bin:0.9239, F1_9:0.7340, F1_avg:0.8289
Epoch 10 | Loss:0.7195 | Acc18:0.6832, F1_18:0.6911 | F1_bin:0.9229, F1_9:0.7086, F1_avg:0.8158
Epoch 11 | Loss:0.6799 | Acc18:0.6814, F1_18:0.6869 | F1_bin:0.9233, F1_9:0.7087, F1_avg:0.8160
Epoch 12 | Loss:0.6787 | Acc18:0.7083, F1_18:0.7033 | F1_bin:0.9362, F1_9:0.7259, F1_avg:0.8310
Epoch 00013: reducing learning rate of group 0 to 5.0000e-04.
Epoch 13 | Loss:0.6464 | Acc18:0.6808, F1_18:0.6897 | F1_bin:0.9124, F1_9:0.7059, F1_avg:0.8091
Epoch 14 | Loss:0.5413 | Acc18:0.7292, F1_18:0.7310 | F1_bin:0.9385, F1_9:0.7489, F1_avg:0.8437
Epoch 15 | Loss:0.4972 | Acc18:0.7402, F1_18:0.7395 | F1_bin:0.9422, F1_9:0.7576, F1_avg:0.8499
Epoch 16 | Loss:0.4713 | Acc18:0.7138, F1_18:0.7152 | F1_bin:0.9282, F1_9:0.7300, F1_avg:0.8291
Epoch 17 | Loss:0.4478 | Acc18:0.7181, F1_18:0.7209 | F1_bin:0.9319, F1_9:0.7378, F1_avg:0.8349
Epoch 18 | Loss:0.4336 | Acc18:0.7249, F1_18:0.7292 | F1_bin:0.9313, F1_9:0.7421, F1_avg:0.8367
Epoch 00019: reducing learning rate of group 0 to 2.5000e-04.
Epoch 19 | Loss:0.4109 | Acc18:0.7279, F1_18:0.7346 | F1_bin:0.9411, F1_9:0.7450, F1_avg:0.8430
Epoch 20 | Loss:0.3631 | Acc18:0.7341, F1_18:0.7415 | F1_bin:0.9368, F1_9:0.7546, F1_avg:0.8457
Epoch 21 | Loss:0.3402 | Acc18:0.7316, F1_18:0.7409 | F1_bin:0.9361, F1_9:0.7560, F1_avg:0.8461
Epoch 22 | Loss:0.3280 | Acc18:0.7328, F1_18:0.7441 | F1_bin:0.9356, F1_9:0.7564, F1_avg:0.8460
Epoch 23 | Loss:0.3040 | Acc18:0.7328, F1_18:0.7446 | F1_bin:0.9380, F1_9:0.7577, F1_avg:0.8478
Epoch 24 | Loss:0.2933 | Acc18:0.7249, F1_18:0.7302 | F1_bin:0.9354, F1_9:0.7494, F1_avg:0.8424
Epoch 25 | Loss:0.2939 | Acc18:0.7384, F1_18:0.7443 | F1_bin:0.9392, F1_9:0.7586, F1_avg:0.8489
Epoch 26 | Loss:0.2773 | Acc18:0.7390, F1_18:0.7439 | F1_bin:0.9368, F1_9:0.7569, F1_avg:0.8468
Epoch 00027: reducing learning rate of group 0 to 1.2500e-04.
Epoch 27 | Loss:0.2725 | Acc18:0.7255, F1_18:0.7316 | F1_bin:0.9356, F1_9:0.7490, F1_avg:0.8423
Epoch 28 | Loss:0.2529 | Acc18:0.7390, F1_18:0.7491 | F1_bin:0.9349, F1_9:0.7625, F1_avg:0.8487
Epoch 29 | Loss:0.2396 | Acc18:0.7390, F1_18:0.7423 | F1_bin:0.9374, F1_9:0.7580, F1_avg:0.8477
Epoch 30 | Loss:0.2267 | Acc18:0.7365, F1_18:0.7412 | F1_bin:0.9368, F1_9:0.7561, F1_avg:0.8464
Epoch 31 | Loss:0.2170 | Acc18:0.7279, F1_18:0.7325 | F1_bin:0.9356, F1_9:0.7453, F1_avg:0.8405
Epoch 00032: reducing learning rate of group 0 to 6.2500e-05.
Epoch 32 | Loss:0.2144 | Acc18:0.7292, F1_18:0.7369 | F1_bin:0.9349, F1_9:0.7502, F1_avg:0.8425
Epoch 33 | Loss:0.1986 | Acc18:0.7396, F1_18:0.7496 | F1_bin:0.9362, F1_9:0.7611, F1_avg:0.8487
Epoch 34 | Loss:0.2016 | Acc18:0.7371, F1_18:0.7427 | F1_bin:0.9368, F1_9:0.7543, F1_avg:0.8455
Epoch 35 | Loss:0.1965 | Acc18:0.7347, F1_18:0.7400 | F1_bin:0.9361, F1_9:0.7541, F1_avg:0.8451
Epoch 36 | Loss:0.1949 | Acc18:0.7377, F1_18:0.7468 | F1_bin:0.9374, F1_9:0.7588, F1_avg:0.8481
Epoch 00037: reducing learning rate of group 0 to 3.1250e-05.
Epoch 37 | Loss:0.1894 | Acc18:0.7347, F1_18:0.7454 | F1_bin:0.9355, F1_9:0.7578, F1_avg:0.8467
Epoch 38 | Loss:0.1818 | Acc18:0.7347, F1_18:0.7396 | F1_bin:0.9362, F1_9:0.7534, F1_avg:0.8448
Epoch 39 | Loss:0.1720 | Acc18:0.7396, F1_18:0.7431 | F1_bin:0.9411, F1_9:0.7554, F1_avg:0.8482
Epoch 40 | Loss:0.1793 | Acc18:0.7365, F1_18:0.7440 | F1_bin:0.9368, F1_9:0.7575, F1_avg:0.8471
Epoch 00041: reducing learning rate of group 0 to 1.5625e-05.
Epoch 41 | Loss:0.1719 | Acc18:0.7322, F1_18:0.7370 | F1_bin:0.9343, F1_9:0.7512, F1_avg:0.8428
Epoch 42 | Loss:0.1697 | Acc18:0.7322, F1_18:0.7404 | F1_bin:0.9349, F1_9:0.7529, F1_avg:0.8439
Epoch 43 | Loss:0.1648 | Acc18:0.7365, F1_18:0.7447 | F1_bin:0.9374, F1_9:0.7586, F1_avg:0.8480
Epoch 44 | Loss:0.1734 | Acc18:0.7414, F1_18:0.7502 | F1_bin:0.9355, F1_9:0.7631, F1_avg:0.8493
Epoch 45 | Loss:0.1726 | Acc18:0.7420, F1_18:0.7491 | F1_bin:0.9374, F1_9:0.7611, F1_avg:0.8493
Epoch 46 | Loss:0.1693 | Acc18:0.7390, F1_18:0.7438 | F1_bin:0.9374, F1_9:0.7570, F1_avg:0.8472
Epoch 47 | Loss:0.1680 | Acc18:0.7371, F1_18:0.7442 | F1_bin:0.9349, F1_9:0.7563, F1_avg:0.8456
Epoch 00048: reducing learning rate of group 0 to 7.8125e-06.
Epoch 48 | Loss:0.1649 | Acc18:0.7384, F1_18:0.7469 | F1_bin:0.9386, F1_9:0.7605, F1_avg:0.8495
Epoch 49 | Loss:0.1659 | Acc18:0.7408, F1_18:0.7463 | F1_bin:0.9386, F1_9:0.7572, F1_avg:0.8479
Epoch 50 | Loss:0.1713 | Acc18:0.7353, F1_18:0.7416 | F1_bin:0.9362, F1_9:0.7548, F1_avg:0.8455
Epoch 51 | Loss:0.1634 | Acc18:0.7359, F1_18:0.7427 | F1_bin:0.9350, F1_9:0.7550, F1_avg:0.8450
Epoch 00052: reducing learning rate of group 0 to 3.9063e-06.
Epoch 52 | Loss:0.1665 | Acc18:0.7335, F1_18:0.7358 | F1_bin:0.9350, F1_9:0.7491, F1_avg:0.8420
Epoch 53 | Loss:0.1658 | Acc18:0.7408, F1_18:0.7498 | F1_bin:0.9386, F1_9:0.7600, F1_avg:0.8493
Epoch 54 | Loss:0.1618 | Acc18:0.7396, F1_18:0.7484 | F1_bin:0.9392, F1_9:0.7591, F1_avg:0.8491
Epoch 55 | Loss:0.1569 | Acc18:0.7377, F1_18:0.7429 | F1_bin:0.9356, F1_9:0.7550, F1_avg:0.8453
Epoch 00056: reducing learning rate of group 0 to 1.9531e-06.
Epoch 56 | Loss:0.1647 | Acc18:0.7353, F1_18:0.7415 | F1_bin:0.9362, F1_9:0.7538, F1_avg:0.8450
Epoch 57 | Loss:0.1599 | Acc18:0.7408, F1_18:0.7459 | F1_bin:0.9392, F1_9:0.7582, F1_avg:0.8487
Epoch 58 | Loss:0.1606 | Acc18:0.7451, F1_18:0.7539 | F1_bin:0.9386, F1_9:0.7649, F1_avg:0.8517
Epoch 59 | Loss:0.1544 | Acc18:0.7408, F1_18:0.7455 | F1_bin:0.9362, F1_9:0.7577, F1_avg:0.8469
Epoch 60 | Loss:0.1598 | Acc18:0.7420, F1_18:0.7501 | F1_bin:0.9374, F1_9:0.7604, F1_avg:0.8489
Epoch 61 | Loss:0.1638 | Acc18:0.7353, F1_18:0.7412 | F1_bin:0.9362, F1_9:0.7549, F1_avg:0.8455
Epoch 00062: reducing learning rate of group 0 to 9.7656e-07.
Epoch 62 | Loss:0.1581 | Acc18:0.7371, F1_18:0.7442 | F1_bin:0.9343, F1_9:0.7582, F1_avg:0.8463
Epoch 63 | Loss:0.1613 | Acc18:0.7426, F1_18:0.7493 | F1_bin:0.9356, F1_9:0.7604, F1_avg:0.8480
Epoch 64 | Loss:0.1618 | Acc18:0.7420, F1_18:0.7480 | F1_bin:0.9361, F1_9:0.7601, F1_avg:0.8481
Epoch 65 | Loss:0.1595 | Acc18:0.7328, F1_18:0.7387 | F1_bin:0.9337, F1_9:0.7519, F1_avg:0.8428
Epoch 00066: reducing learning rate of group 0 to 4.8828e-07.
Epoch 66 | Loss:0.1558 | Acc18:0.7396, F1_18:0.7485 | F1_bin:0.9392, F1_9:0.7599, F1_avg:0.8496
Epoch 67 | Loss:0.1567 | Acc18:0.7377, F1_18:0.7458 | F1_bin:0.9368, F1_9:0.7594, F1_avg:0.8481
Epoch 68 | Loss:0.1638 | Acc18:0.7445, F1_18:0.7539 | F1_bin:0.9368, F1_9:0.7635, F1_avg:0.8501
Epoch 69 | Loss:0.1586 | Acc18:0.7408, F1_18:0.7502 | F1_bin:0.9374, F1_9:0.7619, F1_avg:0.8497
Epoch 00070: reducing learning rate of group 0 to 2.4414e-07.
Epoch 70 | Loss:0.1638 | Acc18:0.7384, F1_18:0.7468 | F1_bin:0.9356, F1_9:0.7583, F1_avg:0.8469
Epoch 71 | Loss:0.1672 | Acc18:0.7414, F1_18:0.7496 | F1_bin:0.9331, F1_9:0.7628, F1_avg:0.8479
Epoch 72 | Loss:0.1582 | Acc18:0.7341, F1_18:0.7418 | F1_bin:0.9331, F1_9:0.7541, F1_avg:0.8436
Epoch 73 | Loss:0.1633 | Acc18:0.7390, F1_18:0.7500 | F1_bin:0.9337, F1_9:0.7611, F1_avg:0.8474
Epoch 00074: reducing learning rate of group 0 to 1.2207e-07.
Epoch 74 | Loss:0.1637 | Acc18:0.7420, F1_18:0.7476 | F1_bin:0.9386, F1_9:0.7595, F1_avg:0.8491
Epoch 75 | Loss:0.1586 | Acc18:0.7408, F1_18:0.7474 | F1_bin:0.9380, F1_9:0.7599, F1_avg:0.8489
Epoch 76 | Loss:0.1554 | Acc18:0.7384, F1_18:0.7432 | F1_bin:0.9368, F1_9:0.7560, F1_avg:0.8464
Epoch 77 | Loss:0.1584 | Acc18:0.7426, F1_18:0.7493 | F1_bin:0.9398, F1_9:0.7604, F1_avg:0.8501
Epoch 00078: reducing learning rate of group 0 to 6.1035e-08.
Epoch 78 | Loss:0.1604 | Acc18:0.7365, F1_18:0.7439 | F1_bin:0.9337, F1_9:0.7561, F1_avg:0.8449
Early stopping.

=== Fold 2 ===
Epoch 1 | Loss:1.8390 | Acc18:0.4531, F1_18:0.4372 | F1_bin:0.8056, F1_9:0.4855, F1_avg:0.6456
Epoch 2 | Loss:1.2920 | Acc18:0.5218, F1_18:0.5169 | F1_bin:0.8370, F1_9:0.5696, F1_avg:0.7033
Epoch 3 | Loss:1.0864 | Acc18:0.5340, F1_18:0.5357 | F1_bin:0.8314, F1_9:0.5923, F1_avg:0.7118
Epoch 4 | Loss:0.9847 | Acc18:0.5573, F1_18:0.5541 | F1_bin:0.8473, F1_9:0.6001, F1_avg:0.7237
Epoch 5 | Loss:0.9215 | Acc18:0.5469, F1_18:0.5623 | F1_bin:0.8396, F1_9:0.5871, F1_avg:0.7133
Epoch 6 | Loss:0.8515 | Acc18:0.5549, F1_18:0.5583 | F1_bin:0.8577, F1_9:0.6114, F1_avg:0.7346
Epoch 7 | Loss:0.8008 | Acc18:0.5684, F1_18:0.5869 | F1_bin:0.8759, F1_9:0.6267, F1_avg:0.7513
Epoch 8 | Loss:0.7580 | Acc18:0.5776, F1_18:0.5829 | F1_bin:0.8479, F1_9:0.6230, F1_avg:0.7355
Epoch 9 | Loss:0.7111 | Acc18:0.5763, F1_18:0.5921 | F1_bin:0.8656, F1_9:0.6312, F1_avg:0.7484
Epoch 10 | Loss:0.6989 | Acc18:0.5788, F1_18:0.5942 | F1_bin:0.8556, F1_9:0.6213, F1_avg:0.7384
Epoch 11 | Loss:0.6503 | Acc18:0.6174, F1_18:0.6421 | F1_bin:0.8832, F1_9:0.6680, F1_avg:0.7756
Epoch 12 | Loss:0.6148 | Acc18:0.6131, F1_18:0.6441 | F1_bin:0.8765, F1_9:0.6755, F1_avg:0.7760
Epoch 13 | Loss:0.6057 | Acc18:0.6150, F1_18:0.6507 | F1_bin:0.8763, F1_9:0.6774, F1_avg:0.7769
Epoch 14 | Loss:0.5696 | Acc18:0.6052, F1_18:0.6358 | F1_bin:0.8745, F1_9:0.6601, F1_avg:0.7673
Epoch 15 | Loss:0.5276 | Acc18:0.6015, F1_18:0.6263 | F1_bin:0.8614, F1_9:0.6581, F1_avg:0.7598
Epoch 16 | Loss:0.5293 | Acc18:0.6094, F1_18:0.6298 | F1_bin:0.8522, F1_9:0.6635, F1_avg:0.7579
Epoch 00017: reducing learning rate of group 0 to 5.0000e-04.
Epoch 17 | Loss:0.5040 | Acc18:0.6021, F1_18:0.6320 | F1_bin:0.8792, F1_9:0.6573, F1_avg:0.7683
Epoch 18 | Loss:0.4098 | Acc18:0.6297, F1_18:0.6553 | F1_bin:0.8760, F1_9:0.6818, F1_avg:0.7789
Epoch 19 | Loss:0.3661 | Acc18:0.6229, F1_18:0.6439 | F1_bin:0.8701, F1_9:0.6776, F1_avg:0.7739
Epoch 20 | Loss:0.3440 | Acc18:0.6278, F1_18:0.6603 | F1_bin:0.8667, F1_9:0.6863, F1_avg:0.7765
Epoch 21 | Loss:0.3280 | Acc18:0.6131, F1_18:0.6488 | F1_bin:0.8714, F1_9:0.6753, F1_avg:0.7734
Epoch 22 | Loss:0.3353 | Acc18:0.6321, F1_18:0.6667 | F1_bin:0.8782, F1_9:0.6880, F1_avg:0.7831
Epoch 23 | Loss:0.3026 | Acc18:0.6223, F1_18:0.6496 | F1_bin:0.8755, F1_9:0.6778, F1_avg:0.7766
Epoch 24 | Loss:0.2927 | Acc18:0.6223, F1_18:0.6442 | F1_bin:0.8717, F1_9:0.6751, F1_avg:0.7734
Epoch 25 | Loss:0.2779 | Acc18:0.6235, F1_18:0.6424 | F1_bin:0.8779, F1_9:0.6716, F1_avg:0.7747
Epoch 00026: reducing learning rate of group 0 to 2.5000e-04.
Epoch 26 | Loss:0.2539 | Acc18:0.6113, F1_18:0.6413 | F1_bin:0.8716, F1_9:0.6664, F1_avg:0.7690
Epoch 27 | Loss:0.2228 | Acc18:0.6272, F1_18:0.6607 | F1_bin:0.8741, F1_9:0.6864, F1_avg:0.7802
Epoch 28 | Loss:0.2046 | Acc18:0.6254, F1_18:0.6544 | F1_bin:0.8699, F1_9:0.6779, F1_avg:0.7739
Epoch 29 | Loss:0.1805 | Acc18:0.6297, F1_18:0.6704 | F1_bin:0.8743, F1_9:0.6943, F1_avg:0.7843
Epoch 30 | Loss:0.1821 | Acc18:0.6334, F1_18:0.6657 | F1_bin:0.8789, F1_9:0.6897, F1_avg:0.7843
Epoch 31 | Loss:0.1771 | Acc18:0.6272, F1_18:0.6505 | F1_bin:0.8730, F1_9:0.6801, F1_avg:0.7766
Epoch 32 | Loss:0.1684 | Acc18:0.6315, F1_18:0.6590 | F1_bin:0.8741, F1_9:0.6891, F1_avg:0.7816
Epoch 00033: reducing learning rate of group 0 to 1.2500e-04.
Epoch 33 | Loss:0.1595 | Acc18:0.6309, F1_18:0.6651 | F1_bin:0.8639, F1_9:0.6959, F1_avg:0.7799
Epoch 34 | Loss:0.1446 | Acc18:0.6272, F1_18:0.6518 | F1_bin:0.8736, F1_9:0.6805, F1_avg:0.7771
Epoch 35 | Loss:0.1282 | Acc18:0.6401, F1_18:0.6618 | F1_bin:0.8767, F1_9:0.6916, F1_avg:0.7842
Epoch 36 | Loss:0.1253 | Acc18:0.6352, F1_18:0.6684 | F1_bin:0.8729, F1_9:0.6959, F1_avg:0.7844
Epoch 00037: reducing learning rate of group 0 to 6.2500e-05.
Epoch 37 | Loss:0.1303 | Acc18:0.6272, F1_18:0.6540 | F1_bin:0.8778, F1_9:0.6813, F1_avg:0.7795
Epoch 38 | Loss:0.1129 | Acc18:0.6217, F1_18:0.6503 | F1_bin:0.8679, F1_9:0.6819, F1_avg:0.7749
Epoch 39 | Loss:0.1004 | Acc18:0.6309, F1_18:0.6574 | F1_bin:0.8693, F1_9:0.6900, F1_avg:0.7797
Epoch 40 | Loss:0.1080 | Acc18:0.6315, F1_18:0.6557 | F1_bin:0.8718, F1_9:0.6871, F1_avg:0.7794
Epoch 00041: reducing learning rate of group 0 to 3.1250e-05.
Epoch 41 | Loss:0.1071 | Acc18:0.6334, F1_18:0.6633 | F1_bin:0.8736, F1_9:0.6909, F1_avg:0.7823
Epoch 42 | Loss:0.1005 | Acc18:0.6340, F1_18:0.6627 | F1_bin:0.8724, F1_9:0.6913, F1_avg:0.7818
Epoch 43 | Loss:0.1032 | Acc18:0.6334, F1_18:0.6645 | F1_bin:0.8748, F1_9:0.6923, F1_avg:0.7836
Epoch 44 | Loss:0.0977 | Acc18:0.6395, F1_18:0.6678 | F1_bin:0.8730, F1_9:0.6976, F1_avg:0.7853
Epoch 00045: reducing learning rate of group 0 to 1.5625e-05.
Epoch 45 | Loss:0.0982 | Acc18:0.6346, F1_18:0.6650 | F1_bin:0.8730, F1_9:0.6930, F1_avg:0.7830
Epoch 46 | Loss:0.0979 | Acc18:0.6334, F1_18:0.6622 | F1_bin:0.8736, F1_9:0.6912, F1_avg:0.7824
Epoch 47 | Loss:0.0941 | Acc18:0.6358, F1_18:0.6649 | F1_bin:0.8711, F1_9:0.6942, F1_avg:0.7827
Epoch 48 | Loss:0.0959 | Acc18:0.6334, F1_18:0.6611 | F1_bin:0.8741, F1_9:0.6911, F1_avg:0.7826
Epoch 00049: reducing learning rate of group 0 to 7.8125e-06.
Epoch 49 | Loss:0.0934 | Acc18:0.6352, F1_18:0.6565 | F1_bin:0.8711, F1_9:0.6899, F1_avg:0.7805
Early stopping.

=== Fold 3 ===
Epoch 1 | Loss:1.8857 | Acc18:0.4948, F1_18:0.4943 | F1_bin:0.8412, F1_9:0.5313, F1_avg:0.6862
Epoch 2 | Loss:1.3163 | Acc18:0.5366, F1_18:0.5363 | F1_bin:0.8532, F1_9:0.5614, F1_avg:0.7073
Epoch 3 | Loss:1.1505 | Acc18:0.6202, F1_18:0.6127 | F1_bin:0.8867, F1_9:0.6547, F1_avg:0.7707
Epoch 4 | Loss:1.0400 | Acc18:0.5833, F1_18:0.6059 | F1_bin:0.8680, F1_9:0.6318, F1_avg:0.7499
Epoch 5 | Loss:0.9619 | Acc18:0.6263, F1_18:0.6429 | F1_bin:0.8800, F1_9:0.6741, F1_avg:0.7771
Epoch 6 | Loss:0.9064 | Acc18:0.6466, F1_18:0.6606 | F1_bin:0.8869, F1_9:0.6753, F1_avg:0.7811
Epoch 7 | Loss:0.8488 | Acc18:0.6331, F1_18:0.6385 | F1_bin:0.8898, F1_9:0.6636, F1_avg:0.7767
Epoch 8 | Loss:0.7976 | Acc18:0.6398, F1_18:0.6497 | F1_bin:0.9132, F1_9:0.6721, F1_avg:0.7926
Epoch 9 | Loss:0.7567 | Acc18:0.6497, F1_18:0.6616 | F1_bin:0.9002, F1_9:0.6805, F1_avg:0.7904
Epoch 10 | Loss:0.7398 | Acc18:0.5956, F1_18:0.5865 | F1_bin:0.8749, F1_9:0.6223, F1_avg:0.7486
Epoch 11 | Loss:0.7062 | Acc18:0.6804, F1_18:0.6695 | F1_bin:0.9100, F1_9:0.6931, F1_avg:0.8016
Epoch 12 | Loss:0.6762 | Acc18:0.6847, F1_18:0.6872 | F1_bin:0.9085, F1_9:0.7185, F1_avg:0.8135
Epoch 13 | Loss:0.6373 | Acc18:0.6724, F1_18:0.6682 | F1_bin:0.8997, F1_9:0.6945, F1_avg:0.7971
Epoch 14 | Loss:0.6167 | Acc18:0.6761, F1_18:0.6856 | F1_bin:0.9095, F1_9:0.7098, F1_avg:0.8097
Epoch 15 | Loss:0.5903 | Acc18:0.6958, F1_18:0.7116 | F1_bin:0.9092, F1_9:0.7298, F1_avg:0.8195
Epoch 16 | Loss:0.5586 | Acc18:0.7007, F1_18:0.7120 | F1_bin:0.9161, F1_9:0.7308, F1_avg:0.8234
Epoch 17 | Loss:0.5262 | Acc18:0.6902, F1_18:0.7019 | F1_bin:0.9142, F1_9:0.7228, F1_avg:0.8185
Epoch 18 | Loss:0.5131 | Acc18:0.6792, F1_18:0.6840 | F1_bin:0.9058, F1_9:0.7116, F1_avg:0.8087
Epoch 19 | Loss:0.4987 | Acc18:0.6958, F1_18:0.6958 | F1_bin:0.9038, F1_9:0.7189, F1_avg:0.8113
Epoch 00020: reducing learning rate of group 0 to 5.0000e-04.
Epoch 20 | Loss:0.4550 | Acc18:0.6890, F1_18:0.7070 | F1_bin:0.9181, F1_9:0.7262, F1_avg:0.8221
Epoch 21 | Loss:0.3757 | Acc18:0.7197, F1_18:0.7333 | F1_bin:0.9190, F1_9:0.7522, F1_avg:0.8356
Epoch 22 | Loss:0.3373 | Acc18:0.7160, F1_18:0.7264 | F1_bin:0.9184, F1_9:0.7496, F1_avg:0.8340
Epoch 23 | Loss:0.3110 | Acc18:0.7136, F1_18:0.7219 | F1_bin:0.9172, F1_9:0.7413, F1_avg:0.8292
Epoch 24 | Loss:0.2928 | Acc18:0.7111, F1_18:0.7268 | F1_bin:0.9259, F1_9:0.7417, F1_avg:0.8338
Epoch 00025: reducing learning rate of group 0 to 2.5000e-04.
Epoch 25 | Loss:0.2797 | Acc18:0.7179, F1_18:0.7256 | F1_bin:0.9228, F1_9:0.7454, F1_avg:0.8341
Epoch 26 | Loss:0.2532 | Acc18:0.7289, F1_18:0.7384 | F1_bin:0.9229, F1_9:0.7567, F1_avg:0.8398
Epoch 27 | Loss:0.2277 | Acc18:0.7351, F1_18:0.7490 | F1_bin:0.9235, F1_9:0.7689, F1_avg:0.8462
Epoch 28 | Loss:0.2200 | Acc18:0.7271, F1_18:0.7369 | F1_bin:0.9272, F1_9:0.7572, F1_avg:0.8422
Epoch 29 | Loss:0.2017 | Acc18:0.7265, F1_18:0.7302 | F1_bin:0.9240, F1_9:0.7505, F1_avg:0.8373
Epoch 30 | Loss:0.1886 | Acc18:0.7216, F1_18:0.7278 | F1_bin:0.9247, F1_9:0.7520, F1_avg:0.8383
Epoch 00031: reducing learning rate of group 0 to 1.2500e-04.
Epoch 31 | Loss:0.1841 | Acc18:0.7160, F1_18:0.7235 | F1_bin:0.9155, F1_9:0.7419, F1_avg:0.8287
Epoch 32 | Loss:0.1700 | Acc18:0.7283, F1_18:0.7426 | F1_bin:0.9235, F1_9:0.7614, F1_avg:0.8424
Epoch 33 | Loss:0.1536 | Acc18:0.7277, F1_18:0.7333 | F1_bin:0.9235, F1_9:0.7535, F1_avg:0.8385
Epoch 34 | Loss:0.1531 | Acc18:0.7369, F1_18:0.7464 | F1_bin:0.9242, F1_9:0.7651, F1_avg:0.8446
Epoch 00035: reducing learning rate of group 0 to 6.2500e-05.
Epoch 35 | Loss:0.1457 | Acc18:0.7283, F1_18:0.7415 | F1_bin:0.9254, F1_9:0.7583, F1_avg:0.8419
Epoch 36 | Loss:0.1387 | Acc18:0.7271, F1_18:0.7383 | F1_bin:0.9204, F1_9:0.7586, F1_avg:0.8395
Epoch 37 | Loss:0.1292 | Acc18:0.7314, F1_18:0.7392 | F1_bin:0.9247, F1_9:0.7605, F1_avg:0.8426
Epoch 38 | Loss:0.1317 | Acc18:0.7357, F1_18:0.7469 | F1_bin:0.9241, F1_9:0.7648, F1_avg:0.8445
Epoch 00039: reducing learning rate of group 0 to 3.1250e-05.
Epoch 39 | Loss:0.1277 | Acc18:0.7302, F1_18:0.7422 | F1_bin:0.9223, F1_9:0.7638, F1_avg:0.8430
Epoch 40 | Loss:0.1200 | Acc18:0.7345, F1_18:0.7453 | F1_bin:0.9247, F1_9:0.7644, F1_avg:0.8446
Epoch 41 | Loss:0.1183 | Acc18:0.7326, F1_18:0.7437 | F1_bin:0.9210, F1_9:0.7634, F1_avg:0.8422
Epoch 42 | Loss:0.1189 | Acc18:0.7320, F1_18:0.7430 | F1_bin:0.9229, F1_9:0.7628, F1_avg:0.8428
Epoch 00043: reducing learning rate of group 0 to 1.5625e-05.
Epoch 43 | Loss:0.1200 | Acc18:0.7296, F1_18:0.7432 | F1_bin:0.9223, F1_9:0.7601, F1_avg:0.8412
Epoch 44 | Loss:0.1174 | Acc18:0.7345, F1_18:0.7472 | F1_bin:0.9235, F1_9:0.7664, F1_avg:0.8450
Epoch 45 | Loss:0.1097 | Acc18:0.7314, F1_18:0.7432 | F1_bin:0.9216, F1_9:0.7609, F1_avg:0.8412
Epoch 46 | Loss:0.1149 | Acc18:0.7351, F1_18:0.7442 | F1_bin:0.9241, F1_9:0.7640, F1_avg:0.8440
Epoch 00047: reducing learning rate of group 0 to 7.8125e-06.
Epoch 47 | Loss:0.1137 | Acc18:0.7320, F1_18:0.7471 | F1_bin:0.9198, F1_9:0.7669, F1_avg:0.8433
Early stopping.

=== Fold 4 ===
Epoch 1 | Loss:1.8938 | Acc18:0.5025, F1_18:0.5045 | F1_bin:0.8537, F1_9:0.5289, F1_avg:0.6913
Epoch 2 | Loss:1.3317 | Acc18:0.5670, F1_18:0.5583 | F1_bin:0.8684, F1_9:0.5869, F1_avg:0.7276
Epoch 3 | Loss:1.1254 | Acc18:0.6181, F1_18:0.5965 | F1_bin:0.8837, F1_9:0.6236, F1_avg:0.7537
Epoch 4 | Loss:1.0182 | Acc18:0.6187, F1_18:0.6131 | F1_bin:0.8621, F1_9:0.6425, F1_avg:0.7523
Epoch 5 | Loss:0.9291 | Acc18:0.6298, F1_18:0.6128 | F1_bin:0.9009, F1_9:0.6451, F1_avg:0.7730
Epoch 6 | Loss:0.8777 | Acc18:0.5990, F1_18:0.6045 | F1_bin:0.8625, F1_9:0.6183, F1_avg:0.7404
Epoch 7 | Loss:0.8291 | Acc18:0.6421, F1_18:0.6309 | F1_bin:0.9009, F1_9:0.6575, F1_avg:0.7792
Epoch 8 | Loss:0.7841 | Acc18:0.6199, F1_18:0.6254 | F1_bin:0.8899, F1_9:0.6489, F1_avg:0.7694
Epoch 9 | Loss:0.7437 | Acc18:0.6624, F1_18:0.6565 | F1_bin:0.8885, F1_9:0.6803, F1_avg:0.7844
Epoch 10 | Loss:0.7074 | Acc18:0.6747, F1_18:0.6919 | F1_bin:0.8956, F1_9:0.7075, F1_avg:0.8016
Epoch 11 | Loss:0.6925 | Acc18:0.6900, F1_18:0.7113 | F1_bin:0.9149, F1_9:0.7218, F1_avg:0.8183
Epoch 12 | Loss:0.6546 | Acc18:0.6716, F1_18:0.6973 | F1_bin:0.8997, F1_9:0.7109, F1_avg:0.8053
Epoch 13 | Loss:0.6179 | Acc18:0.6642, F1_18:0.6818 | F1_bin:0.8972, F1_9:0.6997, F1_avg:0.7985
Epoch 14 | Loss:0.6043 | Acc18:0.6654, F1_18:0.6750 | F1_bin:0.9198, F1_9:0.6916, F1_avg:0.8057
Epoch 00015: reducing learning rate of group 0 to 5.0000e-04.
Epoch 15 | Loss:0.5787 | Acc18:0.6968, F1_18:0.7105 | F1_bin:0.9162, F1_9:0.7225, F1_avg:0.8194
Epoch 16 | Loss:0.4920 | Acc18:0.6974, F1_18:0.7155 | F1_bin:0.9124, F1_9:0.7228, F1_avg:0.8176
Epoch 17 | Loss:0.4420 | Acc18:0.7073, F1_18:0.7275 | F1_bin:0.9119, F1_9:0.7375, F1_avg:0.8247
Epoch 18 | Loss:0.4356 | Acc18:0.6894, F1_18:0.7036 | F1_bin:0.9074, F1_9:0.7248, F1_avg:0.8161
Epoch 19 | Loss:0.4041 | Acc18:0.6888, F1_18:0.7183 | F1_bin:0.9053, F1_9:0.7305, F1_avg:0.8179
Epoch 20 | Loss:0.3920 | Acc18:0.7073, F1_18:0.7225 | F1_bin:0.9174, F1_9:0.7359, F1_avg:0.8266
Epoch 00021: reducing learning rate of group 0 to 2.5000e-04.
Epoch 21 | Loss:0.3805 | Acc18:0.6962, F1_18:0.7219 | F1_bin:0.9138, F1_9:0.7344, F1_avg:0.8241
Epoch 22 | Loss:0.3211 | Acc18:0.7066, F1_18:0.7252 | F1_bin:0.9117, F1_9:0.7387, F1_avg:0.8252
Epoch 23 | Loss:0.3067 | Acc18:0.7079, F1_18:0.7223 | F1_bin:0.9193, F1_9:0.7334, F1_avg:0.8264
Epoch 24 | Loss:0.2866 | Acc18:0.6993, F1_18:0.7123 | F1_bin:0.9135, F1_9:0.7264, F1_avg:0.8199
Epoch 00025: reducing learning rate of group 0 to 1.2500e-04.
Epoch 25 | Loss:0.2728 | Acc18:0.7060, F1_18:0.7257 | F1_bin:0.9161, F1_9:0.7387, F1_avg:0.8274
Epoch 26 | Loss:0.2558 | Acc18:0.7116, F1_18:0.7286 | F1_bin:0.9157, F1_9:0.7381, F1_avg:0.8269
Epoch 27 | Loss:0.2387 | Acc18:0.7073, F1_18:0.7189 | F1_bin:0.9125, F1_9:0.7339, F1_avg:0.8232
Epoch 28 | Loss:0.2345 | Acc18:0.7085, F1_18:0.7291 | F1_bin:0.9101, F1_9:0.7384, F1_avg:0.8242
Epoch 29 | Loss:0.2205 | Acc18:0.7177, F1_18:0.7364 | F1_bin:0.9126, F1_9:0.7504, F1_avg:0.8315
Epoch 30 | Loss:0.2197 | Acc18:0.7091, F1_18:0.7287 | F1_bin:0.9149, F1_9:0.7405, F1_avg:0.8277
Epoch 31 | Loss:0.2199 | Acc18:0.7183, F1_18:0.7390 | F1_bin:0.9149, F1_9:0.7496, F1_avg:0.8323
Epoch 32 | Loss:0.2095 | Acc18:0.7085, F1_18:0.7273 | F1_bin:0.9137, F1_9:0.7395, F1_avg:0.8266
Epoch 33 | Loss:0.2054 | Acc18:0.7146, F1_18:0.7299 | F1_bin:0.9186, F1_9:0.7430, F1_avg:0.8308
Epoch 34 | Loss:0.2009 | Acc18:0.7183, F1_18:0.7385 | F1_bin:0.9125, F1_9:0.7516, F1_avg:0.8321
Epoch 00035: reducing learning rate of group 0 to 6.2500e-05.
Epoch 35 | Loss:0.2013 | Acc18:0.7140, F1_18:0.7329 | F1_bin:0.9113, F1_9:0.7469, F1_avg:0.8291
Epoch 36 | Loss:0.1847 | Acc18:0.7159, F1_18:0.7332 | F1_bin:0.9174, F1_9:0.7460, F1_avg:0.8317
Epoch 37 | Loss:0.1808 | Acc18:0.7183, F1_18:0.7390 | F1_bin:0.9144, F1_9:0.7526, F1_avg:0.8335
Epoch 38 | Loss:0.1762 | Acc18:0.7146, F1_18:0.7319 | F1_bin:0.9155, F1_9:0.7419, F1_avg:0.8287
Epoch 00039: reducing learning rate of group 0 to 3.1250e-05.
Epoch 39 | Loss:0.1698 | Acc18:0.7165, F1_18:0.7360 | F1_bin:0.9137, F1_9:0.7509, F1_avg:0.8323
Epoch 40 | Loss:0.1669 | Acc18:0.7134, F1_18:0.7317 | F1_bin:0.9149, F1_9:0.7439, F1_avg:0.8294
Epoch 41 | Loss:0.1647 | Acc18:0.7128, F1_18:0.7327 | F1_bin:0.9113, F1_9:0.7451, F1_avg:0.8282
Epoch 42 | Loss:0.1583 | Acc18:0.7171, F1_18:0.7387 | F1_bin:0.9156, F1_9:0.7511, F1_avg:0.8333
Epoch 00043: reducing learning rate of group 0 to 1.5625e-05.
Epoch 43 | Loss:0.1695 | Acc18:0.7183, F1_18:0.7374 | F1_bin:0.9137, F1_9:0.7486, F1_avg:0.8311
Epoch 44 | Loss:0.1611 | Acc18:0.7232, F1_18:0.7426 | F1_bin:0.9150, F1_9:0.7553, F1_avg:0.8351
Epoch 45 | Loss:0.1602 | Acc18:0.7183, F1_18:0.7373 | F1_bin:0.9144, F1_9:0.7485, F1_avg:0.8315
Epoch 46 | Loss:0.1649 | Acc18:0.7159, F1_18:0.7341 | F1_bin:0.9137, F1_9:0.7462, F1_avg:0.8299
Epoch 47 | Loss:0.1537 | Acc18:0.7177, F1_18:0.7374 | F1_bin:0.9156, F1_9:0.7500, F1_avg:0.8328
Epoch 00048: reducing learning rate of group 0 to 7.8125e-06.
Epoch 48 | Loss:0.1542 | Acc18:0.7153, F1_18:0.7342 | F1_bin:0.9138, F1_9:0.7479, F1_avg:0.8308
Epoch 49 | Loss:0.1512 | Acc18:0.7214, F1_18:0.7405 | F1_bin:0.9163, F1_9:0.7529, F1_avg:0.8346
Epoch 50 | Loss:0.1500 | Acc18:0.7189, F1_18:0.7376 | F1_bin:0.9137, F1_9:0.7500, F1_avg:0.8318
Epoch 51 | Loss:0.1504 | Acc18:0.7165, F1_18:0.7354 | F1_bin:0.9131, F1_9:0.7478, F1_avg:0.8304
Epoch 00052: reducing learning rate of group 0 to 3.9063e-06.
Epoch 52 | Loss:0.1553 | Acc18:0.7196, F1_18:0.7411 | F1_bin:0.9150, F1_9:0.7532, F1_avg:0.8341
Epoch 53 | Loss:0.1538 | Acc18:0.7214, F1_18:0.7415 | F1_bin:0.9181, F1_9:0.7527, F1_avg:0.8354
Epoch 54 | Loss:0.1558 | Acc18:0.7177, F1_18:0.7379 | F1_bin:0.9125, F1_9:0.7512, F1_avg:0.8318
Epoch 55 | Loss:0.1523 | Acc18:0.7183, F1_18:0.7365 | F1_bin:0.9137, F1_9:0.7495, F1_avg:0.8316
Epoch 00056: reducing learning rate of group 0 to 1.9531e-06.
Epoch 56 | Loss:0.1447 | Acc18:0.7171, F1_18:0.7356 | F1_bin:0.9124, F1_9:0.7478, F1_avg:0.8301
Epoch 57 | Loss:0.1446 | Acc18:0.7146, F1_18:0.7341 | F1_bin:0.9137, F1_9:0.7485, F1_avg:0.8311
Epoch 58 | Loss:0.1569 | Acc18:0.7165, F1_18:0.7386 | F1_bin:0.9150, F1_9:0.7510, F1_avg:0.8330
Epoch 59 | Loss:0.1523 | Acc18:0.7159, F1_18:0.7331 | F1_bin:0.9143, F1_9:0.7462, F1_avg:0.8303
Epoch 00060: reducing learning rate of group 0 to 9.7656e-07.
Epoch 60 | Loss:0.1491 | Acc18:0.7196, F1_18:0.7370 | F1_bin:0.9119, F1_9:0.7501, F1_avg:0.8310
Epoch 61 | Loss:0.1493 | Acc18:0.7146, F1_18:0.7354 | F1_bin:0.9131, F1_9:0.7487, F1_avg:0.8309
Epoch 62 | Loss:0.1515 | Acc18:0.7189, F1_18:0.7350 | F1_bin:0.9131, F1_9:0.7490, F1_avg:0.8311
Epoch 63 | Loss:0.1546 | Acc18:0.7226, F1_18:0.7414 | F1_bin:0.9138, F1_9:0.7535, F1_avg:0.8336
Epoch 00064: reducing learning rate of group 0 to 4.8828e-07.
Epoch 64 | Loss:0.1587 | Acc18:0.7171, F1_18:0.7351 | F1_bin:0.9143, F1_9:0.7457, F1_avg:0.8300
Early stopping.

=== Fold 5 ===
Epoch 1 | Loss:1.8507 | Acc18:0.4691, F1_18:0.4793 | F1_bin:0.8188, F1_9:0.5010, F1_avg:0.6599
Epoch 2 | Loss:1.2815 | Acc18:0.5113, F1_18:0.5222 | F1_bin:0.8597, F1_9:0.5512, F1_avg:0.7054
Epoch 3 | Loss:1.1067 | Acc18:0.5370, F1_18:0.5551 | F1_bin:0.8526, F1_9:0.5839, F1_avg:0.7183
Epoch 4 | Loss:1.0136 | Acc18:0.5902, F1_18:0.6110 | F1_bin:0.8825, F1_9:0.6315, F1_avg:0.7570
Epoch 5 | Loss:0.9374 | Acc18:0.5853, F1_18:0.5944 | F1_bin:0.8746, F1_9:0.6189, F1_avg:0.7468
Epoch 6 | Loss:0.8546 | Acc18:0.6135, F1_18:0.6210 | F1_bin:0.9010, F1_9:0.6462, F1_avg:0.7736
Epoch 7 | Loss:0.8145 | Acc18:0.6263, F1_18:0.6361 | F1_bin:0.9049, F1_9:0.6572, F1_avg:0.7810
Epoch 8 | Loss:0.7801 | Acc18:0.6257, F1_18:0.6320 | F1_bin:0.9003, F1_9:0.6552, F1_avg:0.7777
Epoch 9 | Loss:0.7359 | Acc18:0.6220, F1_18:0.6272 | F1_bin:0.9131, F1_9:0.6448, F1_avg:0.7789
Epoch 10 | Loss:0.7080 | Acc18:0.6349, F1_18:0.6483 | F1_bin:0.9103, F1_9:0.6687, F1_avg:0.7895
Epoch 11 | Loss:0.6727 | Acc18:0.6514, F1_18:0.6619 | F1_bin:0.9148, F1_9:0.6821, F1_avg:0.7985
Epoch 12 | Loss:0.6382 | Acc18:0.6532, F1_18:0.6726 | F1_bin:0.9147, F1_9:0.6862, F1_avg:0.8005
Epoch 13 | Loss:0.6105 | Acc18:0.6287, F1_18:0.6319 | F1_bin:0.9000, F1_9:0.6568, F1_avg:0.7784
Epoch 14 | Loss:0.5964 | Acc18:0.6336, F1_18:0.6627 | F1_bin:0.9073, F1_9:0.6804, F1_avg:0.7939
Epoch 15 | Loss:0.5646 | Acc18:0.6538, F1_18:0.6523 | F1_bin:0.9184, F1_9:0.6785, F1_avg:0.7985
Epoch 00016: reducing learning rate of group 0 to 5.0000e-04.
Epoch 16 | Loss:0.5430 | Acc18:0.6508, F1_18:0.6541 | F1_bin:0.9137, F1_9:0.6792, F1_avg:0.7964
Epoch 17 | Loss:0.4487 | Acc18:0.6581, F1_18:0.6779 | F1_bin:0.9131, F1_9:0.6938, F1_avg:0.8035
Epoch 18 | Loss:0.3976 | Acc18:0.6679, F1_18:0.6858 | F1_bin:0.9179, F1_9:0.7027, F1_avg:0.8103
Epoch 19 | Loss:0.3813 | Acc18:0.6642, F1_18:0.6802 | F1_bin:0.9106, F1_9:0.6958, F1_avg:0.8032
Epoch 20 | Loss:0.3587 | Acc18:0.6636, F1_18:0.6815 | F1_bin:0.9161, F1_9:0.6943, F1_avg:0.8052
Epoch 21 | Loss:0.3653 | Acc18:0.6532, F1_18:0.6689 | F1_bin:0.9136, F1_9:0.6840, F1_avg:0.7988
Epoch 00022: reducing learning rate of group 0 to 2.5000e-04.
Epoch 22 | Loss:0.3307 | Acc18:0.6612, F1_18:0.6753 | F1_bin:0.9186, F1_9:0.6895, F1_avg:0.8041
Epoch 23 | Loss:0.2936 | Acc18:0.6826, F1_18:0.6990 | F1_bin:0.9222, F1_9:0.7111, F1_avg:0.8166
Epoch 24 | Loss:0.2687 | Acc18:0.6679, F1_18:0.6868 | F1_bin:0.9155, F1_9:0.7009, F1_avg:0.8082
Epoch 25 | Loss:0.2427 | Acc18:0.6728, F1_18:0.6921 | F1_bin:0.9197, F1_9:0.7068, F1_avg:0.8133
Epoch 26 | Loss:0.2372 | Acc18:0.6746, F1_18:0.6951 | F1_bin:0.9172, F1_9:0.7058, F1_avg:0.8115
Epoch 00027: reducing learning rate of group 0 to 1.2500e-04.
Epoch 27 | Loss:0.2361 | Acc18:0.6703, F1_18:0.6909 | F1_bin:0.9210, F1_9:0.7035, F1_avg:0.8123
Epoch 28 | Loss:0.2064 | Acc18:0.6783, F1_18:0.7014 | F1_bin:0.9204, F1_9:0.7116, F1_avg:0.8160
Epoch 29 | Loss:0.1990 | Acc18:0.6862, F1_18:0.7009 | F1_bin:0.9234, F1_9:0.7145, F1_avg:0.8190
Epoch 30 | Loss:0.1916 | Acc18:0.6820, F1_18:0.6981 | F1_bin:0.9210, F1_9:0.7113, F1_avg:0.8161
Epoch 31 | Loss:0.1843 | Acc18:0.6752, F1_18:0.6921 | F1_bin:0.9198, F1_9:0.7053, F1_avg:0.8125
Epoch 00032: reducing learning rate of group 0 to 6.2500e-05.
Epoch 32 | Loss:0.1799 | Acc18:0.6820, F1_18:0.7001 | F1_bin:0.9215, F1_9:0.7140, F1_avg:0.8178
Epoch 33 | Loss:0.1677 | Acc18:0.6758, F1_18:0.6953 | F1_bin:0.9203, F1_9:0.7060, F1_avg:0.8132
Epoch 34 | Loss:0.1641 | Acc18:0.6691, F1_18:0.6874 | F1_bin:0.9148, F1_9:0.7029, F1_avg:0.8089
Epoch 35 | Loss:0.1590 | Acc18:0.6789, F1_18:0.6950 | F1_bin:0.9210, F1_9:0.7114, F1_avg:0.8162
Epoch 00036: reducing learning rate of group 0 to 3.1250e-05.
Epoch 36 | Loss:0.1560 | Acc18:0.6801, F1_18:0.6950 | F1_bin:0.9179, F1_9:0.7077, F1_avg:0.8128
Epoch 37 | Loss:0.1469 | Acc18:0.6838, F1_18:0.7000 | F1_bin:0.9185, F1_9:0.7140, F1_avg:0.8162
Epoch 38 | Loss:0.1494 | Acc18:0.6771, F1_18:0.6967 | F1_bin:0.9179, F1_9:0.7098, F1_avg:0.8138
Epoch 39 | Loss:0.1429 | Acc18:0.6740, F1_18:0.6941 | F1_bin:0.9179, F1_9:0.7071, F1_avg:0.8125
Epoch 40 | Loss:0.1398 | Acc18:0.6838, F1_18:0.7050 | F1_bin:0.9197, F1_9:0.7169, F1_avg:0.8183
Epoch 41 | Loss:0.1484 | Acc18:0.6813, F1_18:0.6999 | F1_bin:0.9185, F1_9:0.7119, F1_avg:0.8152
Epoch 42 | Loss:0.1462 | Acc18:0.6777, F1_18:0.6979 | F1_bin:0.9179, F1_9:0.7104, F1_avg:0.8141
Epoch 43 | Loss:0.1424 | Acc18:0.6771, F1_18:0.6982 | F1_bin:0.9167, F1_9:0.7119, F1_avg:0.8143
Epoch 00044: reducing learning rate of group 0 to 1.5625e-05.
Epoch 44 | Loss:0.1401 | Acc18:0.6832, F1_18:0.7013 | F1_bin:0.9228, F1_9:0.7145, F1_avg:0.8187
Epoch 45 | Loss:0.1434 | Acc18:0.6838, F1_18:0.7024 | F1_bin:0.9161, F1_9:0.7159, F1_avg:0.8160
Epoch 46 | Loss:0.1393 | Acc18:0.6789, F1_18:0.7020 | F1_bin:0.9173, F1_9:0.7148, F1_avg:0.8160
Epoch 47 | Loss:0.1414 | Acc18:0.6777, F1_18:0.6985 | F1_bin:0.9185, F1_9:0.7106, F1_avg:0.8145
Epoch 00048: reducing learning rate of group 0 to 7.8125e-06.
Epoch 48 | Loss:0.1396 | Acc18:0.6820, F1_18:0.7007 | F1_bin:0.9166, F1_9:0.7125, F1_avg:0.8145
Epoch 49 | Loss:0.1368 | Acc18:0.6795, F1_18:0.6994 | F1_bin:0.9160, F1_9:0.7120, F1_avg:0.8140
Epoch 50 | Loss:0.1386 | Acc18:0.6771, F1_18:0.6965 | F1_bin:0.9197, F1_9:0.7095, F1_avg:0.8146
Epoch 51 | Loss:0.1369 | Acc18:0.6789, F1_18:0.6979 | F1_bin:0.9166, F1_9:0.7117, F1_avg:0.8142
Epoch 00052: reducing learning rate of group 0 to 3.9063e-06.
Epoch 52 | Loss:0.1356 | Acc18:0.6850, F1_18:0.7029 | F1_bin:0.9191, F1_9:0.7168, F1_avg:0.8180
Epoch 53 | Loss:0.1286 | Acc18:0.6771, F1_18:0.6942 | F1_bin:0.9179, F1_9:0.7066, F1_avg:0.8123
Epoch 54 | Loss:0.1293 | Acc18:0.6758, F1_18:0.6947 | F1_bin:0.9185, F1_9:0.7076, F1_avg:0.8130
Epoch 55 | Loss:0.1259 | Acc18:0.6758, F1_18:0.6960 | F1_bin:0.9185, F1_9:0.7102, F1_avg:0.8144
Epoch 00056: reducing learning rate of group 0 to 1.9531e-06.
Epoch 56 | Loss:0.1361 | Acc18:0.6795, F1_18:0.6987 | F1_bin:0.9191, F1_9:0.7124, F1_avg:0.8157
Epoch 57 | Loss:0.1320 | Acc18:0.6801, F1_18:0.6999 | F1_bin:0.9185, F1_9:0.7149, F1_avg:0.8167
Epoch 58 | Loss:0.1302 | Acc18:0.6789, F1_18:0.7002 | F1_bin:0.9179, F1_9:0.7129, F1_avg:0.8154
Epoch 59 | Loss:0.1370 | Acc18:0.6856, F1_18:0.7051 | F1_bin:0.9197, F1_9:0.7190, F1_avg:0.8194
Epoch 60 | Loss:0.1302 | Acc18:0.6813, F1_18:0.6978 | F1_bin:0.9167, F1_9:0.7132, F1_avg:0.8149
Epoch 61 | Loss:0.1327 | Acc18:0.6807, F1_18:0.6995 | F1_bin:0.9185, F1_9:0.7135, F1_avg:0.8160
Epoch 62 | Loss:0.1384 | Acc18:0.6789, F1_18:0.6995 | F1_bin:0.9160, F1_9:0.7125, F1_avg:0.8143
Epoch 00063: reducing learning rate of group 0 to 9.7656e-07.
Epoch 63 | Loss:0.1329 | Acc18:0.6783, F1_18:0.6967 | F1_bin:0.9185, F1_9:0.7098, F1_avg:0.8141
Epoch 64 | Loss:0.1278 | Acc18:0.6807, F1_18:0.7009 | F1_bin:0.9179, F1_9:0.7154, F1_avg:0.8166
Epoch 65 | Loss:0.1296 | Acc18:0.6801, F1_18:0.7011 | F1_bin:0.9160, F1_9:0.7133, F1_avg:0.8146
Epoch 66 | Loss:0.1285 | Acc18:0.6820, F1_18:0.7044 | F1_bin:0.9179, F1_9:0.7163, F1_avg:0.8171
Epoch 00067: reducing learning rate of group 0 to 4.8828e-07.
Epoch 67 | Loss:0.1282 | Acc18:0.6777, F1_18:0.6945 | F1_bin:0.9173, F1_9:0.7098, F1_avg:0.8135
Epoch 68 | Loss:0.1269 | Acc18:0.6813, F1_18:0.7030 | F1_bin:0.9148, F1_9:0.7169, F1_avg:0.8159
Epoch 69 | Loss:0.1326 | Acc18:0.6801, F1_18:0.6989 | F1_bin:0.9166, F1_9:0.7121, F1_avg:0.8144
Epoch 70 | Loss:0.1353 | Acc18:0.6777, F1_18:0.6946 | F1_bin:0.9167, F1_9:0.7103, F1_avg:0.8135
Epoch 00071: reducing learning rate of group 0 to 2.4414e-07.
Epoch 71 | Loss:0.1347 | Acc18:0.6789, F1_18:0.6995 | F1_bin:0.9172, F1_9:0.7133, F1_avg:0.8153
Epoch 72 | Loss:0.1306 | Acc18:0.6771, F1_18:0.6954 | F1_bin:0.9191, F1_9:0.7094, F1_avg:0.8143
Epoch 73 | Loss:0.1284 | Acc18:0.6807, F1_18:0.7006 | F1_bin:0.9173, F1_9:0.7117, F1_avg:0.8145
Epoch 74 | Loss:0.1314 | Acc18:0.6783, F1_18:0.6974 | F1_bin:0.9173, F1_9:0.7123, F1_avg:0.8148
Epoch 00075: reducing learning rate of group 0 to 1.2207e-07.
Epoch 75 | Loss:0.1359 | Acc18:0.6838, F1_18:0.7019 | F1_bin:0.9179, F1_9:0.7163, F1_avg:0.8171
Epoch 76 | Loss:0.1318 | Acc18:0.6838, F1_18:0.7023 | F1_bin:0.9185, F1_9:0.7155, F1_avg:0.8170
Epoch 77 | Loss:0.1316 | Acc18:0.6807, F1_18:0.6990 | F1_bin:0.9166, F1_9:0.7105, F1_avg:0.8136
Epoch 78 | Loss:0.1274 | Acc18:0.6826, F1_18:0.7030 | F1_bin:0.9179, F1_9:0.7159, F1_avg:0.8169
Epoch 00079: reducing learning rate of group 0 to 6.1035e-08.
Epoch 79 | Loss:0.1321 | Acc18:0.6789, F1_18:0.6972 | F1_bin:0.9210, F1_9:0.7105, F1_avg:0.8158
Early stopping.
"""
