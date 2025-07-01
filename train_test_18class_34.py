#ModelVariant_LSTMGRU_TinyCNN 特徴量に角速度を追加　allモデル
#モデル保存条件をf1_18からf1_avgに変更
# CV=0.851
#CMI 2025 デモ提出 バージョン71　IMUonly+all LB=81


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.transform import Rotation as R
import warnings

# ============= 角速度・角距離の関数 ==============
def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200):
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    angular_vel = np.zeros((quat_values.shape[0], 3))
    for i in range(quat_values.shape[0] - 1):
        q1, q2 = quat_values[i], quat_values[i+1]
        if np.any(np.isnan(q1)) or np.any(np.isnan(q2)):
            continue
        try:
            delta_rot = R.from_quat(q1).inv() * R.from_quat(q2)
            angular_vel[i] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            pass
    return angular_vel

def calculate_angular_distance(rot_data):
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    angular_dist = np.zeros(quat_values.shape[0])
    for i in range(quat_values.shape[0] - 1):
        q1, q2 = quat_values[i], quat_values[i+1]
        if np.any(np.isnan(q1)) or np.any(np.isnan(q2)):
            continue
        try:
            delta_rot = R.from_quat(q1).inv() * R.from_quat(q2)
            angular_dist[i] = np.linalg.norm(delta_rot.as_rotvec())
        except ValueError:
            pass
    return angular_dist

# ============= 特徴量エンジニアリング ==============
def feature_eng(df):
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))
    df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)

    # ===== 四元数から角速度・角距離 =====
    rot_data = df[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
    angular_vel_group = calculate_angular_velocity_from_quat(rot_data)
    df['angular_vel_x'] = angular_vel_group[:, 0]
    df['angular_vel_y'] = angular_vel_group[:, 1]
    df['angular_vel_z'] = angular_vel_group[:, 2]
    df['angular_dist'] = calculate_angular_distance(rot_data)

    return df

# ============= 以下、あなたの既存のコード構造と同じ ==============
# 必要なTinyCNN, MultiScaleConv1d, SEBlock, ResidualSEBlock, AttentionLayer, MetaFeatureExtractor, GaussianNoise
# それにModelVariant_LSTMGRU_TinyCNN もそのまま使ってください。
# （ここでは省略。あなたが貼った長いコードのままでOK）
class TinyCNN(nn.Module):
    def __init__(self, in_channels=5, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels,16,3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self,x):
        x = self.net(x)
        return x.view(x.size(0), -1)

# ========= その他のサブモジュール（そのまま） =========
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
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
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

class AttentionLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model,1)
    def forward(self, x):
        score = torch.tanh(self.fc(x)).squeeze(-1)
        weights = F.softmax(score, dim=1).unsqueeze(-1)
        return (x*weights).sum(dim=1)

class MetaFeatureExtractor(nn.Module):
    def forward(self,x):
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        maxv,_ = x.max(dim=1)
        minv,_ = x.min(dim=1)
        slope = (x[:,-1,:] - x[:,0,:]) / max(x.size(1)-1,1)
        return torch.cat([mean,std,maxv,minv,slope],dim=1)

class GaussianNoise(nn.Module):
    def __init__(self, stddev=0.09):
        super().__init__()
        self.stddev = stddev
    def forward(self,x):
        if self.training:
            return x + torch.randn_like(x) * self.stddev
        return x

# ========= 最終モデル =========
class ModelVariant_LSTMGRU_TinyCNN(nn.Module):
    def __init__(self, imu_dim, num_classes):
        super().__init__()
        self.imu_dim = imu_dim
        self.tof_out_dim = 32

        # IMU branches
        self.imu_branches = nn.ModuleList([
            nn.Sequential(
                MultiScaleConv1d(1,12),
                ResidualSEBlock(36,48),
                ResidualSEBlock(48,48),
            ) for _ in range(imu_dim)
        ])

        # TOF branch
        self.tof_cnn = TinyCNN(in_channels=5, out_dim=32)

        # Meta feature
        self.meta = MetaFeatureExtractor()
        self.meta_dense = nn.Sequential(
            nn.Linear(5*(imu_dim + self.tof_out_dim), 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Sequence encoders
        self.bigru = nn.GRU(48*imu_dim + self.tof_out_dim, 128, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2)
        self.bilstm = nn.LSTM(48*imu_dim + self.tof_out_dim, 128, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2)
        self.noise = GaussianNoise(0.09)

        # Attention + Head
        concat_dim = 256 + 256 + (48*imu_dim + self.tof_out_dim)
        self.attn = AttentionLayer(concat_dim)
        self.head = nn.Sequential(
            nn.Linear(concat_dim + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_imu, x_tof):


        # ===== IMU branch =====
        imu_feats = []
        for i in range(x_imu.shape[2]):
            xi = x_imu[:,:,i].unsqueeze(1)  # (B,T,1)
            fi = self.imu_branches[i](xi).transpose(1,2)  # (B,T,F)
            imu_feats.append(fi)
        imu_feat = torch.cat(imu_feats, dim=2)  # (B,T,48*imu_dim)

        # ===== TOF branch =====
        B,T,C,H,W = x_tof.shape
        tof_flat = x_tof.view(B*T, C, H, W)  # (B*T,5,8,8)

        tof_feats = self.tof_cnn(tof_flat).view(B, T, -1)  # (B,T,32)


        # ===== align time dimension =====
        tof_feats = F.adaptive_avg_pool1d(tof_feats.transpose(1,2), output_size=imu_feat.size(1)).transpose(1,2)


        # ===== Meta features =====
        meta_imu = self.meta(x_imu)       # (B,5*imu_dim)
        meta_tof = self.meta(tof_feats)    # (B,5*32)
        meta = torch.cat([meta_imu, meta_tof], dim=1)
        meta = self.meta_dense(meta)       # (B,64)


        # ===== Sequence fusion =====
        seq = torch.cat([imu_feat, tof_feats], dim=2)  # (B,T,48*imu_dim+32)

        gru,_ = self.bigru(seq)   # (B,T,256)
        lstm,_ = self.bilstm(seq) # (B,T,256)
        noise = self.noise(seq)   # (B,T,48*imu_dim+32)
        x = torch.cat([gru, lstm, noise], dim=2)  # (B,T,256+256+...)


        # ===== Attention & Head =====
        x = self.attn(x)  # (B,256+256+...)


        x = torch.cat([x, meta], dim=1)  # (B, ...)
        out = self.head(x)                # (B,num_classes)

        return out, None


# ============= データ読み込み ==============
RAW_CSV = "train.csv"
SAVE_DIR = "train_test_18class_34"
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

# ========= 特徴量カラム =========
num_cols = df.select_dtypes(include=[np.number]).columns
imu_cols = [c for c in num_cols if c.startswith(('acc_', 'rot_', 'acc_mag', 'rot_angle',
                                                 'acc_mag_jerk', 'rot_angle_vel',
                                                 'angular_vel', 'angular_dist', 'thm_'))]
tof_cols = [c for c in num_cols if c.startswith('tof_')]
imu_cols = [c for c in imu_cols if c != 'gesture_class']
tof_cols = [c for c in tof_cols if c != 'gesture_class']

imu_dim, tof_dim = len(imu_cols), len(tof_cols)
lens = df.groupby("sequence_id").size().values
pad_len = int(np.percentile(lens, PAD_PERCENTILE))

def to_binary(y): return [0 if i<9 else 1 for i in y]
def to_9class(y): return [i%9 for i in y]

# ========= Dataset =========
class TwoBranchDataset(Dataset):
    def __init__(self, imu_seqs, tof_seqs, y):
        self.imu_seqs = torch.tensor(np.array(imu_seqs), dtype=torch.float32)
        self.tof_seqs = torch.tensor(np.array(tof_seqs), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.imu_seqs)
    def __getitem__(self, idx): return self.imu_seqs[idx], self.tof_seqs[idx], self.y[idx]

# ========= Fold loop =========
kf = GroupKFold(n_splits=5)
seq_ids = df["sequence_id"].unique()
subject_map = df.drop_duplicates("sequence_id").set_index("sequence_id")["subject"]
groups = [subject_map[sid] for sid in seq_ids]

for fold, (tr_idx, va_idx) in enumerate(kf.split(seq_ids, groups=groups)):
    print(f"\n=== Fold {fold+1} ===")
    fold_dir = os.path.join(SAVE_DIR, f"fold{fold+1}")
    os.makedirs(fold_dir, exist_ok=True)

    train_ids, val_ids = seq_ids[tr_idx], seq_ids[va_idx]
    train_df = df[df["sequence_id"].isin(train_ids)]
    val_df = df[df["sequence_id"].isin(val_ids)]

    scaler = StandardScaler().fit(train_df[imu_cols + tof_cols].fillna(0))
    joblib.dump(scaler, os.path.join(fold_dir, "scaler.pkl"))

    def prepare_data(ids, df):
        X_imu, X_tof, y = [], [], []
        for sid in ids:
            g = df[df["sequence_id"]==sid]
            m = scaler.transform(g[imu_cols + tof_cols].ffill().bfill().fillna(0))
            imu_m, tof_m = m[:,:imu_dim], m[:,imu_dim:]
            imu_m = np.pad(imu_m, ((0,max(0,pad_len-len(imu_m))), (0,0)))
            tof_m = np.pad(tof_m, ((0,max(0,pad_len-len(tof_m))), (0,0)))
            tof_img = tof_m.reshape(-1,5,8,8)
            tof_img = np.pad(tof_img, ((0,max(0,pad_len-len(tof_img))), (0,0),(0,0),(0,0)))
            X_imu.append(imu_m[:pad_len])
            X_tof.append(tof_img[:pad_len])
            y.append(g["gesture_class"].iloc[0])
        return X_imu, X_tof, y

    X_imu_train, X_tof_train, y_train = prepare_data(train_ids, train_df)
    X_imu_val, X_tof_val, y_val = prepare_data(val_ids, val_df)

    train_loader = DataLoader(TwoBranchDataset(X_imu_train, X_tof_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TwoBranchDataset(X_imu_val, X_tof_val, y_val), batch_size=BATCH_SIZE)

    # ========= モデル =========
    model = ModelVariant_LSTMGRU_TinyCNN(imu_dim, len(le.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    best_f1_avg, patience_counter = 0, 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb_imu, xb_tof, yb in train_loader:
            xb_imu, xb_tof, yb = xb_imu.to(device), xb_tof.to(device), yb.to(device)
            optimizer.zero_grad()
            logits,_ = model(xb_imu, xb_tof)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb_imu)

        # ========= Validation =========
        model.eval()
        preds, trues = [], []
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


        print(
            f"Epoch {epoch+1:3d} | "
            f"Loss: {total_loss / len(train_loader.dataset):.4f} | "
            f"Acc18: {acc_18:.4f} F1_18: {f1_18:.4f} | "
            f"F1_bin: {f1_bin:.4f} F1_9: {f1_9:.4f} F1_avg: {f1_avg:.4f}",
            flush=True
        )

        if f1_avg > best_f1_avg:
            best_f1_avg, patience_counter = f1_avg, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "imu_dim": imu_dim,
                "tof_dim": tof_dim,
                "pad_len": pad_len,
                "classes": le.classes_,
            }, os.path.join(fold_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

r"""
=== Fold 1 ===
Epoch   1 | Loss: 1.7099 | Acc18: 0.5821 F1_18: 0.5920 | F1_bin: 0.8774 F1_9: 0.6295 F1_avg: 0.7534
Epoch   2 | Loss: 1.1456 | Acc18: 0.6183 F1_18: 0.6383 | F1_bin: 0.8716 F1_9: 0.6647 F1_avg: 0.7682
Epoch   3 | Loss: 0.9931 | Acc18: 0.6489 F1_18: 0.6816 | F1_bin: 0.8930 F1_9: 0.7037 F1_avg: 0.7983
Epoch   4 | Loss: 0.8748 | Acc18: 0.6789 F1_18: 0.7002 | F1_bin: 0.8956 F1_9: 0.7276 F1_avg: 0.8116
Epoch   5 | Loss: 0.7949 | Acc18: 0.6893 F1_18: 0.7051 | F1_bin: 0.9091 F1_9: 0.7306 F1_avg: 0.8198
Epoch   6 | Loss: 0.7551 | Acc18: 0.6765 F1_18: 0.6967 | F1_bin: 0.9030 F1_9: 0.7152 F1_avg: 0.8091
Epoch   7 | Loss: 0.6960 | Acc18: 0.7163 F1_18: 0.7360 | F1_bin: 0.9176 F1_9: 0.7446 F1_avg: 0.8311
Epoch   8 | Loss: 0.6576 | Acc18: 0.7328 F1_18: 0.7485 | F1_bin: 0.9226 F1_9: 0.7609 F1_avg: 0.8418
Epoch   9 | Loss: 0.6240 | Acc18: 0.7169 F1_18: 0.7494 | F1_bin: 0.9219 F1_9: 0.7634 F1_avg: 0.8426
Epoch  10 | Loss: 0.5912 | Acc18: 0.7335 F1_18: 0.7602 | F1_bin: 0.9316 F1_9: 0.7718 F1_avg: 0.8517
Epoch  11 | Loss: 0.5595 | Acc18: 0.7420 F1_18: 0.7577 | F1_bin: 0.9322 F1_9: 0.7693 F1_avg: 0.8508
Epoch  12 | Loss: 0.5285 | Acc18: 0.7537 F1_18: 0.7709 | F1_bin: 0.9325 F1_9: 0.7727 F1_avg: 0.8526
Epoch  13 | Loss: 0.5107 | Acc18: 0.7341 F1_18: 0.7593 | F1_bin: 0.9319 F1_9: 0.7715 F1_avg: 0.8517
Epoch  14 | Loss: 0.4838 | Acc18: 0.7328 F1_18: 0.7589 | F1_bin: 0.9342 F1_9: 0.7651 F1_avg: 0.8497
Epoch  15 | Loss: 0.4837 | Acc18: 0.7390 F1_18: 0.7564 | F1_bin: 0.9367 F1_9: 0.7707 F1_avg: 0.8537
Epoch  16 | Loss: 0.4473 | Acc18: 0.7561 F1_18: 0.7865 | F1_bin: 0.9347 F1_9: 0.7905 F1_avg: 0.8626
Epoch  17 | Loss: 0.4240 | Acc18: 0.7659 F1_18: 0.7878 | F1_bin: 0.9404 F1_9: 0.7982 F1_avg: 0.8693
Epoch  18 | Loss: 0.4142 | Acc18: 0.7433 F1_18: 0.7727 | F1_bin: 0.9244 F1_9: 0.7864 F1_avg: 0.8554
Epoch  19 | Loss: 0.3831 | Acc18: 0.7629 F1_18: 0.8028 | F1_bin: 0.9252 F1_9: 0.8056 F1_avg: 0.8654
Epoch  20 | Loss: 0.3922 | Acc18: 0.7616 F1_18: 0.7877 | F1_bin: 0.9410 F1_9: 0.7937 F1_avg: 0.8673
Epoch  21 | Loss: 0.3645 | Acc18: 0.7635 F1_18: 0.7995 | F1_bin: 0.9376 F1_9: 0.8032 F1_avg: 0.8704
Epoch  22 | Loss: 0.3653 | Acc18: 0.7555 F1_18: 0.7764 | F1_bin: 0.9216 F1_9: 0.7823 F1_avg: 0.8519
Epoch 00023: reducing learning rate of group 0 to 5.0000e-04.
Epoch  23 | Loss: 0.3477 | Acc18: 0.7623 F1_18: 0.7919 | F1_bin: 0.9277 F1_9: 0.8023 F1_avg: 0.8650
Epoch  24 | Loss: 0.2709 | Acc18: 0.7843 F1_18: 0.8091 | F1_bin: 0.9417 F1_9: 0.8153 F1_avg: 0.8785
Epoch  25 | Loss: 0.2361 | Acc18: 0.7788 F1_18: 0.8088 | F1_bin: 0.9471 F1_9: 0.8142 F1_avg: 0.8807
Epoch  26 | Loss: 0.2191 | Acc18: 0.7751 F1_18: 0.8105 | F1_bin: 0.9414 F1_9: 0.8181 F1_avg: 0.8798
Epoch  27 | Loss: 0.2118 | Acc18: 0.7751 F1_18: 0.8097 | F1_bin: 0.9403 F1_9: 0.8163 F1_avg: 0.8783
Epoch  28 | Loss: 0.1986 | Acc18: 0.7794 F1_18: 0.8091 | F1_bin: 0.9477 F1_9: 0.8150 F1_avg: 0.8814
Epoch  29 | Loss: 0.1882 | Acc18: 0.7708 F1_18: 0.7926 | F1_bin: 0.9490 F1_9: 0.8025 F1_avg: 0.8758
Epoch 00030: reducing learning rate of group 0 to 2.5000e-04.
Epoch  30 | Loss: 0.1904 | Acc18: 0.7702 F1_18: 0.8033 | F1_bin: 0.9323 F1_9: 0.8118 F1_avg: 0.8720
Epoch  31 | Loss: 0.1607 | Acc18: 0.7868 F1_18: 0.8169 | F1_bin: 0.9495 F1_9: 0.8228 F1_avg: 0.8862
Epoch  32 | Loss: 0.1476 | Acc18: 0.7849 F1_18: 0.8160 | F1_bin: 0.9452 F1_9: 0.8216 F1_avg: 0.8834
Epoch  33 | Loss: 0.1481 | Acc18: 0.7751 F1_18: 0.8018 | F1_bin: 0.9422 F1_9: 0.8111 F1_avg: 0.8766
Epoch  34 | Loss: 0.1351 | Acc18: 0.7770 F1_18: 0.8037 | F1_bin: 0.9477 F1_9: 0.8124 F1_avg: 0.8801
Epoch 00035: reducing learning rate of group 0 to 1.2500e-04.
Epoch  35 | Loss: 0.1338 | Acc18: 0.7831 F1_18: 0.8080 | F1_bin: 0.9452 F1_9: 0.8134 F1_avg: 0.8793
Epoch  36 | Loss: 0.1116 | Acc18: 0.7862 F1_18: 0.8081 | F1_bin: 0.9459 F1_9: 0.8155 F1_avg: 0.8807
Epoch  37 | Loss: 0.1073 | Acc18: 0.7800 F1_18: 0.8090 | F1_bin: 0.9452 F1_9: 0.8152 F1_avg: 0.8802
Epoch  38 | Loss: 0.1021 | Acc18: 0.7782 F1_18: 0.8086 | F1_bin: 0.9465 F1_9: 0.8148 F1_avg: 0.8807
Epoch 00039: reducing learning rate of group 0 to 6.2500e-05.
Epoch  39 | Loss: 0.1013 | Acc18: 0.7862 F1_18: 0.8153 | F1_bin: 0.9440 F1_9: 0.8234 F1_avg: 0.8837
Epoch  40 | Loss: 0.0910 | Acc18: 0.7806 F1_18: 0.8124 | F1_bin: 0.9465 F1_9: 0.8199 F1_avg: 0.8832
Epoch  41 | Loss: 0.0944 | Acc18: 0.7819 F1_18: 0.8107 | F1_bin: 0.9446 F1_9: 0.8169 F1_avg: 0.8807
Epoch  42 | Loss: 0.0893 | Acc18: 0.7831 F1_18: 0.8133 | F1_bin: 0.9483 F1_9: 0.8193 F1_avg: 0.8838
Epoch 00043: reducing learning rate of group 0 to 3.1250e-05.
Epoch  43 | Loss: 0.0821 | Acc18: 0.7874 F1_18: 0.8146 | F1_bin: 0.9484 F1_9: 0.8200 F1_avg: 0.8842
Epoch  44 | Loss: 0.0867 | Acc18: 0.7886 F1_18: 0.8149 | F1_bin: 0.9477 F1_9: 0.8201 F1_avg: 0.8839
Epoch  45 | Loss: 0.0879 | Acc18: 0.7849 F1_18: 0.8128 | F1_bin: 0.9440 F1_9: 0.8199 F1_avg: 0.8820
Epoch  46 | Loss: 0.0796 | Acc18: 0.7880 F1_18: 0.8151 | F1_bin: 0.9471 F1_9: 0.8218 F1_avg: 0.8845
Epoch 00047: reducing learning rate of group 0 to 1.5625e-05.
Epoch  47 | Loss: 0.0831 | Acc18: 0.7868 F1_18: 0.8140 | F1_bin: 0.9452 F1_9: 0.8209 F1_avg: 0.8831
Epoch  48 | Loss: 0.0866 | Acc18: 0.7874 F1_18: 0.8159 | F1_bin: 0.9465 F1_9: 0.8234 F1_avg: 0.8849
Epoch  49 | Loss: 0.0800 | Acc18: 0.7843 F1_18: 0.8136 | F1_bin: 0.9452 F1_9: 0.8204 F1_avg: 0.8828
Epoch  50 | Loss: 0.0768 | Acc18: 0.7874 F1_18: 0.8159 | F1_bin: 0.9452 F1_9: 0.8228 F1_avg: 0.8840
Epoch 00051: reducing learning rate of group 0 to 7.8125e-06.
Epoch  51 | Loss: 0.0784 | Acc18: 0.7898 F1_18: 0.8146 | F1_bin: 0.9465 F1_9: 0.8217 F1_avg: 0.8841
Early stopping.

=== Fold 2 ===
Epoch   1 | Loss: 1.6995 | Acc18: 0.4384 F1_18: 0.4337 | F1_bin: 0.8239 F1_9: 0.4789 F1_avg: 0.6514
Epoch   2 | Loss: 1.1315 | Acc18: 0.5843 F1_18: 0.6020 | F1_bin: 0.8665 F1_9: 0.6366 F1_avg: 0.7515
Epoch   3 | Loss: 0.9616 | Acc18: 0.5984 F1_18: 0.6202 | F1_bin: 0.8681 F1_9: 0.6507 F1_avg: 0.7594
Epoch   4 | Loss: 0.8268 | Acc18: 0.5917 F1_18: 0.6162 | F1_bin: 0.8261 F1_9: 0.6586 F1_avg: 0.7423
Epoch   5 | Loss: 0.7642 | Acc18: 0.6211 F1_18: 0.6322 | F1_bin: 0.8651 F1_9: 0.6625 F1_avg: 0.7638
Epoch   6 | Loss: 0.7120 | Acc18: 0.6327 F1_18: 0.6741 | F1_bin: 0.8660 F1_9: 0.6985 F1_avg: 0.7823
Epoch   7 | Loss: 0.6647 | Acc18: 0.6229 F1_18: 0.6541 | F1_bin: 0.8711 F1_9: 0.6740 F1_avg: 0.7725
Epoch   8 | Loss: 0.6285 | Acc18: 0.6468 F1_18: 0.6728 | F1_bin: 0.8754 F1_9: 0.6968 F1_avg: 0.7861
Epoch   9 | Loss: 0.5870 | Acc18: 0.6493 F1_18: 0.6773 | F1_bin: 0.8798 F1_9: 0.6936 F1_avg: 0.7867
Epoch  10 | Loss: 0.5471 | Acc18: 0.6321 F1_18: 0.6598 | F1_bin: 0.8789 F1_9: 0.6789 F1_avg: 0.7789
Epoch  11 | Loss: 0.5316 | Acc18: 0.6536 F1_18: 0.6854 | F1_bin: 0.8879 F1_9: 0.7001 F1_avg: 0.7940
Epoch  12 | Loss: 0.5002 | Acc18: 0.6383 F1_18: 0.6724 | F1_bin: 0.8675 F1_9: 0.6948 F1_avg: 0.7812
Epoch  13 | Loss: 0.4859 | Acc18: 0.6560 F1_18: 0.6881 | F1_bin: 0.8823 F1_9: 0.7032 F1_avg: 0.7927
Epoch  14 | Loss: 0.4655 | Acc18: 0.6444 F1_18: 0.6824 | F1_bin: 0.8828 F1_9: 0.7035 F1_avg: 0.7931
Epoch  15 | Loss: 0.4372 | Acc18: 0.6573 F1_18: 0.6967 | F1_bin: 0.8822 F1_9: 0.7143 F1_avg: 0.7983
Epoch  16 | Loss: 0.4147 | Acc18: 0.6603 F1_18: 0.7048 | F1_bin: 0.8797 F1_9: 0.7183 F1_avg: 0.7990
Epoch  17 | Loss: 0.4064 | Acc18: 0.6646 F1_18: 0.7051 | F1_bin: 0.8840 F1_9: 0.7175 F1_avg: 0.8007
Epoch  18 | Loss: 0.3790 | Acc18: 0.6530 F1_18: 0.7031 | F1_bin: 0.8840 F1_9: 0.7120 F1_avg: 0.7980
Epoch  19 | Loss: 0.3872 | Acc18: 0.6622 F1_18: 0.7088 | F1_bin: 0.8858 F1_9: 0.7194 F1_avg: 0.8026
Epoch  20 | Loss: 0.3393 | Acc18: 0.6444 F1_18: 0.6898 | F1_bin: 0.8804 F1_9: 0.7075 F1_avg: 0.7939
Epoch  21 | Loss: 0.3286 | Acc18: 0.6579 F1_18: 0.6965 | F1_bin: 0.8821 F1_9: 0.7141 F1_avg: 0.7981
Epoch  22 | Loss: 0.3156 | Acc18: 0.6542 F1_18: 0.6945 | F1_bin: 0.8797 F1_9: 0.7038 F1_avg: 0.7917
Epoch 00023: reducing learning rate of group 0 to 5.0000e-04.
Epoch  23 | Loss: 0.3136 | Acc18: 0.6401 F1_18: 0.6875 | F1_bin: 0.8823 F1_9: 0.7015 F1_avg: 0.7919
Epoch  24 | Loss: 0.2494 | Acc18: 0.6855 F1_18: 0.7239 | F1_bin: 0.8902 F1_9: 0.7384 F1_avg: 0.8143
Epoch  25 | Loss: 0.2160 | Acc18: 0.6750 F1_18: 0.7139 | F1_bin: 0.8876 F1_9: 0.7301 F1_avg: 0.8088
Epoch  26 | Loss: 0.2125 | Acc18: 0.6720 F1_18: 0.7173 | F1_bin: 0.8901 F1_9: 0.7278 F1_avg: 0.8090
Epoch  27 | Loss: 0.1908 | Acc18: 0.6701 F1_18: 0.7108 | F1_bin: 0.8815 F1_9: 0.7287 F1_avg: 0.8051
Epoch 00028: reducing learning rate of group 0 to 2.5000e-04.
Epoch  28 | Loss: 0.1952 | Acc18: 0.6579 F1_18: 0.7095 | F1_bin: 0.8795 F1_9: 0.7234 F1_avg: 0.8014
Epoch  29 | Loss: 0.1650 | Acc18: 0.6757 F1_18: 0.7155 | F1_bin: 0.8835 F1_9: 0.7280 F1_avg: 0.8057
Epoch  30 | Loss: 0.1486 | Acc18: 0.6646 F1_18: 0.7080 | F1_bin: 0.8797 F1_9: 0.7233 F1_avg: 0.8015
Epoch  31 | Loss: 0.1403 | Acc18: 0.6665 F1_18: 0.7087 | F1_bin: 0.8829 F1_9: 0.7207 F1_avg: 0.8018
Epoch 00032: reducing learning rate of group 0 to 1.2500e-04.
Epoch  32 | Loss: 0.1367 | Acc18: 0.6769 F1_18: 0.7215 | F1_bin: 0.8872 F1_9: 0.7318 F1_avg: 0.8095
Epoch  33 | Loss: 0.1215 | Acc18: 0.6836 F1_18: 0.7299 | F1_bin: 0.8914 F1_9: 0.7381 F1_avg: 0.8148
Epoch  34 | Loss: 0.1147 | Acc18: 0.6701 F1_18: 0.7169 | F1_bin: 0.8871 F1_9: 0.7278 F1_avg: 0.8075
Epoch  35 | Loss: 0.1092 | Acc18: 0.6732 F1_18: 0.7177 | F1_bin: 0.8871 F1_9: 0.7306 F1_avg: 0.8089
Epoch  36 | Loss: 0.1090 | Acc18: 0.6732 F1_18: 0.7210 | F1_bin: 0.8859 F1_9: 0.7343 F1_avg: 0.8101
Epoch 00037: reducing learning rate of group 0 to 6.2500e-05.
Epoch  37 | Loss: 0.1100 | Acc18: 0.6714 F1_18: 0.7116 | F1_bin: 0.8853 F1_9: 0.7267 F1_avg: 0.8060
Epoch  38 | Loss: 0.0985 | Acc18: 0.6726 F1_18: 0.7139 | F1_bin: 0.8853 F1_9: 0.7278 F1_avg: 0.8065
Epoch  39 | Loss: 0.0967 | Acc18: 0.6720 F1_18: 0.7176 | F1_bin: 0.8877 F1_9: 0.7306 F1_avg: 0.8091
Epoch  40 | Loss: 0.0959 | Acc18: 0.6750 F1_18: 0.7171 | F1_bin: 0.8859 F1_9: 0.7323 F1_avg: 0.8091
Epoch 00041: reducing learning rate of group 0 to 3.1250e-05.
Epoch  41 | Loss: 0.0929 | Acc18: 0.6714 F1_18: 0.7153 | F1_bin: 0.8883 F1_9: 0.7293 F1_avg: 0.8088
Epoch  42 | Loss: 0.0923 | Acc18: 0.6714 F1_18: 0.7158 | F1_bin: 0.8878 F1_9: 0.7278 F1_avg: 0.8078
Epoch  43 | Loss: 0.0979 | Acc18: 0.6695 F1_18: 0.7096 | F1_bin: 0.8884 F1_9: 0.7248 F1_avg: 0.8066
Epoch  44 | Loss: 0.0924 | Acc18: 0.6695 F1_18: 0.7136 | F1_bin: 0.8896 F1_9: 0.7272 F1_avg: 0.8084
Epoch 00045: reducing learning rate of group 0 to 1.5625e-05.
Epoch  45 | Loss: 0.0862 | Acc18: 0.6701 F1_18: 0.7124 | F1_bin: 0.8859 F1_9: 0.7295 F1_avg: 0.8077
Epoch  46 | Loss: 0.0889 | Acc18: 0.6726 F1_18: 0.7149 | F1_bin: 0.8865 F1_9: 0.7305 F1_avg: 0.8085
Epoch  47 | Loss: 0.0833 | Acc18: 0.6744 F1_18: 0.7168 | F1_bin: 0.8914 F1_9: 0.7317 F1_avg: 0.8116
Epoch  48 | Loss: 0.0902 | Acc18: 0.6665 F1_18: 0.7111 | F1_bin: 0.8853 F1_9: 0.7261 F1_avg: 0.8057
Epoch 00049: reducing learning rate of group 0 to 7.8125e-06.
Epoch  49 | Loss: 0.0836 | Acc18: 0.6738 F1_18: 0.7159 | F1_bin: 0.8878 F1_9: 0.7283 F1_avg: 0.8080
Epoch  50 | Loss: 0.0820 | Acc18: 0.6726 F1_18: 0.7158 | F1_bin: 0.8902 F1_9: 0.7268 F1_avg: 0.8085
Epoch  51 | Loss: 0.0842 | Acc18: 0.6701 F1_18: 0.7152 | F1_bin: 0.8889 F1_9: 0.7278 F1_avg: 0.8084
Epoch  52 | Loss: 0.0858 | Acc18: 0.6689 F1_18: 0.7111 | F1_bin: 0.8871 F1_9: 0.7263 F1_avg: 0.8067
Epoch 00053: reducing learning rate of group 0 to 3.9063e-06.
Epoch  53 | Loss: 0.0815 | Acc18: 0.6708 F1_18: 0.7148 | F1_bin: 0.8896 F1_9: 0.7281 F1_avg: 0.8088
Early stopping.

=== Fold 3 ===
Epoch   1 | Loss: 1.6438 | Acc18: 0.5802 F1_18: 0.5696 | F1_bin: 0.8622 F1_9: 0.6113 F1_avg: 0.7368
Epoch   2 | Loss: 1.1205 | Acc18: 0.6472 F1_18: 0.6533 | F1_bin: 0.8739 F1_9: 0.6821 F1_avg: 0.7780
Epoch   3 | Loss: 0.9614 | Acc18: 0.6337 F1_18: 0.6333 | F1_bin: 0.8810 F1_9: 0.6759 F1_avg: 0.7785
Epoch   4 | Loss: 0.8745 | Acc18: 0.6829 F1_18: 0.7005 | F1_bin: 0.9051 F1_9: 0.7252 F1_avg: 0.8151
Epoch   5 | Loss: 0.8180 | Acc18: 0.6724 F1_18: 0.6898 | F1_bin: 0.8851 F1_9: 0.7156 F1_avg: 0.8003
Epoch   6 | Loss: 0.7292 | Acc18: 0.7044 F1_18: 0.7236 | F1_bin: 0.9079 F1_9: 0.7441 F1_avg: 0.8260
Epoch   7 | Loss: 0.6879 | Acc18: 0.7167 F1_18: 0.7188 | F1_bin: 0.9179 F1_9: 0.7483 F1_avg: 0.8331
Epoch   8 | Loss: 0.6547 | Acc18: 0.7271 F1_18: 0.7407 | F1_bin: 0.9136 F1_9: 0.7624 F1_avg: 0.8380
Epoch   9 | Loss: 0.6270 | Acc18: 0.7062 F1_18: 0.7169 | F1_bin: 0.9070 F1_9: 0.7316 F1_avg: 0.8193
Epoch  10 | Loss: 0.5852 | Acc18: 0.7191 F1_18: 0.7326 | F1_bin: 0.9046 F1_9: 0.7569 F1_avg: 0.8307
Epoch  11 | Loss: 0.5689 | Acc18: 0.7124 F1_18: 0.7359 | F1_bin: 0.9077 F1_9: 0.7543 F1_avg: 0.8310
Epoch  12 | Loss: 0.5348 | Acc18: 0.7326 F1_18: 0.7504 | F1_bin: 0.9128 F1_9: 0.7635 F1_avg: 0.8381
Epoch  13 | Loss: 0.5129 | Acc18: 0.7480 F1_18: 0.7619 | F1_bin: 0.9149 F1_9: 0.7795 F1_avg: 0.8472
Epoch  14 | Loss: 0.4924 | Acc18: 0.7246 F1_18: 0.7581 | F1_bin: 0.9130 F1_9: 0.7716 F1_avg: 0.8423
Epoch  15 | Loss: 0.4702 | Acc18: 0.7265 F1_18: 0.7473 | F1_bin: 0.9126 F1_9: 0.7680 F1_avg: 0.8403
Epoch  16 | Loss: 0.4545 | Acc18: 0.7308 F1_18: 0.7512 | F1_bin: 0.9156 F1_9: 0.7711 F1_avg: 0.8434
Epoch 00017: reducing learning rate of group 0 to 5.0000e-04.
Epoch  17 | Loss: 0.4542 | Acc18: 0.7283 F1_18: 0.7494 | F1_bin: 0.9180 F1_9: 0.7680 F1_avg: 0.8430
Epoch  18 | Loss: 0.3584 | Acc18: 0.7640 F1_18: 0.7918 | F1_bin: 0.9265 F1_9: 0.8055 F1_avg: 0.8660
Epoch  19 | Loss: 0.3189 | Acc18: 0.7468 F1_18: 0.7669 | F1_bin: 0.9156 F1_9: 0.7871 F1_avg: 0.8513
Epoch  20 | Loss: 0.3034 | Acc18: 0.7468 F1_18: 0.7716 | F1_bin: 0.9242 F1_9: 0.7839 F1_avg: 0.8541
Epoch  21 | Loss: 0.2904 | Acc18: 0.7628 F1_18: 0.7862 | F1_bin: 0.9223 F1_9: 0.8038 F1_avg: 0.8631
Epoch 00022: reducing learning rate of group 0 to 2.5000e-04.
Epoch  22 | Loss: 0.2713 | Acc18: 0.7382 F1_18: 0.7659 | F1_bin: 0.9171 F1_9: 0.7785 F1_avg: 0.8478
Epoch  23 | Loss: 0.2360 | Acc18: 0.7671 F1_18: 0.7903 | F1_bin: 0.9162 F1_9: 0.8108 F1_avg: 0.8635
Epoch  24 | Loss: 0.2138 | Acc18: 0.7646 F1_18: 0.7872 | F1_bin: 0.9180 F1_9: 0.8067 F1_avg: 0.8624
Epoch  25 | Loss: 0.2112 | Acc18: 0.7591 F1_18: 0.7831 | F1_bin: 0.9193 F1_9: 0.7993 F1_avg: 0.8593
Epoch 00026: reducing learning rate of group 0 to 1.2500e-04.
Epoch  26 | Loss: 0.2018 | Acc18: 0.7535 F1_18: 0.7763 | F1_bin: 0.9180 F1_9: 0.7936 F1_avg: 0.8558
Epoch  27 | Loss: 0.1852 | Acc18: 0.7664 F1_18: 0.7926 | F1_bin: 0.9216 F1_9: 0.8071 F1_avg: 0.8644
Epoch  28 | Loss: 0.1730 | Acc18: 0.7677 F1_18: 0.7916 | F1_bin: 0.9186 F1_9: 0.8077 F1_avg: 0.8631
Epoch  29 | Loss: 0.1707 | Acc18: 0.7634 F1_18: 0.7899 | F1_bin: 0.9186 F1_9: 0.8052 F1_avg: 0.8619
Epoch  30 | Loss: 0.1693 | Acc18: 0.7628 F1_18: 0.7864 | F1_bin: 0.9204 F1_9: 0.8050 F1_avg: 0.8627
Epoch 00031: reducing learning rate of group 0 to 6.2500e-05.
Epoch  31 | Loss: 0.1604 | Acc18: 0.7560 F1_18: 0.7858 | F1_bin: 0.9174 F1_9: 0.8026 F1_avg: 0.8600
Epoch  32 | Loss: 0.1629 | Acc18: 0.7560 F1_18: 0.7838 | F1_bin: 0.9173 F1_9: 0.8025 F1_avg: 0.8599
Epoch  33 | Loss: 0.1454 | Acc18: 0.7585 F1_18: 0.7835 | F1_bin: 0.9192 F1_9: 0.8025 F1_avg: 0.8609
Epoch  34 | Loss: 0.1440 | Acc18: 0.7548 F1_18: 0.7801 | F1_bin: 0.9173 F1_9: 0.7985 F1_avg: 0.8579
Epoch 00035: reducing learning rate of group 0 to 3.1250e-05.
Epoch  35 | Loss: 0.1360 | Acc18: 0.7566 F1_18: 0.7822 | F1_bin: 0.9168 F1_9: 0.8017 F1_avg: 0.8592
Epoch  36 | Loss: 0.1352 | Acc18: 0.7603 F1_18: 0.7862 | F1_bin: 0.9192 F1_9: 0.8051 F1_avg: 0.8622
Epoch  37 | Loss: 0.1382 | Acc18: 0.7566 F1_18: 0.7788 | F1_bin: 0.9186 F1_9: 0.7984 F1_avg: 0.8585
Epoch  38 | Loss: 0.1361 | Acc18: 0.7603 F1_18: 0.7857 | F1_bin: 0.9217 F1_9: 0.8004 F1_avg: 0.8610
Early stopping.

=== Fold 4 ===
Epoch   1 | Loss: 1.6720 | Acc18: 0.5578 F1_18: 0.5778 | F1_bin: 0.8360 F1_9: 0.5976 F1_avg: 0.7168
Epoch   2 | Loss: 1.1180 | Acc18: 0.6181 F1_18: 0.6181 | F1_bin: 0.8714 F1_9: 0.6466 F1_avg: 0.7590
Epoch   3 | Loss: 0.9596 | Acc18: 0.6107 F1_18: 0.6025 | F1_bin: 0.8813 F1_9: 0.6266 F1_avg: 0.7539
Epoch   4 | Loss: 0.8665 | Acc18: 0.6464 F1_18: 0.6770 | F1_bin: 0.8895 F1_9: 0.6962 F1_avg: 0.7928
Epoch   5 | Loss: 0.7802 | Acc18: 0.6673 F1_18: 0.6991 | F1_bin: 0.9014 F1_9: 0.7017 F1_avg: 0.8016
Epoch   6 | Loss: 0.7334 | Acc18: 0.6636 F1_18: 0.6759 | F1_bin: 0.8812 F1_9: 0.7022 F1_avg: 0.7917
Epoch   7 | Loss: 0.6767 | Acc18: 0.6784 F1_18: 0.7059 | F1_bin: 0.9082 F1_9: 0.7101 F1_avg: 0.8091
Epoch   8 | Loss: 0.6356 | Acc18: 0.6980 F1_18: 0.7103 | F1_bin: 0.9114 F1_9: 0.7279 F1_avg: 0.8196
Epoch   9 | Loss: 0.6046 | Acc18: 0.6907 F1_18: 0.7127 | F1_bin: 0.9077 F1_9: 0.7308 F1_avg: 0.8193
Epoch  10 | Loss: 0.5566 | Acc18: 0.6968 F1_18: 0.7254 | F1_bin: 0.9051 F1_9: 0.7400 F1_avg: 0.8226
Epoch  11 | Loss: 0.5432 | Acc18: 0.6882 F1_18: 0.7090 | F1_bin: 0.8960 F1_9: 0.7223 F1_avg: 0.8092
Epoch  12 | Loss: 0.5115 | Acc18: 0.6777 F1_18: 0.7135 | F1_bin: 0.8825 F1_9: 0.7329 F1_avg: 0.8077
Epoch  13 | Loss: 0.4830 | Acc18: 0.6980 F1_18: 0.7237 | F1_bin: 0.9144 F1_9: 0.7364 F1_avg: 0.8254
Epoch  14 | Loss: 0.4657 | Acc18: 0.6919 F1_18: 0.7286 | F1_bin: 0.9162 F1_9: 0.7393 F1_avg: 0.8278
Epoch  15 | Loss: 0.4431 | Acc18: 0.7146 F1_18: 0.7455 | F1_bin: 0.9105 F1_9: 0.7471 F1_avg: 0.8288
Epoch  16 | Loss: 0.4228 | Acc18: 0.7036 F1_18: 0.7431 | F1_bin: 0.9114 F1_9: 0.7506 F1_avg: 0.8310
Epoch  17 | Loss: 0.4013 | Acc18: 0.7189 F1_18: 0.7522 | F1_bin: 0.9200 F1_9: 0.7585 F1_avg: 0.8393
Epoch  18 | Loss: 0.3872 | Acc18: 0.7116 F1_18: 0.7392 | F1_bin: 0.9136 F1_9: 0.7524 F1_avg: 0.8330
Epoch  19 | Loss: 0.3819 | Acc18: 0.7165 F1_18: 0.7500 | F1_bin: 0.9188 F1_9: 0.7585 F1_avg: 0.8386
Epoch  20 | Loss: 0.3830 | Acc18: 0.7183 F1_18: 0.7536 | F1_bin: 0.9261 F1_9: 0.7572 F1_avg: 0.8417
Epoch  21 | Loss: 0.3531 | Acc18: 0.7146 F1_18: 0.7424 | F1_bin: 0.9182 F1_9: 0.7520 F1_avg: 0.8351
Epoch  22 | Loss: 0.3425 | Acc18: 0.7036 F1_18: 0.7383 | F1_bin: 0.9047 F1_9: 0.7559 F1_avg: 0.8303
Epoch  23 | Loss: 0.3227 | Acc18: 0.7177 F1_18: 0.7513 | F1_bin: 0.9163 F1_9: 0.7592 F1_avg: 0.8377
Epoch 00024: reducing learning rate of group 0 to 5.0000e-04.
Epoch  24 | Loss: 0.3023 | Acc18: 0.6980 F1_18: 0.7365 | F1_bin: 0.9089 F1_9: 0.7412 F1_avg: 0.8251
Epoch  25 | Loss: 0.2501 | Acc18: 0.7251 F1_18: 0.7591 | F1_bin: 0.9170 F1_9: 0.7673 F1_avg: 0.8421
Epoch  26 | Loss: 0.2161 | Acc18: 0.7337 F1_18: 0.7703 | F1_bin: 0.9193 F1_9: 0.7729 F1_avg: 0.8461
Epoch  27 | Loss: 0.2039 | Acc18: 0.7306 F1_18: 0.7643 | F1_bin: 0.9261 F1_9: 0.7703 F1_avg: 0.8482
Epoch  28 | Loss: 0.1877 | Acc18: 0.7263 F1_18: 0.7549 | F1_bin: 0.9231 F1_9: 0.7619 F1_avg: 0.8425
Epoch  29 | Loss: 0.1744 | Acc18: 0.7214 F1_18: 0.7587 | F1_bin: 0.9188 F1_9: 0.7646 F1_avg: 0.8417
Epoch 00030: reducing learning rate of group 0 to 2.5000e-04.
Epoch  30 | Loss: 0.1667 | Acc18: 0.7288 F1_18: 0.7642 | F1_bin: 0.9169 F1_9: 0.7676 F1_avg: 0.8423
Epoch  31 | Loss: 0.1503 | Acc18: 0.7349 F1_18: 0.7707 | F1_bin: 0.9267 F1_9: 0.7777 F1_avg: 0.8522
Epoch  32 | Loss: 0.1310 | Acc18: 0.7288 F1_18: 0.7671 | F1_bin: 0.9200 F1_9: 0.7752 F1_avg: 0.8476
Epoch  33 | Loss: 0.1243 | Acc18: 0.7368 F1_18: 0.7691 | F1_bin: 0.9266 F1_9: 0.7768 F1_avg: 0.8517
Epoch  34 | Loss: 0.1200 | Acc18: 0.7386 F1_18: 0.7728 | F1_bin: 0.9273 F1_9: 0.7789 F1_avg: 0.8531
Epoch  35 | Loss: 0.1259 | Acc18: 0.7349 F1_18: 0.7705 | F1_bin: 0.9266 F1_9: 0.7768 F1_avg: 0.8517
Epoch  36 | Loss: 0.1143 | Acc18: 0.7319 F1_18: 0.7654 | F1_bin: 0.9206 F1_9: 0.7706 F1_avg: 0.8456
Epoch  37 | Loss: 0.1107 | Acc18: 0.7257 F1_18: 0.7606 | F1_bin: 0.9266 F1_9: 0.7668 F1_avg: 0.8467
Epoch 00038: reducing learning rate of group 0 to 1.2500e-04.
Epoch  38 | Loss: 0.1110 | Acc18: 0.7226 F1_18: 0.7572 | F1_bin: 0.9182 F1_9: 0.7640 F1_avg: 0.8411
Epoch  39 | Loss: 0.0950 | Acc18: 0.7312 F1_18: 0.7631 | F1_bin: 0.9188 F1_9: 0.7730 F1_avg: 0.8459
Epoch  40 | Loss: 0.0854 | Acc18: 0.7392 F1_18: 0.7709 | F1_bin: 0.9267 F1_9: 0.7782 F1_avg: 0.8525
Epoch  41 | Loss: 0.0871 | Acc18: 0.7374 F1_18: 0.7700 | F1_bin: 0.9237 F1_9: 0.7778 F1_avg: 0.8507
Epoch 00042: reducing learning rate of group 0 to 6.2500e-05.
Epoch  42 | Loss: 0.0840 | Acc18: 0.7325 F1_18: 0.7671 | F1_bin: 0.9187 F1_9: 0.7786 F1_avg: 0.8486
Epoch  43 | Loss: 0.0766 | Acc18: 0.7343 F1_18: 0.7674 | F1_bin: 0.9224 F1_9: 0.7766 F1_avg: 0.8495
Epoch  44 | Loss: 0.0744 | Acc18: 0.7325 F1_18: 0.7638 | F1_bin: 0.9169 F1_9: 0.7763 F1_avg: 0.8466
Epoch  45 | Loss: 0.0703 | Acc18: 0.7337 F1_18: 0.7651 | F1_bin: 0.9212 F1_9: 0.7747 F1_avg: 0.8480
Epoch 00046: reducing learning rate of group 0 to 3.1250e-05.
Epoch  46 | Loss: 0.0749 | Acc18: 0.7399 F1_18: 0.7708 | F1_bin: 0.9249 F1_9: 0.7804 F1_avg: 0.8526
Epoch  47 | Loss: 0.0731 | Acc18: 0.7337 F1_18: 0.7636 | F1_bin: 0.9218 F1_9: 0.7737 F1_avg: 0.8477
Epoch  48 | Loss: 0.0719 | Acc18: 0.7355 F1_18: 0.7679 | F1_bin: 0.9224 F1_9: 0.7783 F1_avg: 0.8504
Epoch  49 | Loss: 0.0676 | Acc18: 0.7368 F1_18: 0.7665 | F1_bin: 0.9199 F1_9: 0.7781 F1_avg: 0.8490
Epoch 00050: reducing learning rate of group 0 to 1.5625e-05.
Epoch  50 | Loss: 0.0693 | Acc18: 0.7288 F1_18: 0.7581 | F1_bin: 0.9187 F1_9: 0.7684 F1_avg: 0.8436
Epoch  51 | Loss: 0.0676 | Acc18: 0.7399 F1_18: 0.7694 | F1_bin: 0.9249 F1_9: 0.7789 F1_avg: 0.8519
Epoch  52 | Loss: 0.0628 | Acc18: 0.7349 F1_18: 0.7652 | F1_bin: 0.9194 F1_9: 0.7745 F1_avg: 0.8469
Epoch  53 | Loss: 0.0646 | Acc18: 0.7368 F1_18: 0.7689 | F1_bin: 0.9243 F1_9: 0.7763 F1_avg: 0.8503
Epoch 00054: reducing learning rate of group 0 to 7.8125e-06.
Epoch  54 | Loss: 0.0672 | Acc18: 0.7337 F1_18: 0.7661 | F1_bin: 0.9236 F1_9: 0.7737 F1_avg: 0.8486
Early stopping.

=== Fold 5 ===
Epoch   1 | Loss: 1.6960 | Acc18: 0.5162 F1_18: 0.5326 | F1_bin: 0.8219 F1_9: 0.5474 F1_avg: 0.6847
Epoch   2 | Loss: 1.1237 | Acc18: 0.5749 F1_18: 0.6017 | F1_bin: 0.8489 F1_9: 0.6226 F1_avg: 0.7357
Epoch   3 | Loss: 0.9406 | Acc18: 0.6349 F1_18: 0.6440 | F1_bin: 0.8939 F1_9: 0.6669 F1_avg: 0.7804
Epoch   4 | Loss: 0.8644 | Acc18: 0.6294 F1_18: 0.6419 | F1_bin: 0.8819 F1_9: 0.6620 F1_avg: 0.7719
Epoch   5 | Loss: 0.7752 | Acc18: 0.6404 F1_18: 0.6600 | F1_bin: 0.8985 F1_9: 0.6680 F1_avg: 0.7832
Epoch   6 | Loss: 0.7103 | Acc18: 0.6673 F1_18: 0.6766 | F1_bin: 0.9019 F1_9: 0.6925 F1_avg: 0.7972
Epoch   7 | Loss: 0.6883 | Acc18: 0.6789 F1_18: 0.6934 | F1_bin: 0.9124 F1_9: 0.7107 F1_avg: 0.8115
Epoch   8 | Loss: 0.6280 | Acc18: 0.6685 F1_18: 0.7023 | F1_bin: 0.9012 F1_9: 0.7132 F1_avg: 0.8072
Epoch   9 | Loss: 0.5956 | Acc18: 0.6165 F1_18: 0.6527 | F1_bin: 0.8942 F1_9: 0.6605 F1_avg: 0.7773
Epoch  10 | Loss: 0.5541 | Acc18: 0.6856 F1_18: 0.7111 | F1_bin: 0.9050 F1_9: 0.7213 F1_avg: 0.8131
Epoch  11 | Loss: 0.5314 | Acc18: 0.6899 F1_18: 0.7180 | F1_bin: 0.9110 F1_9: 0.7265 F1_avg: 0.8188
Epoch  12 | Loss: 0.5026 | Acc18: 0.6954 F1_18: 0.7135 | F1_bin: 0.9123 F1_9: 0.7235 F1_avg: 0.8179
Epoch  13 | Loss: 0.4850 | Acc18: 0.6624 F1_18: 0.6912 | F1_bin: 0.8820 F1_9: 0.7195 F1_avg: 0.8007
Epoch  14 | Loss: 0.4572 | Acc18: 0.6905 F1_18: 0.7175 | F1_bin: 0.9137 F1_9: 0.7266 F1_avg: 0.8201
Epoch 00015: reducing learning rate of group 0 to 5.0000e-04.
Epoch  15 | Loss: 0.4428 | Acc18: 0.6752 F1_18: 0.6882 | F1_bin: 0.8920 F1_9: 0.7122 F1_avg: 0.8021
Epoch  16 | Loss: 0.3640 | Acc18: 0.7015 F1_18: 0.7346 | F1_bin: 0.9063 F1_9: 0.7478 F1_avg: 0.8270
Epoch  17 | Loss: 0.3424 | Acc18: 0.7199 F1_18: 0.7458 | F1_bin: 0.9227 F1_9: 0.7562 F1_avg: 0.8395
Epoch  18 | Loss: 0.3071 | Acc18: 0.7021 F1_18: 0.7339 | F1_bin: 0.9112 F1_9: 0.7435 F1_avg: 0.8273
Epoch  19 | Loss: 0.2785 | Acc18: 0.7076 F1_18: 0.7406 | F1_bin: 0.9112 F1_9: 0.7537 F1_avg: 0.8324
Epoch  20 | Loss: 0.2755 | Acc18: 0.7107 F1_18: 0.7428 | F1_bin: 0.9216 F1_9: 0.7528 F1_avg: 0.8372
Epoch 00021: reducing learning rate of group 0 to 2.5000e-04.
Epoch  21 | Loss: 0.2652 | Acc18: 0.7089 F1_18: 0.7342 | F1_bin: 0.9142 F1_9: 0.7418 F1_avg: 0.8280
Epoch  22 | Loss: 0.2236 | Acc18: 0.7076 F1_18: 0.7427 | F1_bin: 0.9125 F1_9: 0.7549 F1_avg: 0.8337
Epoch  23 | Loss: 0.2076 | Acc18: 0.7058 F1_18: 0.7409 | F1_bin: 0.9155 F1_9: 0.7532 F1_avg: 0.8343
Epoch  24 | Loss: 0.2025 | Acc18: 0.7058 F1_18: 0.7424 | F1_bin: 0.9119 F1_9: 0.7521 F1_avg: 0.8320
Epoch 00025: reducing learning rate of group 0 to 1.2500e-04.
Epoch  25 | Loss: 0.2040 | Acc18: 0.7119 F1_18: 0.7379 | F1_bin: 0.9143 F1_9: 0.7504 F1_avg: 0.8323
Epoch  26 | Loss: 0.1762 | Acc18: 0.7180 F1_18: 0.7485 | F1_bin: 0.9185 F1_9: 0.7630 F1_avg: 0.8408
Epoch  27 | Loss: 0.1606 | Acc18: 0.7291 F1_18: 0.7596 | F1_bin: 0.9264 F1_9: 0.7702 F1_avg: 0.8483
Epoch  28 | Loss: 0.1591 | Acc18: 0.7180 F1_18: 0.7477 | F1_bin: 0.9209 F1_9: 0.7587 F1_avg: 0.8398
Epoch  29 | Loss: 0.1539 | Acc18: 0.7174 F1_18: 0.7432 | F1_bin: 0.9222 F1_9: 0.7547 F1_avg: 0.8384
Epoch  30 | Loss: 0.1561 | Acc18: 0.7107 F1_18: 0.7423 | F1_bin: 0.9155 F1_9: 0.7557 F1_avg: 0.8356
Epoch 00031: reducing learning rate of group 0 to 6.2500e-05.
Epoch  31 | Loss: 0.1466 | Acc18: 0.7144 F1_18: 0.7445 | F1_bin: 0.9191 F1_9: 0.7552 F1_avg: 0.8371
Epoch  32 | Loss: 0.1342 | Acc18: 0.7168 F1_18: 0.7483 | F1_bin: 0.9215 F1_9: 0.7576 F1_avg: 0.8396
Epoch  33 | Loss: 0.1361 | Acc18: 0.7144 F1_18: 0.7393 | F1_bin: 0.9197 F1_9: 0.7519 F1_avg: 0.8358
Epoch  34 | Loss: 0.1383 | Acc18: 0.7168 F1_18: 0.7498 | F1_bin: 0.9197 F1_9: 0.7583 F1_avg: 0.8390
Epoch 00035: reducing learning rate of group 0 to 3.1250e-05.
Epoch  35 | Loss: 0.1257 | Acc18: 0.7168 F1_18: 0.7440 | F1_bin: 0.9228 F1_9: 0.7566 F1_avg: 0.8397
Epoch  36 | Loss: 0.1277 | Acc18: 0.7131 F1_18: 0.7399 | F1_bin: 0.9185 F1_9: 0.7544 F1_avg: 0.8365
Epoch  37 | Loss: 0.1240 | Acc18: 0.7144 F1_18: 0.7446 | F1_bin: 0.9185 F1_9: 0.7539 F1_avg: 0.8362
Epoch  38 | Loss: 0.1145 | Acc18: 0.7150 F1_18: 0.7442 | F1_bin: 0.9216 F1_9: 0.7546 F1_avg: 0.8381
Epoch 00039: reducing learning rate of group 0 to 1.5625e-05.
Epoch  39 | Loss: 0.1129 | Acc18: 0.7193 F1_18: 0.7502 | F1_bin: 0.9173 F1_9: 0.7613 F1_avg: 0.8393
Epoch  40 | Loss: 0.1154 | Acc18: 0.7187 F1_18: 0.7503 | F1_bin: 0.9210 F1_9: 0.7614 F1_avg: 0.8412
Epoch  41 | Loss: 0.1137 | Acc18: 0.7119 F1_18: 0.7417 | F1_bin: 0.9154 F1_9: 0.7546 F1_avg: 0.8350
Epoch  42 | Loss: 0.1148 | Acc18: 0.7119 F1_18: 0.7462 | F1_bin: 0.9203 F1_9: 0.7575 F1_avg: 0.8389
Epoch 00043: reducing learning rate of group 0 to 7.8125e-06.
Epoch  43 | Loss: 0.1078 | Acc18: 0.7174 F1_18: 0.7475 | F1_bin: 0.9185 F1_9: 0.7588 F1_avg: 0.8386
Epoch  44 | Loss: 0.1093 | Acc18: 0.7162 F1_18: 0.7480 | F1_bin: 0.9215 F1_9: 0.7589 F1_avg: 0.8402
Epoch  45 | Loss: 0.1077 | Acc18: 0.7205 F1_18: 0.7489 | F1_bin: 0.9203 F1_9: 0.7603 F1_avg: 0.8403
Epoch  46 | Loss: 0.1080 | Acc18: 0.7223 F1_18: 0.7531 | F1_bin: 0.9203 F1_9: 0.7627 F1_avg: 0.8415
Epoch 00047: reducing learning rate of group 0 to 3.9063e-06.
Epoch  47 | Loss: 0.1041 | Acc18: 0.7242 F1_18: 0.7549 | F1_bin: 0.9228 F1_9: 0.7641 F1_avg: 0.8434
Early stopping.
"""
