#ModelVariant_LSTMGRU 特徴量に角速度を追加。シーケンスをまたがないように修正　IMUonlyモデル
#ミックスアップを追加
#CV=0.803
#CMI 2025 デモ提出 バージョン73　IMUonly+all(IMUモデルのみミックスアップ) LB=0.82

import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.transform import Rotation as R

# ========= 設定 =========
RAW_CSV = "train.csv"
SAVE_DIR = "train_test_18class_36"
PAD_PERCENTILE = 95
BATCH_SIZE = 64
EPOCHS = 100
LR_INIT = 1e-3
WD = 1e-4
PATIENCE = 20

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= サポート関数 =========
def to_binary(y): return np.array([0 if i<9 else 1 for i in y])
def to_9class(y): return np.array([i%9 for i in y])

# ========= 特徴量エンジニアリング =========
def remove_gravity_from_acc(acc_data, rot_data):
    acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    linear_accel = np.zeros_like(acc_values)
    gravity_world = np.array([0, 0, 9.81])

    for i in range(acc_values.shape[0]):
        if np.any(np.isnan(quat_values[i])):
            linear_accel[i] = acc_values[i]
            continue
        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i] = acc_values[i] - gravity_sensor_frame
        except ValueError:
            linear_accel[i] = acc_values[i]
    return linear_accel

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

def feature_eng(df):
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1,1))
    df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)

    def calc_linear_acc(sub_df):
        linear_acc = remove_gravity_from_acc(sub_df[['acc_x', 'acc_y', 'acc_z']], sub_df[['rot_x', 'rot_y', 'rot_z', 'rot_w']])
        return pd.DataFrame(linear_acc, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=sub_df.index)
    linear_acc_df = df.groupby('sequence_id').apply(calc_linear_acc).reset_index(level=0, drop=True)
    df = pd.concat([df, linear_acc_df], axis=1)

    df['linear_acc_mag'] = np.sqrt(df['linear_acc_x']**2 + df['linear_acc_y']**2 + df['linear_acc_z']**2)
    df['linear_acc_mag_jerk'] = df.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)

    def calc_ang(sub_df):
        rot_data = sub_df[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        ang_vel = calculate_angular_velocity_from_quat(rot_data)
        ang_dist = calculate_angular_distance(rot_data)
        return pd.DataFrame(
            np.hstack([ang_vel, ang_dist.reshape(-1,1)]),
            columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_dist'],
            index=sub_df.index
        )
    ang_df = df.groupby('sequence_id').apply(calc_ang).reset_index(level=0, drop=True)
    df = pd.concat([df, ang_df], axis=1)
    return df

# ========= データセット =========
class IMUDataset(Dataset):
    def __init__(self, X, y, mixup=False, alpha=0.4):
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.mixup = mixup
        self.alpha = alpha

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x, y = self.X[i], self.y[i]
        if self.mixup and np.random.rand() < 0.5:
            j = np.random.randint(0, len(self.X))
            lam = np.random.beta(self.alpha, self.alpha)
            x = lam * x + (1 - lam) * self.X[j]
            y_onehot = F.one_hot(y, num_classes=18).float()
            yj_onehot = F.one_hot(self.y[j], num_classes=18).float()
            y = lam * y_onehot + (1 - lam) * yj_onehot
            return x, y
        else:
            return x, F.one_hot(y, num_classes=18).float()

# ========= モデル定義 =========
# （ここは先ほどの MultiScaleConv1d, SEBlock, ResidualSEBlock, AttentionLayer, MetaFeatureExtractor, GaussianNoise, ModelVariant_LSTMGRU をそのまま使う）
# ...
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
        y = self.pool(x).view(x.size(0), -1)
        y = self.fc(y).view(x.size(0), -1, 1)
        return x * y

class ResidualSEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, pool=2, drop=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=k//2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=k//2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.bn_sc = nn.BatchNorm1d(out_ch) if in_ch != out_ch else nn.Identity()
        self.se = SEBlock(out_ch)
        self.pool = nn.MaxPool1d(pool)
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
        w = torch.softmax(torch.tanh(self.fc(x)).squeeze(-1), dim=1).unsqueeze(-1)
        return (x * w).sum(dim=1)

class MetaFeatureExtractor(nn.Module):
    def forward(self, x):
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        maxv, _ = x.max(dim=1)
        minv, _ = x.min(dim=1)
        slope = (x[:, -1, :] - x[:, 0, :]) / max(x.size(1)-1,1)
        return torch.cat([mean, std, maxv, minv, slope], dim=1)

class GaussianNoise(nn.Module):
    def __init__(self, std=0.09):
        super().__init__()
        self.std = std
    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x

class ModelVariant_LSTMGRU(nn.Module):
    def __init__(self, imu_dim, num_classes):
        super().__init__()
        self.meta = MetaFeatureExtractor()
        self.meta_dense = nn.Sequential(
            nn.Linear(5*imu_dim,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.branches = nn.ModuleList([
            nn.Sequential(
                MultiScaleConv1d(1,12),
                ResidualSEBlock(36,48),
                ResidualSEBlock(48,48)
            ) for _ in range(imu_dim)
        ])
        self.bigru = nn.GRU(48*imu_dim,128,batch_first=True,bidirectional=True,num_layers=2,dropout=0.2)
        self.bilstm = nn.LSTM(48*imu_dim,128,batch_first=True,bidirectional=True,num_layers=2,dropout=0.2)
        self.noise = GaussianNoise(0.09)
        self.attn = AttentionLayer(256+256+48*imu_dim)
        self.head_1 = nn.Sequential(
            nn.Linear(256+256+48*imu_dim+32,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes)
        )
        self.head_2 = nn.Sequential(
            nn.Linear(256+256+48*imu_dim+32,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,1)
        )
    def forward(self,x):
        meta = self.meta_dense(self.meta(x))
        branches = [branch(x[:,:,i].unsqueeze(1)).transpose(1,2) for i,branch in enumerate(self.branches)]
        combined = torch.cat(branches, dim=2)
        gru_out,_ = self.bigru(combined)
        lstm_out,_ = self.bilstm(combined)
        noise = self.noise(combined)
        pooled = self.attn(torch.cat([gru_out,lstm_out,noise],dim=2))
        fused = torch.cat([pooled,meta],dim=1)
        return self.head_1(fused), self.head_2(fused)

# ========= 学習 =========
df = pd.read_csv(RAW_CSV)
df = feature_eng(df)
df["gesture"] = df["gesture"].fillna("unknown")
le = LabelEncoder()
df["gesture_class"] = le.fit_transform(df["gesture"])

imu_cols = [c for c in df.select_dtypes(include=np.number).columns
            if c.startswith(('acc_','rot_','acc_mag','rot_angle','acc_mag_jerk',
                             'rot_angle_vel','angular_vel','angular_dist')) and c!="gesture_class"]

imu_dim = len(imu_cols)
pad_len = int(np.percentile(df.groupby("sequence_id").size().values, PAD_PERCENTILE))

def prepare(ids, df, scaler):
    X,y = [],[]
    for sid in ids:
        m = scaler.transform(df[df["sequence_id"]==sid][imu_cols].ffill().bfill().fillna(0))
        m = np.pad(m,((0,max(0,pad_len-len(m))),(0,0)))
        X.append(m[:pad_len])
        y.append(df[df["sequence_id"]==sid]["gesture_class"].iloc[0])
    return X,y

kf = GroupKFold(n_splits=5)
seq_ids = df["sequence_id"].unique()
subject_map = df.drop_duplicates("sequence_id").set_index("sequence_id")["subject"]
groups = [subject_map[sid] for sid in seq_ids]

for fold, (tr_idx, va_idx) in enumerate(kf.split(seq_ids, groups=groups)):
    print(f"\n=== Fold {fold+1} ===")
    fold_dir = os.path.join(SAVE_DIR, f"fold{fold+1}")
    os.makedirs(fold_dir, exist_ok=True)
    train_df,val_df = df[df["sequence_id"].isin(seq_ids[tr_idx])],df[df["sequence_id"].isin(seq_ids[va_idx])]
    scaler = StandardScaler().fit(train_df[imu_cols].fillna(0))
    joblib.dump(scaler, os.path.join(fold_dir,"scaler.pkl"))
    X_train,y_train = prepare(seq_ids[tr_idx], train_df, scaler)
    X_val,y_val = prepare(seq_ids[va_idx], val_df, scaler)

    train_loader = DataLoader(IMUDataset(X_train, y_train, mixup=True, alpha=0.4), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(IMUDataset(X_val, y_val, mixup=False), batch_size=BATCH_SIZE)

    model = ModelVariant_LSTMGRU(imu_dim, len(le.classes_)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5, verbose=True)
    best_f1_avg, patience_counter = 0,0

    def soft_cross_entropy(preds, soft_targets):
        log_probs = F.log_softmax(preds, dim=1)
        return -(soft_targets * log_probs).sum(dim=1).mean()

    for epoch in range(EPOCHS):
        model.train()
        loss_sum=0
        for xb,yb in train_loader:
            xb,yb=xb.to(device),yb.to(device)
            opt.zero_grad()
            logits,_ = model(xb)
            loss = soft_cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item()*len(xb)

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits,_ = model(xb)
                preds.extend(logits.argmax(1).cpu().numpy())
                trues.extend(torch.argmax(yb, dim=1).numpy())

        acc_18 = accuracy_score(trues, preds)
        f1_18 = f1_score(trues, preds, average="macro")
        acc_bin = accuracy_score(to_binary(trues), to_binary(preds))
        f1_bin = f1_score(to_binary(trues), to_binary(preds), average="macro")
        acc_9 = accuracy_score(to_9class(trues), to_9class(preds))
        f1_9 = f1_score(to_9class(trues), to_9class(preds), average="macro")
        f1_avg = (f1_bin + f1_9) / 2

        sched.step(1 - f1_18)

        print(
            f"Epoch {epoch+1:3d} | "
            f"Loss: {loss_sum / len(train_loader.dataset):.4f} | "
            f"Acc18: {acc_18:.4f} F1_18: {f1_18:.4f} | "
            f"F1_bin: {f1_bin:.4f} F1_9: {f1_9:.4f} F1_avg: {f1_avg:.4f}",
            flush=True
        )

        if f1_avg > best_f1_avg:
            best_f1_avg, patience_counter = f1_avg, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "imu_dim": imu_dim,
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
Epoch   1 | Loss: 2.0346 | Acc18: 0.4265 F1_18: 0.4469 | F1_bin: 0.7619 F1_9: 0.4887 F1_avg: 0.6253
Epoch   2 | Loss: 1.6790 | Acc18: 0.5288 F1_18: 0.5800 | F1_bin: 0.8155 F1_9: 0.6014 F1_avg: 0.7085
Epoch   3 | Loss: 1.5260 | Acc18: 0.4963 F1_18: 0.5094 | F1_bin: 0.8051 F1_9: 0.5441 F1_avg: 0.6746
Epoch   4 | Loss: 1.4726 | Acc18: 0.5895 F1_18: 0.6253 | F1_bin: 0.8507 F1_9: 0.6555 F1_avg: 0.7531
Epoch   5 | Loss: 1.4071 | Acc18: 0.5846 F1_18: 0.6165 | F1_bin: 0.8582 F1_9: 0.6341 F1_avg: 0.7462
Epoch   6 | Loss: 1.3546 | Acc18: 0.6036 F1_18: 0.6265 | F1_bin: 0.8603 F1_9: 0.6554 F1_avg: 0.7579
Epoch   7 | Loss: 1.3298 | Acc18: 0.6317 F1_18: 0.6665 | F1_bin: 0.8636 F1_9: 0.6860 F1_avg: 0.7748
Epoch   8 | Loss: 1.3130 | Acc18: 0.6336 F1_18: 0.6727 | F1_bin: 0.8667 F1_9: 0.6891 F1_avg: 0.7779
Epoch   9 | Loss: 1.2665 | Acc18: 0.6115 F1_18: 0.6496 | F1_bin: 0.8717 F1_9: 0.6775 F1_avg: 0.7746
Epoch  10 | Loss: 1.2397 | Acc18: 0.6305 F1_18: 0.6626 | F1_bin: 0.8577 F1_9: 0.6945 F1_avg: 0.7761
Epoch  11 | Loss: 1.2329 | Acc18: 0.6403 F1_18: 0.6879 | F1_bin: 0.8632 F1_9: 0.7062 F1_avg: 0.7847
Epoch  12 | Loss: 1.1917 | Acc18: 0.6415 F1_18: 0.6696 | F1_bin: 0.8615 F1_9: 0.7004 F1_avg: 0.7810
Epoch  13 | Loss: 1.1845 | Acc18: 0.6489 F1_18: 0.6815 | F1_bin: 0.8757 F1_9: 0.7016 F1_avg: 0.7887
Epoch  14 | Loss: 1.1528 | Acc18: 0.6287 F1_18: 0.6595 | F1_bin: 0.8632 F1_9: 0.6850 F1_avg: 0.7741
Epoch  15 | Loss: 1.1559 | Acc18: 0.6667 F1_18: 0.7132 | F1_bin: 0.8716 F1_9: 0.7334 F1_avg: 0.8025
Epoch  16 | Loss: 1.1019 | Acc18: 0.6477 F1_18: 0.6999 | F1_bin: 0.8710 F1_9: 0.7121 F1_avg: 0.7915
Epoch  17 | Loss: 1.1034 | Acc18: 0.6373 F1_18: 0.6541 | F1_bin: 0.8799 F1_9: 0.6901 F1_avg: 0.7850
Epoch  18 | Loss: 1.0866 | Acc18: 0.6434 F1_18: 0.6863 | F1_bin: 0.8656 F1_9: 0.7051 F1_avg: 0.7854
Epoch  19 | Loss: 1.0801 | Acc18: 0.6636 F1_18: 0.7155 | F1_bin: 0.8671 F1_9: 0.7343 F1_avg: 0.8007
Epoch  20 | Loss: 1.0718 | Acc18: 0.6857 F1_18: 0.7337 | F1_bin: 0.8854 F1_9: 0.7543 F1_avg: 0.8199
Epoch  21 | Loss: 1.0498 | Acc18: 0.6624 F1_18: 0.6913 | F1_bin: 0.8756 F1_9: 0.7225 F1_avg: 0.7990
Epoch  22 | Loss: 1.0215 | Acc18: 0.6605 F1_18: 0.6946 | F1_bin: 0.8670 F1_9: 0.7208 F1_avg: 0.7939
Epoch  23 | Loss: 1.0106 | Acc18: 0.6801 F1_18: 0.7257 | F1_bin: 0.8913 F1_9: 0.7373 F1_avg: 0.8143
Epoch 00024: reducing learning rate of group 0 to 5.0000e-04.
Epoch  24 | Loss: 1.0008 | Acc18: 0.6814 F1_18: 0.7282 | F1_bin: 0.8835 F1_9: 0.7395 F1_avg: 0.8115
Epoch  25 | Loss: 0.9283 | Acc18: 0.6973 F1_18: 0.7375 | F1_bin: 0.8914 F1_9: 0.7491 F1_avg: 0.8203
Epoch  26 | Loss: 0.9068 | Acc18: 0.6949 F1_18: 0.7320 | F1_bin: 0.8930 F1_9: 0.7456 F1_avg: 0.8193
Epoch  27 | Loss: 0.8712 | Acc18: 0.7016 F1_18: 0.7422 | F1_bin: 0.8883 F1_9: 0.7561 F1_avg: 0.8222
Epoch  28 | Loss: 0.8769 | Acc18: 0.6912 F1_18: 0.7416 | F1_bin: 0.8955 F1_9: 0.7457 F1_avg: 0.8206
Epoch  29 | Loss: 0.8425 | Acc18: 0.6918 F1_18: 0.7316 | F1_bin: 0.8930 F1_9: 0.7407 F1_avg: 0.8169
Epoch  30 | Loss: 0.8272 | Acc18: 0.6893 F1_18: 0.7412 | F1_bin: 0.8859 F1_9: 0.7501 F1_avg: 0.8180
Epoch  31 | Loss: 0.8311 | Acc18: 0.7028 F1_18: 0.7475 | F1_bin: 0.8915 F1_9: 0.7571 F1_avg: 0.8243
Epoch  32 | Loss: 0.7954 | Acc18: 0.6979 F1_18: 0.7495 | F1_bin: 0.8960 F1_9: 0.7529 F1_avg: 0.8244
Epoch  33 | Loss: 0.7815 | Acc18: 0.6949 F1_18: 0.7437 | F1_bin: 0.8908 F1_9: 0.7506 F1_avg: 0.8207
Epoch  34 | Loss: 0.7897 | Acc18: 0.6942 F1_18: 0.7443 | F1_bin: 0.8885 F1_9: 0.7494 F1_avg: 0.8190
Epoch  35 | Loss: 0.7850 | Acc18: 0.7083 F1_18: 0.7593 | F1_bin: 0.8985 F1_9: 0.7671 F1_avg: 0.8328
Epoch  36 | Loss: 0.7759 | Acc18: 0.7077 F1_18: 0.7546 | F1_bin: 0.8988 F1_9: 0.7625 F1_avg: 0.8306
Epoch  37 | Loss: 0.7553 | Acc18: 0.6936 F1_18: 0.7406 | F1_bin: 0.8872 F1_9: 0.7524 F1_avg: 0.8198
Epoch  38 | Loss: 0.7611 | Acc18: 0.6936 F1_18: 0.7376 | F1_bin: 0.8963 F1_9: 0.7451 F1_avg: 0.8207
Epoch 00039: reducing learning rate of group 0 to 2.5000e-04.
Epoch  39 | Loss: 0.7339 | Acc18: 0.6955 F1_18: 0.7426 | F1_bin: 0.8957 F1_9: 0.7527 F1_avg: 0.8242
Epoch  40 | Loss: 0.7090 | Acc18: 0.7059 F1_18: 0.7507 | F1_bin: 0.9017 F1_9: 0.7629 F1_avg: 0.8323
Epoch  41 | Loss: 0.6827 | Acc18: 0.7047 F1_18: 0.7542 | F1_bin: 0.8972 F1_9: 0.7635 F1_avg: 0.8304
Epoch  42 | Loss: 0.6698 | Acc18: 0.7034 F1_18: 0.7507 | F1_bin: 0.9012 F1_9: 0.7585 F1_avg: 0.8299
Epoch 00043: reducing learning rate of group 0 to 1.2500e-04.
Epoch  43 | Loss: 0.6692 | Acc18: 0.7083 F1_18: 0.7584 | F1_bin: 0.8981 F1_9: 0.7602 F1_avg: 0.8292
Epoch  44 | Loss: 0.6439 | Acc18: 0.6967 F1_18: 0.7492 | F1_bin: 0.8971 F1_9: 0.7541 F1_avg: 0.8256
Epoch  45 | Loss: 0.6507 | Acc18: 0.7053 F1_18: 0.7582 | F1_bin: 0.9037 F1_9: 0.7575 F1_avg: 0.8306
Epoch  46 | Loss: 0.6470 | Acc18: 0.7102 F1_18: 0.7641 | F1_bin: 0.9013 F1_9: 0.7673 F1_avg: 0.8343
Epoch  47 | Loss: 0.6192 | Acc18: 0.7188 F1_18: 0.7663 | F1_bin: 0.9033 F1_9: 0.7707 F1_avg: 0.8370
Epoch  48 | Loss: 0.6135 | Acc18: 0.7065 F1_18: 0.7567 | F1_bin: 0.8992 F1_9: 0.7620 F1_avg: 0.8306
Epoch  49 | Loss: 0.6011 | Acc18: 0.7126 F1_18: 0.7643 | F1_bin: 0.9005 F1_9: 0.7684 F1_avg: 0.8344
Epoch  50 | Loss: 0.6252 | Acc18: 0.7120 F1_18: 0.7613 | F1_bin: 0.9024 F1_9: 0.7676 F1_avg: 0.8350
Epoch 00051: reducing learning rate of group 0 to 6.2500e-05.
Epoch  51 | Loss: 0.6116 | Acc18: 0.7114 F1_18: 0.7572 | F1_bin: 0.8971 F1_9: 0.7669 F1_avg: 0.8320
Epoch  52 | Loss: 0.6062 | Acc18: 0.7132 F1_18: 0.7614 | F1_bin: 0.9014 F1_9: 0.7675 F1_avg: 0.8345
Epoch  53 | Loss: 0.5896 | Acc18: 0.7126 F1_18: 0.7606 | F1_bin: 0.9056 F1_9: 0.7654 F1_avg: 0.8355
Epoch  54 | Loss: 0.6050 | Acc18: 0.7071 F1_18: 0.7524 | F1_bin: 0.9030 F1_9: 0.7595 F1_avg: 0.8312
Epoch 00055: reducing learning rate of group 0 to 3.1250e-05.
Epoch  55 | Loss: 0.5915 | Acc18: 0.7077 F1_18: 0.7566 | F1_bin: 0.8998 F1_9: 0.7603 F1_avg: 0.8301
Epoch  56 | Loss: 0.5870 | Acc18: 0.7047 F1_18: 0.7520 | F1_bin: 0.9005 F1_9: 0.7577 F1_avg: 0.8291
Epoch  57 | Loss: 0.5803 | Acc18: 0.7083 F1_18: 0.7567 | F1_bin: 0.9018 F1_9: 0.7629 F1_avg: 0.8324
Epoch  58 | Loss: 0.5993 | Acc18: 0.7102 F1_18: 0.7557 | F1_bin: 0.9049 F1_9: 0.7627 F1_avg: 0.8338
Epoch 00059: reducing learning rate of group 0 to 1.5625e-05.
Epoch  59 | Loss: 0.5817 | Acc18: 0.7126 F1_18: 0.7599 | F1_bin: 0.9049 F1_9: 0.7673 F1_avg: 0.8361
Epoch  60 | Loss: 0.5911 | Acc18: 0.7132 F1_18: 0.7625 | F1_bin: 0.9037 F1_9: 0.7687 F1_avg: 0.8362
Epoch  61 | Loss: 0.5644 | Acc18: 0.7102 F1_18: 0.7569 | F1_bin: 0.9044 F1_9: 0.7628 F1_avg: 0.8336
Epoch  62 | Loss: 0.5806 | Acc18: 0.7132 F1_18: 0.7600 | F1_bin: 0.9051 F1_9: 0.7650 F1_avg: 0.8350
Epoch 00063: reducing learning rate of group 0 to 7.8125e-06.
Epoch  63 | Loss: 0.5950 | Acc18: 0.7126 F1_18: 0.7588 | F1_bin: 0.9056 F1_9: 0.7631 F1_avg: 0.8344
Epoch  64 | Loss: 0.5842 | Acc18: 0.7132 F1_18: 0.7574 | F1_bin: 0.9037 F1_9: 0.7657 F1_avg: 0.8347
Epoch  65 | Loss: 0.5872 | Acc18: 0.7102 F1_18: 0.7564 | F1_bin: 0.9031 F1_9: 0.7646 F1_avg: 0.8338
Epoch  66 | Loss: 0.5914 | Acc18: 0.7102 F1_18: 0.7571 | F1_bin: 0.9017 F1_9: 0.7664 F1_avg: 0.8341
Epoch 00067: reducing learning rate of group 0 to 3.9063e-06.
Epoch  67 | Loss: 0.5994 | Acc18: 0.7126 F1_18: 0.7612 | F1_bin: 0.9037 F1_9: 0.7666 F1_avg: 0.8351
Early stopping.

=== Fold 2 ===
Epoch   1 | Loss: 1.9935 | Acc18: 0.4224 F1_18: 0.4103 | F1_bin: 0.7626 F1_9: 0.4710 F1_avg: 0.6168
Epoch   2 | Loss: 1.6383 | Acc18: 0.4893 F1_18: 0.5159 | F1_bin: 0.7984 F1_9: 0.5497 F1_avg: 0.6741
Epoch   3 | Loss: 1.4952 | Acc18: 0.5254 F1_18: 0.5499 | F1_bin: 0.8187 F1_9: 0.5865 F1_avg: 0.7026
Epoch   4 | Loss: 1.4150 | Acc18: 0.5579 F1_18: 0.5850 | F1_bin: 0.8116 F1_9: 0.6211 F1_avg: 0.7163
Epoch   5 | Loss: 1.3843 | Acc18: 0.5156 F1_18: 0.5492 | F1_bin: 0.8274 F1_9: 0.5684 F1_avg: 0.6979
Epoch   6 | Loss: 1.3100 | Acc18: 0.5530 F1_18: 0.5932 | F1_bin: 0.8124 F1_9: 0.6257 F1_avg: 0.7191
Epoch   7 | Loss: 1.2816 | Acc18: 0.5622 F1_18: 0.5937 | F1_bin: 0.8099 F1_9: 0.6366 F1_avg: 0.7232
Epoch   8 | Loss: 1.2605 | Acc18: 0.5745 F1_18: 0.6118 | F1_bin: 0.8294 F1_9: 0.6439 F1_avg: 0.7367
Epoch   9 | Loss: 1.2225 | Acc18: 0.5868 F1_18: 0.6226 | F1_bin: 0.8398 F1_9: 0.6553 F1_avg: 0.7476
Epoch  10 | Loss: 1.2190 | Acc18: 0.5727 F1_18: 0.6141 | F1_bin: 0.8301 F1_9: 0.6463 F1_avg: 0.7382
Epoch  11 | Loss: 1.1721 | Acc18: 0.5843 F1_18: 0.6386 | F1_bin: 0.8383 F1_9: 0.6643 F1_avg: 0.7513
Epoch  12 | Loss: 1.1577 | Acc18: 0.5788 F1_18: 0.6292 | F1_bin: 0.8270 F1_9: 0.6520 F1_avg: 0.7395
Epoch  13 | Loss: 1.1332 | Acc18: 0.5898 F1_18: 0.6429 | F1_bin: 0.8501 F1_9: 0.6639 F1_avg: 0.7570
Epoch  14 | Loss: 1.1038 | Acc18: 0.6015 F1_18: 0.6501 | F1_bin: 0.8418 F1_9: 0.6792 F1_avg: 0.7605
Epoch  15 | Loss: 1.0922 | Acc18: 0.5861 F1_18: 0.6282 | F1_bin: 0.8357 F1_9: 0.6651 F1_avg: 0.7504
Epoch  16 | Loss: 1.0772 | Acc18: 0.5739 F1_18: 0.6135 | F1_bin: 0.8028 F1_9: 0.6499 F1_avg: 0.7263
Epoch  17 | Loss: 1.0381 | Acc18: 0.6009 F1_18: 0.6552 | F1_bin: 0.8338 F1_9: 0.6796 F1_avg: 0.7567
Epoch  18 | Loss: 1.0545 | Acc18: 0.6070 F1_18: 0.6568 | F1_bin: 0.8301 F1_9: 0.6854 F1_avg: 0.7577
Epoch  19 | Loss: 1.0215 | Acc18: 0.5880 F1_18: 0.6508 | F1_bin: 0.8491 F1_9: 0.6662 F1_avg: 0.7577
Epoch  20 | Loss: 1.0196 | Acc18: 0.6217 F1_18: 0.6633 | F1_bin: 0.8545 F1_9: 0.6897 F1_avg: 0.7721
Epoch  21 | Loss: 0.9824 | Acc18: 0.5947 F1_18: 0.6491 | F1_bin: 0.8460 F1_9: 0.6729 F1_avg: 0.7595
Epoch  22 | Loss: 0.9870 | Acc18: 0.5978 F1_18: 0.6450 | F1_bin: 0.8490 F1_9: 0.6718 F1_avg: 0.7604
Epoch  23 | Loss: 0.9724 | Acc18: 0.6076 F1_18: 0.6571 | F1_bin: 0.8405 F1_9: 0.6792 F1_avg: 0.7598
Epoch 00024: reducing learning rate of group 0 to 5.0000e-04.
Epoch  24 | Loss: 0.9838 | Acc18: 0.6039 F1_18: 0.6563 | F1_bin: 0.8410 F1_9: 0.6825 F1_avg: 0.7617
Epoch  25 | Loss: 0.9142 | Acc18: 0.6260 F1_18: 0.6851 | F1_bin: 0.8436 F1_9: 0.7063 F1_avg: 0.7750
Epoch  26 | Loss: 0.8781 | Acc18: 0.6211 F1_18: 0.6823 | F1_bin: 0.8449 F1_9: 0.7059 F1_avg: 0.7754
Epoch  27 | Loss: 0.8493 | Acc18: 0.6389 F1_18: 0.6949 | F1_bin: 0.8582 F1_9: 0.7137 F1_avg: 0.7859
Epoch  28 | Loss: 0.8332 | Acc18: 0.6376 F1_18: 0.6904 | F1_bin: 0.8626 F1_9: 0.7124 F1_avg: 0.7875
Epoch  29 | Loss: 0.8358 | Acc18: 0.6303 F1_18: 0.6793 | F1_bin: 0.8522 F1_9: 0.7040 F1_avg: 0.7781
Epoch  30 | Loss: 0.8144 | Acc18: 0.6327 F1_18: 0.6798 | F1_bin: 0.8680 F1_9: 0.6971 F1_avg: 0.7826
Epoch 00031: reducing learning rate of group 0 to 2.5000e-04.
Epoch  31 | Loss: 0.7992 | Acc18: 0.6334 F1_18: 0.6882 | F1_bin: 0.8534 F1_9: 0.7087 F1_avg: 0.7810
Epoch  32 | Loss: 0.7454 | Acc18: 0.6413 F1_18: 0.6904 | F1_bin: 0.8636 F1_9: 0.7094 F1_avg: 0.7865
Epoch  33 | Loss: 0.7437 | Acc18: 0.6407 F1_18: 0.6971 | F1_bin: 0.8650 F1_9: 0.7162 F1_avg: 0.7906
Epoch  34 | Loss: 0.7382 | Acc18: 0.6260 F1_18: 0.6849 | F1_bin: 0.8486 F1_9: 0.7102 F1_avg: 0.7794
Epoch  35 | Loss: 0.7267 | Acc18: 0.6419 F1_18: 0.6928 | F1_bin: 0.8685 F1_9: 0.7109 F1_avg: 0.7897
Epoch  36 | Loss: 0.7297 | Acc18: 0.6291 F1_18: 0.6803 | F1_bin: 0.8602 F1_9: 0.6981 F1_avg: 0.7792
Epoch 00037: reducing learning rate of group 0 to 1.2500e-04.
Epoch  37 | Loss: 0.7030 | Acc18: 0.6205 F1_18: 0.6828 | F1_bin: 0.8605 F1_9: 0.6963 F1_avg: 0.7784
Epoch  38 | Loss: 0.6955 | Acc18: 0.6340 F1_18: 0.6902 | F1_bin: 0.8607 F1_9: 0.7085 F1_avg: 0.7846
Epoch  39 | Loss: 0.6826 | Acc18: 0.6327 F1_18: 0.6936 | F1_bin: 0.8540 F1_9: 0.7114 F1_avg: 0.7827
Epoch  40 | Loss: 0.6643 | Acc18: 0.6303 F1_18: 0.6894 | F1_bin: 0.8601 F1_9: 0.7045 F1_avg: 0.7823
Epoch 00041: reducing learning rate of group 0 to 6.2500e-05.
Epoch  41 | Loss: 0.6825 | Acc18: 0.6309 F1_18: 0.6895 | F1_bin: 0.8662 F1_9: 0.7047 F1_avg: 0.7854
Epoch  42 | Loss: 0.6396 | Acc18: 0.6370 F1_18: 0.6922 | F1_bin: 0.8583 F1_9: 0.7122 F1_avg: 0.7852
Epoch  43 | Loss: 0.6313 | Acc18: 0.6291 F1_18: 0.6898 | F1_bin: 0.8570 F1_9: 0.7073 F1_avg: 0.7821
Epoch  44 | Loss: 0.6431 | Acc18: 0.6309 F1_18: 0.6897 | F1_bin: 0.8564 F1_9: 0.7067 F1_avg: 0.7816
Epoch  45 | Loss: 0.6310 | Acc18: 0.6432 F1_18: 0.6985 | F1_bin: 0.8650 F1_9: 0.7140 F1_avg: 0.7895
Epoch  46 | Loss: 0.6561 | Acc18: 0.6426 F1_18: 0.6963 | F1_bin: 0.8668 F1_9: 0.7127 F1_avg: 0.7897
Epoch  47 | Loss: 0.6487 | Acc18: 0.6376 F1_18: 0.6934 | F1_bin: 0.8613 F1_9: 0.7108 F1_avg: 0.7861
Epoch  48 | Loss: 0.6384 | Acc18: 0.6346 F1_18: 0.6882 | F1_bin: 0.8631 F1_9: 0.7012 F1_avg: 0.7822
Epoch 00049: reducing learning rate of group 0 to 3.1250e-05.
Epoch  49 | Loss: 0.6141 | Acc18: 0.6389 F1_18: 0.6890 | F1_bin: 0.8674 F1_9: 0.7067 F1_avg: 0.7871
Epoch  50 | Loss: 0.6212 | Acc18: 0.6389 F1_18: 0.6916 | F1_bin: 0.8656 F1_9: 0.7086 F1_avg: 0.7871
Epoch  51 | Loss: 0.6288 | Acc18: 0.6413 F1_18: 0.6962 | F1_bin: 0.8699 F1_9: 0.7079 F1_avg: 0.7889
Epoch  52 | Loss: 0.6112 | Acc18: 0.6364 F1_18: 0.6942 | F1_bin: 0.8613 F1_9: 0.7115 F1_avg: 0.7864
Epoch 00053: reducing learning rate of group 0 to 1.5625e-05.
Epoch  53 | Loss: 0.6077 | Acc18: 0.6389 F1_18: 0.6914 | F1_bin: 0.8681 F1_9: 0.7092 F1_avg: 0.7886
Early stopping.

=== Fold 3 ===
Epoch   1 | Loss: 2.0108 | Acc18: 0.4788 F1_18: 0.4731 | F1_bin: 0.7767 F1_9: 0.5090 F1_avg: 0.6429
Epoch   2 | Loss: 1.6274 | Acc18: 0.5157 F1_18: 0.5471 | F1_bin: 0.8094 F1_9: 0.5798 F1_avg: 0.6946
Epoch   3 | Loss: 1.4862 | Acc18: 0.5317 F1_18: 0.5463 | F1_bin: 0.8144 F1_9: 0.5986 F1_avg: 0.7065
Epoch   4 | Loss: 1.4379 | Acc18: 0.5778 F1_18: 0.5832 | F1_bin: 0.8286 F1_9: 0.6303 F1_avg: 0.7294
Epoch   5 | Loss: 1.3949 | Acc18: 0.5673 F1_18: 0.5934 | F1_bin: 0.8364 F1_9: 0.6265 F1_avg: 0.7314
Epoch   6 | Loss: 1.3315 | Acc18: 0.5950 F1_18: 0.6226 | F1_bin: 0.8258 F1_9: 0.6481 F1_avg: 0.7370
Epoch   7 | Loss: 1.3087 | Acc18: 0.6116 F1_18: 0.6366 | F1_bin: 0.8284 F1_9: 0.6645 F1_avg: 0.7464
Epoch   8 | Loss: 1.2590 | Acc18: 0.6152 F1_18: 0.6398 | F1_bin: 0.8310 F1_9: 0.6772 F1_avg: 0.7541
Epoch   9 | Loss: 1.2348 | Acc18: 0.5796 F1_18: 0.6074 | F1_bin: 0.8390 F1_9: 0.6289 F1_avg: 0.7340
Epoch  10 | Loss: 1.2140 | Acc18: 0.6054 F1_18: 0.6358 | F1_bin: 0.8509 F1_9: 0.6579 F1_avg: 0.7544
Epoch  11 | Loss: 1.2170 | Acc18: 0.6152 F1_18: 0.6434 | F1_bin: 0.8440 F1_9: 0.6818 F1_avg: 0.7629
Epoch  12 | Loss: 1.1931 | Acc18: 0.6325 F1_18: 0.6746 | F1_bin: 0.8383 F1_9: 0.7003 F1_avg: 0.7693
Epoch  13 | Loss: 1.1787 | Acc18: 0.6368 F1_18: 0.6691 | F1_bin: 0.8580 F1_9: 0.6962 F1_avg: 0.7771
Epoch  14 | Loss: 1.1366 | Acc18: 0.6411 F1_18: 0.6705 | F1_bin: 0.8405 F1_9: 0.7026 F1_avg: 0.7716
Epoch  15 | Loss: 1.1251 | Acc18: 0.6570 F1_18: 0.6909 | F1_bin: 0.8627 F1_9: 0.7109 F1_avg: 0.7868
Epoch  16 | Loss: 1.1192 | Acc18: 0.6208 F1_18: 0.6617 | F1_bin: 0.8639 F1_9: 0.6786 F1_avg: 0.7713
Epoch  17 | Loss: 1.0888 | Acc18: 0.6325 F1_18: 0.6719 | F1_bin: 0.8530 F1_9: 0.7042 F1_avg: 0.7786
Epoch  18 | Loss: 1.0619 | Acc18: 0.6490 F1_18: 0.6834 | F1_bin: 0.8556 F1_9: 0.7109 F1_avg: 0.7832
Epoch 00019: reducing learning rate of group 0 to 5.0000e-04.
Epoch  19 | Loss: 1.0462 | Acc18: 0.6478 F1_18: 0.6731 | F1_bin: 0.8624 F1_9: 0.7081 F1_avg: 0.7852
Epoch  20 | Loss: 1.0304 | Acc18: 0.6632 F1_18: 0.6998 | F1_bin: 0.8715 F1_9: 0.7223 F1_avg: 0.7969
Epoch  21 | Loss: 0.9607 | Acc18: 0.6632 F1_18: 0.6949 | F1_bin: 0.8692 F1_9: 0.7223 F1_avg: 0.7958
Epoch  22 | Loss: 0.9495 | Acc18: 0.6620 F1_18: 0.6975 | F1_bin: 0.8561 F1_9: 0.7234 F1_avg: 0.7897
Epoch  23 | Loss: 0.9681 | Acc18: 0.6681 F1_18: 0.7076 | F1_bin: 0.8695 F1_9: 0.7265 F1_avg: 0.7980
Epoch  24 | Loss: 0.9286 | Acc18: 0.6693 F1_18: 0.7087 | F1_bin: 0.8591 F1_9: 0.7328 F1_avg: 0.7959
Epoch  25 | Loss: 0.9280 | Acc18: 0.6613 F1_18: 0.6976 | F1_bin: 0.8667 F1_9: 0.7211 F1_avg: 0.7939
Epoch  26 | Loss: 0.9094 | Acc18: 0.6613 F1_18: 0.7025 | F1_bin: 0.8615 F1_9: 0.7266 F1_avg: 0.7940
Epoch  27 | Loss: 0.9031 | Acc18: 0.6675 F1_18: 0.6948 | F1_bin: 0.8632 F1_9: 0.7199 F1_avg: 0.7916
Epoch 00028: reducing learning rate of group 0 to 2.5000e-04.
Epoch  28 | Loss: 0.8711 | Acc18: 0.6718 F1_18: 0.7056 | F1_bin: 0.8761 F1_9: 0.7339 F1_avg: 0.8050
Epoch  29 | Loss: 0.8430 | Acc18: 0.6749 F1_18: 0.7150 | F1_bin: 0.8646 F1_9: 0.7398 F1_avg: 0.8022
Epoch  30 | Loss: 0.8376 | Acc18: 0.6767 F1_18: 0.7203 | F1_bin: 0.8729 F1_9: 0.7438 F1_avg: 0.8083
Epoch  31 | Loss: 0.8210 | Acc18: 0.6896 F1_18: 0.7320 | F1_bin: 0.8780 F1_9: 0.7510 F1_avg: 0.8145
Epoch  32 | Loss: 0.8084 | Acc18: 0.6767 F1_18: 0.7227 | F1_bin: 0.8663 F1_9: 0.7402 F1_avg: 0.8032
Epoch  33 | Loss: 0.8151 | Acc18: 0.6841 F1_18: 0.7252 | F1_bin: 0.8847 F1_9: 0.7419 F1_avg: 0.8133
Epoch  34 | Loss: 0.7910 | Acc18: 0.6742 F1_18: 0.7132 | F1_bin: 0.8738 F1_9: 0.7330 F1_avg: 0.8034
Epoch 00035: reducing learning rate of group 0 to 1.2500e-04.
Epoch  35 | Loss: 0.7863 | Acc18: 0.6841 F1_18: 0.7271 | F1_bin: 0.8774 F1_9: 0.7497 F1_avg: 0.8136
Epoch  36 | Loss: 0.7766 | Acc18: 0.6865 F1_18: 0.7290 | F1_bin: 0.8769 F1_9: 0.7497 F1_avg: 0.8133
Epoch  37 | Loss: 0.7542 | Acc18: 0.6853 F1_18: 0.7232 | F1_bin: 0.8783 F1_9: 0.7429 F1_avg: 0.8106
Epoch  38 | Loss: 0.7627 | Acc18: 0.6822 F1_18: 0.7239 | F1_bin: 0.8734 F1_9: 0.7433 F1_avg: 0.8084
Epoch 00039: reducing learning rate of group 0 to 6.2500e-05.
Epoch  39 | Loss: 0.7418 | Acc18: 0.6718 F1_18: 0.7140 | F1_bin: 0.8650 F1_9: 0.7355 F1_avg: 0.8002
Epoch  40 | Loss: 0.7229 | Acc18: 0.6835 F1_18: 0.7283 | F1_bin: 0.8721 F1_9: 0.7484 F1_avg: 0.8102
Epoch  41 | Loss: 0.7313 | Acc18: 0.6767 F1_18: 0.7214 | F1_bin: 0.8778 F1_9: 0.7400 F1_avg: 0.8089
Epoch  42 | Loss: 0.7159 | Acc18: 0.6810 F1_18: 0.7254 | F1_bin: 0.8748 F1_9: 0.7445 F1_avg: 0.8096
Epoch 00043: reducing learning rate of group 0 to 3.1250e-05.
Epoch  43 | Loss: 0.7171 | Acc18: 0.6810 F1_18: 0.7227 | F1_bin: 0.8736 F1_9: 0.7400 F1_avg: 0.8068
Epoch  44 | Loss: 0.7048 | Acc18: 0.6841 F1_18: 0.7281 | F1_bin: 0.8760 F1_9: 0.7461 F1_avg: 0.8111
Epoch  45 | Loss: 0.7058 | Acc18: 0.6847 F1_18: 0.7266 | F1_bin: 0.8742 F1_9: 0.7475 F1_avg: 0.8108
Epoch  46 | Loss: 0.7125 | Acc18: 0.6767 F1_18: 0.7202 | F1_bin: 0.8735 F1_9: 0.7421 F1_avg: 0.8078
Epoch 00047: reducing learning rate of group 0 to 1.5625e-05.
Epoch  47 | Loss: 0.7136 | Acc18: 0.6792 F1_18: 0.7250 | F1_bin: 0.8749 F1_9: 0.7437 F1_avg: 0.8093
Epoch  48 | Loss: 0.7010 | Acc18: 0.6792 F1_18: 0.7248 | F1_bin: 0.8755 F1_9: 0.7421 F1_avg: 0.8088
Epoch  49 | Loss: 0.7026 | Acc18: 0.6755 F1_18: 0.7215 | F1_bin: 0.8724 F1_9: 0.7414 F1_avg: 0.8069
Epoch  50 | Loss: 0.6990 | Acc18: 0.6816 F1_18: 0.7264 | F1_bin: 0.8731 F1_9: 0.7460 F1_avg: 0.8095
Epoch 00051: reducing learning rate of group 0 to 7.8125e-06.
Epoch  51 | Loss: 0.6885 | Acc18: 0.6767 F1_18: 0.7196 | F1_bin: 0.8736 F1_9: 0.7402 F1_avg: 0.8069
Early stopping.

=== Fold 4 ===
Epoch   1 | Loss: 2.0148 | Acc18: 0.4815 F1_18: 0.4906 | F1_bin: 0.7781 F1_9: 0.5149 F1_avg: 0.6465
Epoch   2 | Loss: 1.6395 | Acc18: 0.5166 F1_18: 0.5316 | F1_bin: 0.7845 F1_9: 0.5708 F1_avg: 0.6777
Epoch   3 | Loss: 1.4979 | Acc18: 0.5677 F1_18: 0.5851 | F1_bin: 0.8177 F1_9: 0.6264 F1_avg: 0.7221
Epoch   4 | Loss: 1.4310 | Acc18: 0.5855 F1_18: 0.6198 | F1_bin: 0.7896 F1_9: 0.6458 F1_avg: 0.7177
Epoch   5 | Loss: 1.3674 | Acc18: 0.5713 F1_18: 0.5910 | F1_bin: 0.8129 F1_9: 0.6120 F1_avg: 0.7124
Epoch   6 | Loss: 1.3079 | Acc18: 0.5959 F1_18: 0.6285 | F1_bin: 0.8030 F1_9: 0.6658 F1_avg: 0.7344
Epoch   7 | Loss: 1.2806 | Acc18: 0.5886 F1_18: 0.6146 | F1_bin: 0.8179 F1_9: 0.6486 F1_avg: 0.7332
Epoch   8 | Loss: 1.2655 | Acc18: 0.6082 F1_18: 0.6384 | F1_bin: 0.8351 F1_9: 0.6622 F1_avg: 0.7487
Epoch   9 | Loss: 1.2181 | Acc18: 0.6144 F1_18: 0.6498 | F1_bin: 0.8329 F1_9: 0.6712 F1_avg: 0.7521
Epoch  10 | Loss: 1.2305 | Acc18: 0.5996 F1_18: 0.6413 | F1_bin: 0.8471 F1_9: 0.6556 F1_avg: 0.7514
Epoch  11 | Loss: 1.1927 | Acc18: 0.6175 F1_18: 0.6535 | F1_bin: 0.8111 F1_9: 0.6917 F1_avg: 0.7514
Epoch  12 | Loss: 1.1622 | Acc18: 0.6298 F1_18: 0.6698 | F1_bin: 0.8456 F1_9: 0.6979 F1_avg: 0.7718
Epoch  13 | Loss: 1.1260 | Acc18: 0.6335 F1_18: 0.6832 | F1_bin: 0.8512 F1_9: 0.7011 F1_avg: 0.7761
Epoch  14 | Loss: 1.1226 | Acc18: 0.6347 F1_18: 0.6701 | F1_bin: 0.8472 F1_9: 0.6905 F1_avg: 0.7689
Epoch  15 | Loss: 1.1022 | Acc18: 0.6273 F1_18: 0.6689 | F1_bin: 0.8449 F1_9: 0.6857 F1_avg: 0.7653
Epoch  16 | Loss: 1.1074 | Acc18: 0.6439 F1_18: 0.6876 | F1_bin: 0.8512 F1_9: 0.7036 F1_avg: 0.7774
Epoch  17 | Loss: 1.0956 | Acc18: 0.6285 F1_18: 0.6678 | F1_bin: 0.8450 F1_9: 0.6895 F1_avg: 0.7672
Epoch  18 | Loss: 1.0649 | Acc18: 0.6255 F1_18: 0.6706 | F1_bin: 0.8598 F1_9: 0.6834 F1_avg: 0.7716
Epoch  19 | Loss: 1.0596 | Acc18: 0.6445 F1_18: 0.6921 | F1_bin: 0.8701 F1_9: 0.7024 F1_avg: 0.7863
Epoch  20 | Loss: 1.0332 | Acc18: 0.6464 F1_18: 0.6892 | F1_bin: 0.8407 F1_9: 0.7086 F1_avg: 0.7747
Epoch  21 | Loss: 1.0296 | Acc18: 0.6488 F1_18: 0.6943 | F1_bin: 0.8622 F1_9: 0.7107 F1_avg: 0.7864
Epoch  22 | Loss: 1.0038 | Acc18: 0.6544 F1_18: 0.7062 | F1_bin: 0.8764 F1_9: 0.7136 F1_avg: 0.7950
Epoch  23 | Loss: 1.0104 | Acc18: 0.6427 F1_18: 0.6818 | F1_bin: 0.8633 F1_9: 0.7002 F1_avg: 0.7817
Epoch  24 | Loss: 0.9852 | Acc18: 0.6445 F1_18: 0.6822 | F1_bin: 0.8633 F1_9: 0.7024 F1_avg: 0.7829
Epoch  25 | Loss: 0.9529 | Acc18: 0.6507 F1_18: 0.6968 | F1_bin: 0.8600 F1_9: 0.7099 F1_avg: 0.7850
Epoch 00026: reducing learning rate of group 0 to 5.0000e-04.
Epoch  26 | Loss: 0.9582 | Acc18: 0.6470 F1_18: 0.6921 | F1_bin: 0.8533 F1_9: 0.7061 F1_avg: 0.7797
Epoch  27 | Loss: 0.9088 | Acc18: 0.6556 F1_18: 0.7028 | F1_bin: 0.8602 F1_9: 0.7205 F1_avg: 0.7904
Epoch  28 | Loss: 0.8697 | Acc18: 0.6538 F1_18: 0.7076 | F1_bin: 0.8589 F1_9: 0.7172 F1_avg: 0.7881
Epoch  29 | Loss: 0.8809 | Acc18: 0.6617 F1_18: 0.7139 | F1_bin: 0.8659 F1_9: 0.7320 F1_avg: 0.7989
Epoch  30 | Loss: 0.8373 | Acc18: 0.6544 F1_18: 0.7026 | F1_bin: 0.8665 F1_9: 0.7141 F1_avg: 0.7903
Epoch  31 | Loss: 0.8324 | Acc18: 0.6581 F1_18: 0.7078 | F1_bin: 0.8634 F1_9: 0.7214 F1_avg: 0.7924
Epoch  32 | Loss: 0.8338 | Acc18: 0.6451 F1_18: 0.6888 | F1_bin: 0.8541 F1_9: 0.7044 F1_avg: 0.7793
Epoch 00033: reducing learning rate of group 0 to 2.5000e-04.
Epoch  33 | Loss: 0.8111 | Acc18: 0.6470 F1_18: 0.6975 | F1_bin: 0.8582 F1_9: 0.7130 F1_avg: 0.7856
Epoch  34 | Loss: 0.7725 | Acc18: 0.6654 F1_18: 0.7074 | F1_bin: 0.8644 F1_9: 0.7247 F1_avg: 0.7946
Epoch  35 | Loss: 0.7641 | Acc18: 0.6544 F1_18: 0.7024 | F1_bin: 0.8701 F1_9: 0.7125 F1_avg: 0.7913
Epoch  36 | Loss: 0.7578 | Acc18: 0.6581 F1_18: 0.7028 | F1_bin: 0.8682 F1_9: 0.7164 F1_avg: 0.7923
Epoch 00037: reducing learning rate of group 0 to 1.2500e-04.
Epoch  37 | Loss: 0.7318 | Acc18: 0.6568 F1_18: 0.7019 | F1_bin: 0.8656 F1_9: 0.7171 F1_avg: 0.7913
Epoch  38 | Loss: 0.7152 | Acc18: 0.6667 F1_18: 0.7124 | F1_bin: 0.8676 F1_9: 0.7293 F1_avg: 0.7985
Epoch  39 | Loss: 0.7126 | Acc18: 0.6630 F1_18: 0.7096 | F1_bin: 0.8664 F1_9: 0.7221 F1_avg: 0.7942
Epoch  40 | Loss: 0.7128 | Acc18: 0.6654 F1_18: 0.7108 | F1_bin: 0.8665 F1_9: 0.7263 F1_avg: 0.7964
Epoch  41 | Loss: 0.6868 | Acc18: 0.6704 F1_18: 0.7180 | F1_bin: 0.8706 F1_9: 0.7339 F1_avg: 0.8023
Epoch  42 | Loss: 0.6786 | Acc18: 0.6691 F1_18: 0.7188 | F1_bin: 0.8688 F1_9: 0.7341 F1_avg: 0.8014
Epoch  43 | Loss: 0.6859 | Acc18: 0.6611 F1_18: 0.7095 | F1_bin: 0.8707 F1_9: 0.7236 F1_avg: 0.7972
Epoch  44 | Loss: 0.6726 | Acc18: 0.6691 F1_18: 0.7210 | F1_bin: 0.8736 F1_9: 0.7336 F1_avg: 0.8036
Epoch  45 | Loss: 0.6841 | Acc18: 0.6691 F1_18: 0.7207 | F1_bin: 0.8666 F1_9: 0.7328 F1_avg: 0.7997
Epoch  46 | Loss: 0.6725 | Acc18: 0.6740 F1_18: 0.7225 | F1_bin: 0.8677 F1_9: 0.7371 F1_avg: 0.8024
Epoch  47 | Loss: 0.6616 | Acc18: 0.6667 F1_18: 0.7163 | F1_bin: 0.8707 F1_9: 0.7263 F1_avg: 0.7985
Epoch  48 | Loss: 0.6528 | Acc18: 0.6685 F1_18: 0.7180 | F1_bin: 0.8726 F1_9: 0.7307 F1_avg: 0.8017
Epoch  49 | Loss: 0.6612 | Acc18: 0.6716 F1_18: 0.7219 | F1_bin: 0.8738 F1_9: 0.7322 F1_avg: 0.8030
Epoch 00050: reducing learning rate of group 0 to 6.2500e-05.
Epoch  50 | Loss: 0.6343 | Acc18: 0.6654 F1_18: 0.7181 | F1_bin: 0.8682 F1_9: 0.7294 F1_avg: 0.7988
Epoch  51 | Loss: 0.6457 | Acc18: 0.6667 F1_18: 0.7186 | F1_bin: 0.8725 F1_9: 0.7278 F1_avg: 0.8001
Epoch  52 | Loss: 0.6552 | Acc18: 0.6704 F1_18: 0.7196 | F1_bin: 0.8768 F1_9: 0.7285 F1_avg: 0.8026
Epoch  53 | Loss: 0.6279 | Acc18: 0.6704 F1_18: 0.7199 | F1_bin: 0.8738 F1_9: 0.7320 F1_avg: 0.8029
Epoch 00054: reducing learning rate of group 0 to 3.1250e-05.
Epoch  54 | Loss: 0.6285 | Acc18: 0.6691 F1_18: 0.7173 | F1_bin: 0.8755 F1_9: 0.7276 F1_avg: 0.8016
Epoch  55 | Loss: 0.6219 | Acc18: 0.6654 F1_18: 0.7119 | F1_bin: 0.8749 F1_9: 0.7235 F1_avg: 0.7992
Epoch  56 | Loss: 0.6413 | Acc18: 0.6697 F1_18: 0.7192 | F1_bin: 0.8742 F1_9: 0.7305 F1_avg: 0.8024
Epoch  57 | Loss: 0.6243 | Acc18: 0.6642 F1_18: 0.7132 | F1_bin: 0.8748 F1_9: 0.7253 F1_avg: 0.8001
Epoch 00058: reducing learning rate of group 0 to 1.5625e-05.
Epoch  58 | Loss: 0.6055 | Acc18: 0.6704 F1_18: 0.7193 | F1_bin: 0.8768 F1_9: 0.7331 F1_avg: 0.8049
Epoch  59 | Loss: 0.6273 | Acc18: 0.6722 F1_18: 0.7192 | F1_bin: 0.8769 F1_9: 0.7305 F1_avg: 0.8037
Epoch  60 | Loss: 0.6229 | Acc18: 0.6673 F1_18: 0.7125 | F1_bin: 0.8755 F1_9: 0.7269 F1_avg: 0.8012
Epoch  61 | Loss: 0.6286 | Acc18: 0.6716 F1_18: 0.7206 | F1_bin: 0.8762 F1_9: 0.7351 F1_avg: 0.8057
Epoch 00062: reducing learning rate of group 0 to 7.8125e-06.
Epoch  62 | Loss: 0.6233 | Acc18: 0.6704 F1_18: 0.7156 | F1_bin: 0.8744 F1_9: 0.7290 F1_avg: 0.8017
Epoch  63 | Loss: 0.6121 | Acc18: 0.6734 F1_18: 0.7190 | F1_bin: 0.8743 F1_9: 0.7338 F1_avg: 0.8041
Epoch  64 | Loss: 0.6302 | Acc18: 0.6722 F1_18: 0.7182 | F1_bin: 0.8750 F1_9: 0.7299 F1_avg: 0.8025
Epoch  65 | Loss: 0.6092 | Acc18: 0.6747 F1_18: 0.7222 | F1_bin: 0.8768 F1_9: 0.7369 F1_avg: 0.8068
Epoch 00066: reducing learning rate of group 0 to 3.9063e-06.
Epoch  66 | Loss: 0.6274 | Acc18: 0.6734 F1_18: 0.7212 | F1_bin: 0.8756 F1_9: 0.7346 F1_avg: 0.8051
Epoch  67 | Loss: 0.6212 | Acc18: 0.6728 F1_18: 0.7189 | F1_bin: 0.8750 F1_9: 0.7327 F1_avg: 0.8038
Epoch  68 | Loss: 0.6216 | Acc18: 0.6691 F1_18: 0.7161 | F1_bin: 0.8732 F1_9: 0.7297 F1_avg: 0.8014
Epoch  69 | Loss: 0.6083 | Acc18: 0.6710 F1_18: 0.7199 | F1_bin: 0.8737 F1_9: 0.7331 F1_avg: 0.8034
Epoch 00070: reducing learning rate of group 0 to 1.9531e-06.
Epoch  70 | Loss: 0.6335 | Acc18: 0.6716 F1_18: 0.7190 | F1_bin: 0.8750 F1_9: 0.7335 F1_avg: 0.8042
Epoch  71 | Loss: 0.6197 | Acc18: 0.6722 F1_18: 0.7173 | F1_bin: 0.8793 F1_9: 0.7312 F1_avg: 0.8052
Epoch  72 | Loss: 0.6207 | Acc18: 0.6667 F1_18: 0.7147 | F1_bin: 0.8755 F1_9: 0.7306 F1_avg: 0.8031
Epoch  73 | Loss: 0.6055 | Acc18: 0.6747 F1_18: 0.7229 | F1_bin: 0.8794 F1_9: 0.7349 F1_avg: 0.8071
Epoch  74 | Loss: 0.6207 | Acc18: 0.6704 F1_18: 0.7190 | F1_bin: 0.8731 F1_9: 0.7310 F1_avg: 0.8021
Epoch  75 | Loss: 0.6119 | Acc18: 0.6747 F1_18: 0.7238 | F1_bin: 0.8768 F1_9: 0.7368 F1_avg: 0.8068
Epoch  76 | Loss: 0.6215 | Acc18: 0.6710 F1_18: 0.7164 | F1_bin: 0.8750 F1_9: 0.7295 F1_avg: 0.8023
Epoch  77 | Loss: 0.6146 | Acc18: 0.6740 F1_18: 0.7231 | F1_bin: 0.8744 F1_9: 0.7353 F1_avg: 0.8049
Epoch  78 | Loss: 0.6303 | Acc18: 0.6704 F1_18: 0.7188 | F1_bin: 0.8774 F1_9: 0.7321 F1_avg: 0.8048
Epoch 00079: reducing learning rate of group 0 to 9.7656e-07.
Epoch  79 | Loss: 0.6289 | Acc18: 0.6716 F1_18: 0.7202 | F1_bin: 0.8749 F1_9: 0.7359 F1_avg: 0.8054
Epoch  80 | Loss: 0.6124 | Acc18: 0.6759 F1_18: 0.7244 | F1_bin: 0.8781 F1_9: 0.7342 F1_avg: 0.8061
Epoch  81 | Loss: 0.6123 | Acc18: 0.6728 F1_18: 0.7196 | F1_bin: 0.8744 F1_9: 0.7333 F1_avg: 0.8038
Epoch  82 | Loss: 0.6128 | Acc18: 0.6710 F1_18: 0.7192 | F1_bin: 0.8713 F1_9: 0.7349 F1_avg: 0.8031
Epoch  83 | Loss: 0.6063 | Acc18: 0.6667 F1_18: 0.7147 | F1_bin: 0.8737 F1_9: 0.7285 F1_avg: 0.8011
Epoch 00084: reducing learning rate of group 0 to 4.8828e-07.
Epoch  84 | Loss: 0.6257 | Acc18: 0.6753 F1_18: 0.7210 | F1_bin: 0.8769 F1_9: 0.7337 F1_avg: 0.8053
Epoch  85 | Loss: 0.6338 | Acc18: 0.6697 F1_18: 0.7180 | F1_bin: 0.8773 F1_9: 0.7319 F1_avg: 0.8046
Epoch  86 | Loss: 0.6025 | Acc18: 0.6710 F1_18: 0.7177 | F1_bin: 0.8756 F1_9: 0.7316 F1_avg: 0.8036
Epoch  87 | Loss: 0.6164 | Acc18: 0.6734 F1_18: 0.7196 | F1_bin: 0.8786 F1_9: 0.7338 F1_avg: 0.8062
Epoch 00088: reducing learning rate of group 0 to 2.4414e-07.
Epoch  88 | Loss: 0.6056 | Acc18: 0.6667 F1_18: 0.7137 | F1_bin: 0.8730 F1_9: 0.7281 F1_avg: 0.8006
Epoch  89 | Loss: 0.6309 | Acc18: 0.6728 F1_18: 0.7207 | F1_bin: 0.8768 F1_9: 0.7336 F1_avg: 0.8052
Epoch  90 | Loss: 0.6242 | Acc18: 0.6679 F1_18: 0.7162 | F1_bin: 0.8767 F1_9: 0.7310 F1_avg: 0.8039
Epoch  91 | Loss: 0.6352 | Acc18: 0.6765 F1_18: 0.7224 | F1_bin: 0.8756 F1_9: 0.7353 F1_avg: 0.8055
Epoch 00092: reducing learning rate of group 0 to 1.2207e-07.
Epoch  92 | Loss: 0.6126 | Acc18: 0.6753 F1_18: 0.7236 | F1_bin: 0.8774 F1_9: 0.7365 F1_avg: 0.8070
Epoch  93 | Loss: 0.6175 | Acc18: 0.6728 F1_18: 0.7209 | F1_bin: 0.8818 F1_9: 0.7310 F1_avg: 0.8064
Early stopping.

=== Fold 5 ===
Epoch   1 | Loss: 1.9624 | Acc18: 0.4330 F1_18: 0.4281 | F1_bin: 0.7524 F1_9: 0.4786 F1_avg: 0.6155
Epoch   2 | Loss: 1.6422 | Acc18: 0.5162 F1_18: 0.5209 | F1_bin: 0.7976 F1_9: 0.5666 F1_avg: 0.6821
Epoch   3 | Loss: 1.4926 | Acc18: 0.5144 F1_18: 0.5404 | F1_bin: 0.8060 F1_9: 0.5822 F1_avg: 0.6941
Epoch   4 | Loss: 1.4132 | Acc18: 0.5443 F1_18: 0.5762 | F1_bin: 0.8262 F1_9: 0.6181 F1_avg: 0.7221
Epoch   5 | Loss: 1.3597 | Acc18: 0.5443 F1_18: 0.5782 | F1_bin: 0.8150 F1_9: 0.6044 F1_avg: 0.7097
Epoch   6 | Loss: 1.3085 | Acc18: 0.5700 F1_18: 0.6038 | F1_bin: 0.8373 F1_9: 0.6431 F1_avg: 0.7402
Epoch   7 | Loss: 1.2754 | Acc18: 0.5517 F1_18: 0.6039 | F1_bin: 0.8279 F1_9: 0.6338 F1_avg: 0.7309
Epoch   8 | Loss: 1.2721 | Acc18: 0.5920 F1_18: 0.6247 | F1_bin: 0.8384 F1_9: 0.6633 F1_avg: 0.7508
Epoch   9 | Loss: 1.2046 | Acc18: 0.5786 F1_18: 0.6340 | F1_bin: 0.8234 F1_9: 0.6556 F1_avg: 0.7395
Epoch  10 | Loss: 1.1951 | Acc18: 0.5920 F1_18: 0.6283 | F1_bin: 0.8341 F1_9: 0.6609 F1_avg: 0.7475
Epoch  11 | Loss: 1.1774 | Acc18: 0.6031 F1_18: 0.6484 | F1_bin: 0.8393 F1_9: 0.6691 F1_avg: 0.7542
Epoch  12 | Loss: 1.1277 | Acc18: 0.6086 F1_18: 0.6578 | F1_bin: 0.8484 F1_9: 0.6840 F1_avg: 0.7662
Epoch  13 | Loss: 1.1289 | Acc18: 0.6024 F1_18: 0.6450 | F1_bin: 0.8484 F1_9: 0.6666 F1_avg: 0.7575
Epoch  14 | Loss: 1.1158 | Acc18: 0.5920 F1_18: 0.6442 | F1_bin: 0.8473 F1_9: 0.6632 F1_avg: 0.7553
Epoch  15 | Loss: 1.0953 | Acc18: 0.5951 F1_18: 0.6425 | F1_bin: 0.8370 F1_9: 0.6735 F1_avg: 0.7553
Epoch  16 | Loss: 1.0801 | Acc18: 0.6031 F1_18: 0.6592 | F1_bin: 0.8456 F1_9: 0.6817 F1_avg: 0.7637
Epoch  17 | Loss: 1.0824 | Acc18: 0.6104 F1_18: 0.6604 | F1_bin: 0.8583 F1_9: 0.6873 F1_avg: 0.7728
Epoch  18 | Loss: 1.0375 | Acc18: 0.5976 F1_18: 0.6487 | F1_bin: 0.8454 F1_9: 0.6757 F1_avg: 0.7605
Epoch  19 | Loss: 1.0195 | Acc18: 0.6067 F1_18: 0.6571 | F1_bin: 0.8439 F1_9: 0.6855 F1_avg: 0.7647
Epoch  20 | Loss: 1.0193 | Acc18: 0.6202 F1_18: 0.6715 | F1_bin: 0.8511 F1_9: 0.6866 F1_avg: 0.7689
Epoch  21 | Loss: 0.9900 | Acc18: 0.6031 F1_18: 0.6571 | F1_bin: 0.8486 F1_9: 0.6778 F1_avg: 0.7632
Epoch  22 | Loss: 0.9887 | Acc18: 0.6024 F1_18: 0.6578 | F1_bin: 0.8384 F1_9: 0.6771 F1_avg: 0.7578
Epoch  23 | Loss: 0.9605 | Acc18: 0.6031 F1_18: 0.6514 | F1_bin: 0.8462 F1_9: 0.6786 F1_avg: 0.7624
Epoch 00024: reducing learning rate of group 0 to 5.0000e-04.
Epoch  24 | Loss: 0.9882 | Acc18: 0.6018 F1_18: 0.6573 | F1_bin: 0.8305 F1_9: 0.6852 F1_avg: 0.7579
Epoch  25 | Loss: 0.8899 | Acc18: 0.6159 F1_18: 0.6673 | F1_bin: 0.8414 F1_9: 0.6947 F1_avg: 0.7681
Epoch  26 | Loss: 0.8675 | Acc18: 0.6159 F1_18: 0.6759 | F1_bin: 0.8482 F1_9: 0.6922 F1_avg: 0.7702
Epoch  27 | Loss: 0.8713 | Acc18: 0.6275 F1_18: 0.6858 | F1_bin: 0.8574 F1_9: 0.7011 F1_avg: 0.7792
Epoch  28 | Loss: 0.8412 | Acc18: 0.6049 F1_18: 0.6654 | F1_bin: 0.8412 F1_9: 0.6868 F1_avg: 0.7640
Epoch  29 | Loss: 0.8148 | Acc18: 0.6202 F1_18: 0.6838 | F1_bin: 0.8444 F1_9: 0.6994 F1_avg: 0.7719
Epoch  30 | Loss: 0.8121 | Acc18: 0.6269 F1_18: 0.6799 | F1_bin: 0.8543 F1_9: 0.7028 F1_avg: 0.7785
Epoch 00031: reducing learning rate of group 0 to 2.5000e-04.
Epoch  31 | Loss: 0.7992 | Acc18: 0.6165 F1_18: 0.6745 | F1_bin: 0.8406 F1_9: 0.7008 F1_avg: 0.7707
Epoch  32 | Loss: 0.7816 | Acc18: 0.6281 F1_18: 0.6808 | F1_bin: 0.8487 F1_9: 0.7046 F1_avg: 0.7767
Epoch  33 | Loss: 0.7501 | Acc18: 0.6245 F1_18: 0.6840 | F1_bin: 0.8504 F1_9: 0.7071 F1_avg: 0.7787
Epoch  34 | Loss: 0.7512 | Acc18: 0.6269 F1_18: 0.6760 | F1_bin: 0.8520 F1_9: 0.6969 F1_avg: 0.7744
Epoch 00035: reducing learning rate of group 0 to 1.2500e-04.
Epoch  35 | Loss: 0.7157 | Acc18: 0.6202 F1_18: 0.6776 | F1_bin: 0.8535 F1_9: 0.6987 F1_avg: 0.7761
Epoch  36 | Loss: 0.7176 | Acc18: 0.6183 F1_18: 0.6770 | F1_bin: 0.8460 F1_9: 0.6979 F1_avg: 0.7719
Epoch  37 | Loss: 0.6965 | Acc18: 0.6171 F1_18: 0.6797 | F1_bin: 0.8508 F1_9: 0.6978 F1_avg: 0.7743
Epoch  38 | Loss: 0.7031 | Acc18: 0.6220 F1_18: 0.6805 | F1_bin: 0.8445 F1_9: 0.7012 F1_avg: 0.7729
Epoch 00039: reducing learning rate of group 0 to 6.2500e-05.
Epoch  39 | Loss: 0.6979 | Acc18: 0.6226 F1_18: 0.6788 | F1_bin: 0.8560 F1_9: 0.6978 F1_avg: 0.7769
Epoch  40 | Loss: 0.6810 | Acc18: 0.6257 F1_18: 0.6798 | F1_bin: 0.8538 F1_9: 0.6996 F1_avg: 0.7767
Epoch  41 | Loss: 0.6672 | Acc18: 0.6232 F1_18: 0.6808 | F1_bin: 0.8502 F1_9: 0.7005 F1_avg: 0.7754
Epoch  42 | Loss: 0.6601 | Acc18: 0.6214 F1_18: 0.6785 | F1_bin: 0.8494 F1_9: 0.7012 F1_avg: 0.7753
Epoch 00043: reducing learning rate of group 0 to 3.1250e-05.
Epoch  43 | Loss: 0.6764 | Acc18: 0.6202 F1_18: 0.6773 | F1_bin: 0.8465 F1_9: 0.6988 F1_avg: 0.7727
Epoch  44 | Loss: 0.6584 | Acc18: 0.6153 F1_18: 0.6759 | F1_bin: 0.8459 F1_9: 0.6967 F1_avg: 0.7713
Epoch  45 | Loss: 0.6542 | Acc18: 0.6196 F1_18: 0.6767 | F1_bin: 0.8513 F1_9: 0.6998 F1_avg: 0.7755
Epoch  46 | Loss: 0.6486 | Acc18: 0.6239 F1_18: 0.6815 | F1_bin: 0.8494 F1_9: 0.7018 F1_avg: 0.7756
Epoch 00047: reducing learning rate of group 0 to 1.5625e-05.
Epoch  47 | Loss: 0.6593 | Acc18: 0.6214 F1_18: 0.6796 | F1_bin: 0.8475 F1_9: 0.7019 F1_avg: 0.7747
Early stopping.
"""
