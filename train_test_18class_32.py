#ModelVariant_LSTMGRU 特徴量に角速度を追加、thmはIMUに含まれないので削除　IMUonlyモデル
# CV=0.793
#CMI 2025 デモ提出 バージョン69　IMUonly+all LB=0.81

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

# ========= 設定 =========
RAW_CSV = "train.csv"
SAVE_DIR = "train_test_18class_32"
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
    rot_data = df[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
    df['angular_vel_x'], df['angular_vel_y'], df['angular_vel_z'] = calculate_angular_velocity_from_quat(rot_data).T
    df['angular_dist'] = calculate_angular_distance(rot_data)
    return df

# ========= モデル定義 =========
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

class IMUDataset(Dataset):
    def __init__(self,X,y):
        self.X = torch.tensor(np.array(X),dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.y[i]

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
    train_loader = DataLoader(IMUDataset(X_train,y_train),batch_size=BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(IMUDataset(X_val,y_val),batch_size=BATCH_SIZE)

    model = ModelVariant_LSTMGRU(imu_dim, len(le.classes_)).to(device)
    opt = torch.optim.Adam(model.parameters(),lr=LR_INIT,weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5, verbose=True)
    best_f1, patience_counter = 0,0

    for epoch in range(EPOCHS):
        model.train()
        loss_sum=0
        for xb,yb in train_loader:
            xb,yb=xb.to(device),yb.to(device)
            opt.zero_grad()
            logits,_ = model(xb)
            loss=F.cross_entropy(logits,yb)
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
                trues.extend(yb.numpy())

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

        if f1_18 > best_f1:
            best_f1, patience_counter = f1_18, 0
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
Epoch   1 | Loss: 1.9173 | Acc18: 0.5104 F1_18: 0.5321 | F1_bin: 0.7966 F1_9: 0.5551 F1_avg: 0.6759
Epoch   2 | Loss: 1.4410 | Acc18: 0.5300 F1_18: 0.5370 | F1_bin: 0.8337 F1_9: 0.5628 F1_avg: 0.6983
Epoch   3 | Loss: 1.2849 | Acc18: 0.5735 F1_18: 0.6157 | F1_bin: 0.8198 F1_9: 0.6427 F1_avg: 0.7312
Epoch   4 | Loss: 1.1741 | Acc18: 0.5968 F1_18: 0.6408 | F1_bin: 0.8454 F1_9: 0.6618 F1_avg: 0.7536
Epoch   5 | Loss: 1.1215 | Acc18: 0.6446 F1_18: 0.6791 | F1_bin: 0.8601 F1_9: 0.6903 F1_avg: 0.7752
Epoch   6 | Loss: 1.0693 | Acc18: 0.6324 F1_18: 0.6746 | F1_bin: 0.8550 F1_9: 0.6957 F1_avg: 0.7753
Epoch   7 | Loss: 1.0104 | Acc18: 0.6366 F1_18: 0.6647 | F1_bin: 0.8626 F1_9: 0.6901 F1_avg: 0.7763
Epoch   8 | Loss: 0.9733 | Acc18: 0.6452 F1_18: 0.6836 | F1_bin: 0.8693 F1_9: 0.6995 F1_avg: 0.7844
Epoch   9 | Loss: 0.9366 | Acc18: 0.6501 F1_18: 0.6961 | F1_bin: 0.8646 F1_9: 0.7247 F1_avg: 0.7947
Epoch  10 | Loss: 0.8925 | Acc18: 0.6201 F1_18: 0.6716 | F1_bin: 0.8535 F1_9: 0.6854 F1_avg: 0.7694
Epoch  11 | Loss: 0.8875 | Acc18: 0.6158 F1_18: 0.6593 | F1_bin: 0.8496 F1_9: 0.6835 F1_avg: 0.7666
Epoch  12 | Loss: 0.8399 | Acc18: 0.6434 F1_18: 0.6916 | F1_bin: 0.8630 F1_9: 0.7081 F1_avg: 0.7855
Epoch  13 | Loss: 0.8409 | Acc18: 0.6440 F1_18: 0.7000 | F1_bin: 0.8582 F1_9: 0.7129 F1_avg: 0.7855
Epoch  14 | Loss: 0.7897 | Acc18: 0.6452 F1_18: 0.6922 | F1_bin: 0.8751 F1_9: 0.7101 F1_avg: 0.7926
Epoch  15 | Loss: 0.7824 | Acc18: 0.6446 F1_18: 0.7103 | F1_bin: 0.8578 F1_9: 0.7219 F1_avg: 0.7898
Epoch  16 | Loss: 0.7631 | Acc18: 0.6630 F1_18: 0.7205 | F1_bin: 0.8698 F1_9: 0.7292 F1_avg: 0.7995
Epoch  17 | Loss: 0.7576 | Acc18: 0.6612 F1_18: 0.7177 | F1_bin: 0.8778 F1_9: 0.7267 F1_avg: 0.8022
Epoch  18 | Loss: 0.7078 | Acc18: 0.6612 F1_18: 0.7127 | F1_bin: 0.8696 F1_9: 0.7335 F1_avg: 0.8016
Epoch  19 | Loss: 0.6801 | Acc18: 0.6746 F1_18: 0.7318 | F1_bin: 0.8792 F1_9: 0.7433 F1_avg: 0.8112
Epoch  20 | Loss: 0.6659 | Acc18: 0.6814 F1_18: 0.7424 | F1_bin: 0.8845 F1_9: 0.7513 F1_avg: 0.8179
Epoch  21 | Loss: 0.6488 | Acc18: 0.6667 F1_18: 0.7186 | F1_bin: 0.8706 F1_9: 0.7350 F1_avg: 0.8028
Epoch  22 | Loss: 0.6237 | Acc18: 0.6605 F1_18: 0.7074 | F1_bin: 0.8770 F1_9: 0.7217 F1_avg: 0.7994
Epoch  23 | Loss: 0.6045 | Acc18: 0.6716 F1_18: 0.7233 | F1_bin: 0.8690 F1_9: 0.7390 F1_avg: 0.8040
Epoch 00024: reducing learning rate of group 0 to 5.0000e-04.
Epoch  24 | Loss: 0.5946 | Acc18: 0.6771 F1_18: 0.7309 | F1_bin: 0.8760 F1_9: 0.7453 F1_avg: 0.8106
Epoch  25 | Loss: 0.4861 | Acc18: 0.6789 F1_18: 0.7410 | F1_bin: 0.8823 F1_9: 0.7477 F1_avg: 0.8150
Epoch  26 | Loss: 0.4586 | Acc18: 0.6838 F1_18: 0.7463 | F1_bin: 0.8823 F1_9: 0.7527 F1_avg: 0.8175
Epoch  27 | Loss: 0.4364 | Acc18: 0.6863 F1_18: 0.7438 | F1_bin: 0.8827 F1_9: 0.7491 F1_avg: 0.8159
Epoch  28 | Loss: 0.3936 | Acc18: 0.6759 F1_18: 0.7333 | F1_bin: 0.8768 F1_9: 0.7453 F1_avg: 0.8110
Epoch  29 | Loss: 0.3942 | Acc18: 0.6844 F1_18: 0.7377 | F1_bin: 0.8809 F1_9: 0.7507 F1_avg: 0.8158
Epoch 00030: reducing learning rate of group 0 to 2.5000e-04.
Epoch  30 | Loss: 0.3705 | Acc18: 0.6820 F1_18: 0.7366 | F1_bin: 0.8870 F1_9: 0.7435 F1_avg: 0.8153
Epoch  31 | Loss: 0.3277 | Acc18: 0.6967 F1_18: 0.7458 | F1_bin: 0.8912 F1_9: 0.7552 F1_avg: 0.8232
Epoch  32 | Loss: 0.2948 | Acc18: 0.6991 F1_18: 0.7483 | F1_bin: 0.8885 F1_9: 0.7540 F1_avg: 0.8213
Epoch  33 | Loss: 0.2881 | Acc18: 0.7071 F1_18: 0.7607 | F1_bin: 0.8891 F1_9: 0.7690 F1_avg: 0.8291
Epoch  34 | Loss: 0.2694 | Acc18: 0.6906 F1_18: 0.7437 | F1_bin: 0.8993 F1_9: 0.7487 F1_avg: 0.8240
Epoch  35 | Loss: 0.2710 | Acc18: 0.6863 F1_18: 0.7431 | F1_bin: 0.8863 F1_9: 0.7507 F1_avg: 0.8185
Epoch  36 | Loss: 0.2555 | Acc18: 0.6961 F1_18: 0.7491 | F1_bin: 0.8940 F1_9: 0.7574 F1_avg: 0.8257
Epoch 00037: reducing learning rate of group 0 to 1.2500e-04.
Epoch  37 | Loss: 0.2483 | Acc18: 0.6814 F1_18: 0.7388 | F1_bin: 0.8858 F1_9: 0.7475 F1_avg: 0.8166
Epoch  38 | Loss: 0.2266 | Acc18: 0.6949 F1_18: 0.7460 | F1_bin: 0.8896 F1_9: 0.7580 F1_avg: 0.8238
Epoch  39 | Loss: 0.2065 | Acc18: 0.6924 F1_18: 0.7469 | F1_bin: 0.8906 F1_9: 0.7570 F1_avg: 0.8238
Epoch  40 | Loss: 0.1957 | Acc18: 0.6924 F1_18: 0.7480 | F1_bin: 0.8943 F1_9: 0.7582 F1_avg: 0.8263
Epoch 00041: reducing learning rate of group 0 to 6.2500e-05.
Epoch  41 | Loss: 0.1954 | Acc18: 0.6967 F1_18: 0.7510 | F1_bin: 0.8893 F1_9: 0.7570 F1_avg: 0.8231
Epoch  42 | Loss: 0.1906 | Acc18: 0.7010 F1_18: 0.7504 | F1_bin: 0.8988 F1_9: 0.7567 F1_avg: 0.8278
Epoch  43 | Loss: 0.1785 | Acc18: 0.6985 F1_18: 0.7504 | F1_bin: 0.8957 F1_9: 0.7564 F1_avg: 0.8261
Epoch  44 | Loss: 0.1765 | Acc18: 0.6942 F1_18: 0.7446 | F1_bin: 0.8944 F1_9: 0.7525 F1_avg: 0.8234
Epoch 00045: reducing learning rate of group 0 to 3.1250e-05.
Epoch  45 | Loss: 0.1784 | Acc18: 0.6942 F1_18: 0.7448 | F1_bin: 0.8973 F1_9: 0.7516 F1_avg: 0.8245
Epoch  46 | Loss: 0.1703 | Acc18: 0.7010 F1_18: 0.7530 | F1_bin: 0.8964 F1_9: 0.7609 F1_avg: 0.8286
Epoch  47 | Loss: 0.1642 | Acc18: 0.7004 F1_18: 0.7505 | F1_bin: 0.8964 F1_9: 0.7582 F1_avg: 0.8273
Epoch  48 | Loss: 0.1587 | Acc18: 0.7004 F1_18: 0.7502 | F1_bin: 0.8933 F1_9: 0.7596 F1_avg: 0.8264
Epoch 00049: reducing learning rate of group 0 to 1.5625e-05.
Epoch  49 | Loss: 0.1575 | Acc18: 0.6973 F1_18: 0.7454 | F1_bin: 0.8943 F1_9: 0.7534 F1_avg: 0.8239
Epoch  50 | Loss: 0.1573 | Acc18: 0.6991 F1_18: 0.7486 | F1_bin: 0.8963 F1_9: 0.7560 F1_avg: 0.8261
Epoch  51 | Loss: 0.1647 | Acc18: 0.6979 F1_18: 0.7470 | F1_bin: 0.8968 F1_9: 0.7537 F1_avg: 0.8253
Epoch  52 | Loss: 0.1522 | Acc18: 0.7022 F1_18: 0.7502 | F1_bin: 0.8933 F1_9: 0.7603 F1_avg: 0.8268
Epoch 00053: reducing learning rate of group 0 to 7.8125e-06.
Epoch  53 | Loss: 0.1557 | Acc18: 0.6998 F1_18: 0.7498 | F1_bin: 0.8927 F1_9: 0.7607 F1_avg: 0.8267
Early stopping.

=== Fold 2 ===
Epoch   1 | Loss: 1.9309 | Acc18: 0.3998 F1_18: 0.4141 | F1_bin: 0.7872 F1_9: 0.4594 F1_avg: 0.6233
Epoch   2 | Loss: 1.4456 | Acc18: 0.5218 F1_18: 0.5561 | F1_bin: 0.8076 F1_9: 0.5871 F1_avg: 0.6974
Epoch   3 | Loss: 1.2673 | Acc18: 0.5077 F1_18: 0.5472 | F1_bin: 0.8210 F1_9: 0.5756 F1_avg: 0.6983
Epoch   4 | Loss: 1.1697 | Acc18: 0.5500 F1_18: 0.6046 | F1_bin: 0.7820 F1_9: 0.6534 F1_avg: 0.7177
Epoch   5 | Loss: 1.0875 | Acc18: 0.5727 F1_18: 0.6204 | F1_bin: 0.8319 F1_9: 0.6527 F1_avg: 0.7423
Epoch   6 | Loss: 1.0420 | Acc18: 0.5837 F1_18: 0.6366 | F1_bin: 0.8246 F1_9: 0.6663 F1_avg: 0.7455
Epoch   7 | Loss: 1.0041 | Acc18: 0.5861 F1_18: 0.6261 | F1_bin: 0.8287 F1_9: 0.6655 F1_avg: 0.7471
Epoch   8 | Loss: 0.9532 | Acc18: 0.5586 F1_18: 0.6017 | F1_bin: 0.8250 F1_9: 0.6293 F1_avg: 0.7271
Epoch   9 | Loss: 0.9117 | Acc18: 0.5763 F1_18: 0.6231 | F1_bin: 0.8485 F1_9: 0.6405 F1_avg: 0.7445
Epoch  10 | Loss: 0.8939 | Acc18: 0.5868 F1_18: 0.6423 | F1_bin: 0.8343 F1_9: 0.6640 F1_avg: 0.7491
Epoch  11 | Loss: 0.8551 | Acc18: 0.5874 F1_18: 0.6442 | F1_bin: 0.8275 F1_9: 0.6689 F1_avg: 0.7482
Epoch  12 | Loss: 0.8253 | Acc18: 0.5690 F1_18: 0.6347 | F1_bin: 0.7863 F1_9: 0.6624 F1_avg: 0.7243
Epoch  13 | Loss: 0.7957 | Acc18: 0.5972 F1_18: 0.6481 | F1_bin: 0.8474 F1_9: 0.6714 F1_avg: 0.7594
Epoch  14 | Loss: 0.7843 | Acc18: 0.5794 F1_18: 0.6313 | F1_bin: 0.8085 F1_9: 0.6528 F1_avg: 0.7307
Epoch  15 | Loss: 0.7541 | Acc18: 0.5966 F1_18: 0.6574 | F1_bin: 0.8436 F1_9: 0.6732 F1_avg: 0.7584
Epoch  16 | Loss: 0.7134 | Acc18: 0.5923 F1_18: 0.6532 | F1_bin: 0.8251 F1_9: 0.6765 F1_avg: 0.7508
Epoch  17 | Loss: 0.6920 | Acc18: 0.5904 F1_18: 0.6473 | F1_bin: 0.8234 F1_9: 0.6756 F1_avg: 0.7495
Epoch  18 | Loss: 0.6755 | Acc18: 0.6002 F1_18: 0.6532 | F1_bin: 0.8307 F1_9: 0.6813 F1_avg: 0.7560
Epoch 00019: reducing learning rate of group 0 to 5.0000e-04.
Epoch  19 | Loss: 0.6649 | Acc18: 0.5868 F1_18: 0.6552 | F1_bin: 0.8069 F1_9: 0.6837 F1_avg: 0.7453
Epoch  20 | Loss: 0.5733 | Acc18: 0.6131 F1_18: 0.6692 | F1_bin: 0.8433 F1_9: 0.6925 F1_avg: 0.7679
Epoch  21 | Loss: 0.5169 | Acc18: 0.6021 F1_18: 0.6669 | F1_bin: 0.8289 F1_9: 0.6867 F1_avg: 0.7578
Epoch  22 | Loss: 0.5047 | Acc18: 0.6156 F1_18: 0.6801 | F1_bin: 0.8283 F1_9: 0.7032 F1_avg: 0.7658
Epoch  23 | Loss: 0.4860 | Acc18: 0.6107 F1_18: 0.6737 | F1_bin: 0.8259 F1_9: 0.6960 F1_avg: 0.7609
Epoch  24 | Loss: 0.4672 | Acc18: 0.6064 F1_18: 0.6715 | F1_bin: 0.8395 F1_9: 0.6864 F1_avg: 0.7629
Epoch  25 | Loss: 0.4499 | Acc18: 0.6143 F1_18: 0.6807 | F1_bin: 0.8221 F1_9: 0.7050 F1_avg: 0.7635
Epoch  26 | Loss: 0.4257 | Acc18: 0.5978 F1_18: 0.6534 | F1_bin: 0.8233 F1_9: 0.6755 F1_avg: 0.7494
Epoch  27 | Loss: 0.4209 | Acc18: 0.6137 F1_18: 0.6756 | F1_bin: 0.8237 F1_9: 0.6994 F1_avg: 0.7616
Epoch  28 | Loss: 0.3945 | Acc18: 0.6101 F1_18: 0.6750 | F1_bin: 0.8422 F1_9: 0.6946 F1_avg: 0.7684
Epoch 00029: reducing learning rate of group 0 to 2.5000e-04.
Epoch  29 | Loss: 0.3807 | Acc18: 0.6137 F1_18: 0.6761 | F1_bin: 0.8244 F1_9: 0.7001 F1_avg: 0.7623
Epoch  30 | Loss: 0.3422 | Acc18: 0.6045 F1_18: 0.6653 | F1_bin: 0.8252 F1_9: 0.6856 F1_avg: 0.7554
Epoch  31 | Loss: 0.3134 | Acc18: 0.6131 F1_18: 0.6775 | F1_bin: 0.8379 F1_9: 0.7002 F1_avg: 0.7691
Epoch  32 | Loss: 0.3012 | Acc18: 0.6174 F1_18: 0.6844 | F1_bin: 0.8398 F1_9: 0.7017 F1_avg: 0.7707
Epoch  33 | Loss: 0.2856 | Acc18: 0.6107 F1_18: 0.6756 | F1_bin: 0.8388 F1_9: 0.6927 F1_avg: 0.7657
Epoch  34 | Loss: 0.2753 | Acc18: 0.6193 F1_18: 0.6850 | F1_bin: 0.8458 F1_9: 0.7000 F1_avg: 0.7729
Epoch  35 | Loss: 0.2582 | Acc18: 0.6058 F1_18: 0.6691 | F1_bin: 0.8283 F1_9: 0.6903 F1_avg: 0.7593
Epoch  36 | Loss: 0.2482 | Acc18: 0.6137 F1_18: 0.6793 | F1_bin: 0.8423 F1_9: 0.6956 F1_avg: 0.7689
Epoch  37 | Loss: 0.2397 | Acc18: 0.6082 F1_18: 0.6785 | F1_bin: 0.8418 F1_9: 0.6922 F1_avg: 0.7670
Epoch 00038: reducing learning rate of group 0 to 1.2500e-04.
Epoch  38 | Loss: 0.2350 | Acc18: 0.6107 F1_18: 0.6785 | F1_bin: 0.8360 F1_9: 0.6968 F1_avg: 0.7664
Epoch  39 | Loss: 0.1983 | Acc18: 0.6156 F1_18: 0.6810 | F1_bin: 0.8368 F1_9: 0.6985 F1_avg: 0.7676
Epoch  40 | Loss: 0.1944 | Acc18: 0.6156 F1_18: 0.6783 | F1_bin: 0.8325 F1_9: 0.6999 F1_avg: 0.7662
Epoch  41 | Loss: 0.2015 | Acc18: 0.6162 F1_18: 0.6826 | F1_bin: 0.8391 F1_9: 0.6991 F1_avg: 0.7691
Epoch 00042: reducing learning rate of group 0 to 6.2500e-05.
Epoch  42 | Loss: 0.1909 | Acc18: 0.6205 F1_18: 0.6844 | F1_bin: 0.8380 F1_9: 0.7051 F1_avg: 0.7715
Epoch  43 | Loss: 0.1745 | Acc18: 0.6205 F1_18: 0.6863 | F1_bin: 0.8390 F1_9: 0.7062 F1_avg: 0.7726
Epoch  44 | Loss: 0.1687 | Acc18: 0.6242 F1_18: 0.6853 | F1_bin: 0.8362 F1_9: 0.7080 F1_avg: 0.7721
Epoch  45 | Loss: 0.1661 | Acc18: 0.6193 F1_18: 0.6827 | F1_bin: 0.8391 F1_9: 0.7035 F1_avg: 0.7713
Epoch  46 | Loss: 0.1592 | Acc18: 0.6205 F1_18: 0.6833 | F1_bin: 0.8354 F1_9: 0.7019 F1_avg: 0.7687
Epoch 00047: reducing learning rate of group 0 to 3.1250e-05.
Epoch  47 | Loss: 0.1566 | Acc18: 0.6125 F1_18: 0.6799 | F1_bin: 0.8264 F1_9: 0.7024 F1_avg: 0.7644
Epoch  48 | Loss: 0.1419 | Acc18: 0.6186 F1_18: 0.6815 | F1_bin: 0.8367 F1_9: 0.7008 F1_avg: 0.7688
Epoch  49 | Loss: 0.1432 | Acc18: 0.6242 F1_18: 0.6870 | F1_bin: 0.8372 F1_9: 0.7079 F1_avg: 0.7725
Epoch  50 | Loss: 0.1413 | Acc18: 0.6248 F1_18: 0.6852 | F1_bin: 0.8429 F1_9: 0.7078 F1_avg: 0.7754
Epoch  51 | Loss: 0.1471 | Acc18: 0.6211 F1_18: 0.6818 | F1_bin: 0.8380 F1_9: 0.7033 F1_avg: 0.7706
Epoch  52 | Loss: 0.1454 | Acc18: 0.6229 F1_18: 0.6837 | F1_bin: 0.8378 F1_9: 0.7046 F1_avg: 0.7712
Epoch 00053: reducing learning rate of group 0 to 1.5625e-05.
Epoch  53 | Loss: 0.1375 | Acc18: 0.6199 F1_18: 0.6831 | F1_bin: 0.8434 F1_9: 0.7018 F1_avg: 0.7726
Epoch  54 | Loss: 0.1438 | Acc18: 0.6211 F1_18: 0.6852 | F1_bin: 0.8398 F1_9: 0.7044 F1_avg: 0.7721
Epoch  55 | Loss: 0.1367 | Acc18: 0.6223 F1_18: 0.6846 | F1_bin: 0.8404 F1_9: 0.7046 F1_avg: 0.7725
Epoch  56 | Loss: 0.1340 | Acc18: 0.6193 F1_18: 0.6829 | F1_bin: 0.8355 F1_9: 0.7042 F1_avg: 0.7698
Epoch 00057: reducing learning rate of group 0 to 7.8125e-06.
Epoch  57 | Loss: 0.1384 | Acc18: 0.6186 F1_18: 0.6846 | F1_bin: 0.8361 F1_9: 0.7037 F1_avg: 0.7699
Epoch  58 | Loss: 0.1300 | Acc18: 0.6186 F1_18: 0.6844 | F1_bin: 0.8360 F1_9: 0.7030 F1_avg: 0.7695
Epoch  59 | Loss: 0.1424 | Acc18: 0.6168 F1_18: 0.6830 | F1_bin: 0.8349 F1_9: 0.7010 F1_avg: 0.7680
Epoch  60 | Loss: 0.1338 | Acc18: 0.6180 F1_18: 0.6829 | F1_bin: 0.8350 F1_9: 0.7034 F1_avg: 0.7692
Epoch 00061: reducing learning rate of group 0 to 3.9063e-06.
Epoch  61 | Loss: 0.1367 | Acc18: 0.6168 F1_18: 0.6803 | F1_bin: 0.8355 F1_9: 0.7004 F1_avg: 0.7680
Epoch  62 | Loss: 0.1283 | Acc18: 0.6235 F1_18: 0.6875 | F1_bin: 0.8392 F1_9: 0.7085 F1_avg: 0.7738
Epoch  63 | Loss: 0.1384 | Acc18: 0.6186 F1_18: 0.6829 | F1_bin: 0.8386 F1_9: 0.7028 F1_avg: 0.7707
Epoch  64 | Loss: 0.1369 | Acc18: 0.6254 F1_18: 0.6900 | F1_bin: 0.8404 F1_9: 0.7107 F1_avg: 0.7755
Epoch  65 | Loss: 0.1358 | Acc18: 0.6174 F1_18: 0.6818 | F1_bin: 0.8361 F1_9: 0.7012 F1_avg: 0.7686
Epoch  66 | Loss: 0.1304 | Acc18: 0.6217 F1_18: 0.6854 | F1_bin: 0.8343 F1_9: 0.7072 F1_avg: 0.7707
Epoch  67 | Loss: 0.1367 | Acc18: 0.6235 F1_18: 0.6866 | F1_bin: 0.8428 F1_9: 0.7057 F1_avg: 0.7742
Epoch 00068: reducing learning rate of group 0 to 1.9531e-06.
Epoch  68 | Loss: 0.1327 | Acc18: 0.6180 F1_18: 0.6840 | F1_bin: 0.8384 F1_9: 0.7009 F1_avg: 0.7697
Epoch  69 | Loss: 0.1268 | Acc18: 0.6150 F1_18: 0.6792 | F1_bin: 0.8313 F1_9: 0.7002 F1_avg: 0.7658
Epoch  70 | Loss: 0.1283 | Acc18: 0.6242 F1_18: 0.6875 | F1_bin: 0.8392 F1_9: 0.7072 F1_avg: 0.7732
Epoch  71 | Loss: 0.1327 | Acc18: 0.6205 F1_18: 0.6844 | F1_bin: 0.8380 F1_9: 0.7046 F1_avg: 0.7713
Epoch 00072: reducing learning rate of group 0 to 9.7656e-07.
Epoch  72 | Loss: 0.1350 | Acc18: 0.6217 F1_18: 0.6853 | F1_bin: 0.8380 F1_9: 0.7078 F1_avg: 0.7729
Epoch  73 | Loss: 0.1254 | Acc18: 0.6193 F1_18: 0.6830 | F1_bin: 0.8386 F1_9: 0.7028 F1_avg: 0.7707
Epoch  74 | Loss: 0.1336 | Acc18: 0.6174 F1_18: 0.6823 | F1_bin: 0.8368 F1_9: 0.7034 F1_avg: 0.7701
Epoch  75 | Loss: 0.1285 | Acc18: 0.6162 F1_18: 0.6804 | F1_bin: 0.8349 F1_9: 0.7007 F1_avg: 0.7678
Epoch 00076: reducing learning rate of group 0 to 4.8828e-07.
Epoch  76 | Loss: 0.1294 | Acc18: 0.6205 F1_18: 0.6843 | F1_bin: 0.8390 F1_9: 0.7042 F1_avg: 0.7716
Epoch  77 | Loss: 0.1288 | Acc18: 0.6180 F1_18: 0.6828 | F1_bin: 0.8360 F1_9: 0.7019 F1_avg: 0.7690
Epoch  78 | Loss: 0.1235 | Acc18: 0.6199 F1_18: 0.6849 | F1_bin: 0.8392 F1_9: 0.7056 F1_avg: 0.7724
Epoch  79 | Loss: 0.1376 | Acc18: 0.6186 F1_18: 0.6830 | F1_bin: 0.8398 F1_9: 0.7039 F1_avg: 0.7718
Epoch 00080: reducing learning rate of group 0 to 2.4414e-07.
Epoch  80 | Loss: 0.1274 | Acc18: 0.6162 F1_18: 0.6824 | F1_bin: 0.8361 F1_9: 0.7017 F1_avg: 0.7689
Epoch  81 | Loss: 0.1296 | Acc18: 0.6174 F1_18: 0.6832 | F1_bin: 0.8379 F1_9: 0.7018 F1_avg: 0.7699
Epoch  82 | Loss: 0.1352 | Acc18: 0.6217 F1_18: 0.6835 | F1_bin: 0.8439 F1_9: 0.7023 F1_avg: 0.7731
Epoch  83 | Loss: 0.1255 | Acc18: 0.6193 F1_18: 0.6830 | F1_bin: 0.8385 F1_9: 0.7038 F1_avg: 0.7712
Epoch 00084: reducing learning rate of group 0 to 1.2207e-07.
Epoch  84 | Loss: 0.1351 | Acc18: 0.6186 F1_18: 0.6808 | F1_bin: 0.8386 F1_9: 0.7022 F1_avg: 0.7704
Early stopping.

=== Fold 3 ===
Epoch   1 | Loss: 1.9386 | Acc18: 0.3866 F1_18: 0.4038 | F1_bin: 0.7350 F1_9: 0.4573 F1_avg: 0.5962
Epoch   2 | Loss: 1.4482 | Acc18: 0.5028 F1_18: 0.4891 | F1_bin: 0.7842 F1_9: 0.5468 F1_avg: 0.6655
Epoch   3 | Loss: 1.2921 | Acc18: 0.5446 F1_18: 0.5590 | F1_bin: 0.8218 F1_9: 0.5880 F1_avg: 0.7049
Epoch   4 | Loss: 1.1936 | Acc18: 0.5814 F1_18: 0.6028 | F1_bin: 0.8529 F1_9: 0.6364 F1_avg: 0.7447
Epoch   5 | Loss: 1.1044 | Acc18: 0.6183 F1_18: 0.6468 | F1_bin: 0.8638 F1_9: 0.6748 F1_avg: 0.7693
Epoch   6 | Loss: 1.0513 | Acc18: 0.5900 F1_18: 0.5990 | F1_bin: 0.8377 F1_9: 0.6312 F1_avg: 0.7345
Epoch   7 | Loss: 1.0012 | Acc18: 0.5925 F1_18: 0.6237 | F1_bin: 0.8340 F1_9: 0.6603 F1_avg: 0.7472
Epoch   8 | Loss: 0.9677 | Acc18: 0.6195 F1_18: 0.6462 | F1_bin: 0.8555 F1_9: 0.6775 F1_avg: 0.7665
Epoch   9 | Loss: 0.9346 | Acc18: 0.6214 F1_18: 0.6523 | F1_bin: 0.8227 F1_9: 0.6900 F1_avg: 0.7564
Epoch  10 | Loss: 0.8959 | Acc18: 0.6091 F1_18: 0.6265 | F1_bin: 0.8437 F1_9: 0.6677 F1_avg: 0.7557
Epoch  11 | Loss: 0.8564 | Acc18: 0.6349 F1_18: 0.6540 | F1_bin: 0.8568 F1_9: 0.6865 F1_avg: 0.7716
Epoch  12 | Loss: 0.8521 | Acc18: 0.6159 F1_18: 0.6592 | F1_bin: 0.8754 F1_9: 0.6798 F1_avg: 0.7776
Epoch  13 | Loss: 0.8183 | Acc18: 0.6361 F1_18: 0.6681 | F1_bin: 0.8453 F1_9: 0.6918 F1_avg: 0.7686
Epoch  14 | Loss: 0.7714 | Acc18: 0.6195 F1_18: 0.6451 | F1_bin: 0.8555 F1_9: 0.6731 F1_avg: 0.7643
Epoch  15 | Loss: 0.7578 | Acc18: 0.6355 F1_18: 0.6671 | F1_bin: 0.8613 F1_9: 0.6931 F1_avg: 0.7772
Epoch  16 | Loss: 0.7435 | Acc18: 0.6503 F1_18: 0.6821 | F1_bin: 0.8664 F1_9: 0.7055 F1_avg: 0.7860
Epoch  17 | Loss: 0.7032 | Acc18: 0.6398 F1_18: 0.6596 | F1_bin: 0.8710 F1_9: 0.6851 F1_avg: 0.7780
Epoch  18 | Loss: 0.6894 | Acc18: 0.6447 F1_18: 0.6651 | F1_bin: 0.8436 F1_9: 0.6941 F1_avg: 0.7688
Epoch  19 | Loss: 0.6706 | Acc18: 0.6552 F1_18: 0.6805 | F1_bin: 0.8735 F1_9: 0.7042 F1_avg: 0.7889
Epoch 00020: reducing learning rate of group 0 to 5.0000e-04.
Epoch  20 | Loss: 0.6520 | Acc18: 0.6546 F1_18: 0.6801 | F1_bin: 0.8688 F1_9: 0.7078 F1_avg: 0.7883
Epoch  21 | Loss: 0.5454 | Acc18: 0.6632 F1_18: 0.6901 | F1_bin: 0.8675 F1_9: 0.7189 F1_avg: 0.7932
Epoch  22 | Loss: 0.5006 | Acc18: 0.6675 F1_18: 0.7064 | F1_bin: 0.8772 F1_9: 0.7272 F1_avg: 0.8022
Epoch  23 | Loss: 0.4929 | Acc18: 0.6749 F1_18: 0.7066 | F1_bin: 0.8818 F1_9: 0.7254 F1_avg: 0.8036
Epoch  24 | Loss: 0.4553 | Acc18: 0.6638 F1_18: 0.7074 | F1_bin: 0.8696 F1_9: 0.7227 F1_avg: 0.7962
Epoch  25 | Loss: 0.4476 | Acc18: 0.6583 F1_18: 0.6911 | F1_bin: 0.8710 F1_9: 0.7143 F1_avg: 0.7927
Epoch  26 | Loss: 0.4138 | Acc18: 0.6644 F1_18: 0.7051 | F1_bin: 0.8726 F1_9: 0.7326 F1_avg: 0.8026
Epoch  27 | Loss: 0.4043 | Acc18: 0.6613 F1_18: 0.6918 | F1_bin: 0.8695 F1_9: 0.7151 F1_avg: 0.7923
Epoch 00028: reducing learning rate of group 0 to 2.5000e-04.
Epoch  28 | Loss: 0.4043 | Acc18: 0.6460 F1_18: 0.6727 | F1_bin: 0.8695 F1_9: 0.6986 F1_avg: 0.7840
Epoch  29 | Loss: 0.3338 | Acc18: 0.6742 F1_18: 0.7037 | F1_bin: 0.8784 F1_9: 0.7245 F1_avg: 0.8014
Epoch  30 | Loss: 0.3084 | Acc18: 0.6779 F1_18: 0.7122 | F1_bin: 0.8858 F1_9: 0.7290 F1_avg: 0.8074
Epoch  31 | Loss: 0.3057 | Acc18: 0.6650 F1_18: 0.6972 | F1_bin: 0.8753 F1_9: 0.7192 F1_avg: 0.7973
Epoch  32 | Loss: 0.2850 | Acc18: 0.6785 F1_18: 0.7173 | F1_bin: 0.8755 F1_9: 0.7381 F1_avg: 0.8068
Epoch  33 | Loss: 0.2757 | Acc18: 0.6638 F1_18: 0.7035 | F1_bin: 0.8716 F1_9: 0.7238 F1_avg: 0.7977
Epoch  34 | Loss: 0.2646 | Acc18: 0.6699 F1_18: 0.7073 | F1_bin: 0.8775 F1_9: 0.7260 F1_avg: 0.8017
Epoch  35 | Loss: 0.2469 | Acc18: 0.6607 F1_18: 0.7024 | F1_bin: 0.8758 F1_9: 0.7190 F1_avg: 0.7974
Epoch 00036: reducing learning rate of group 0 to 1.2500e-04.
Epoch  36 | Loss: 0.2454 | Acc18: 0.6546 F1_18: 0.6822 | F1_bin: 0.8710 F1_9: 0.7060 F1_avg: 0.7885
Epoch  37 | Loss: 0.2246 | Acc18: 0.6687 F1_18: 0.7002 | F1_bin: 0.8740 F1_9: 0.7199 F1_avg: 0.7970
Epoch  38 | Loss: 0.2107 | Acc18: 0.6681 F1_18: 0.7074 | F1_bin: 0.8770 F1_9: 0.7228 F1_avg: 0.7999
Epoch  39 | Loss: 0.2008 | Acc18: 0.6650 F1_18: 0.6990 | F1_bin: 0.8794 F1_9: 0.7165 F1_avg: 0.7980
Epoch 00040: reducing learning rate of group 0 to 6.2500e-05.
Epoch  40 | Loss: 0.1959 | Acc18: 0.6712 F1_18: 0.7100 | F1_bin: 0.8764 F1_9: 0.7282 F1_avg: 0.8023
Epoch  41 | Loss: 0.1756 | Acc18: 0.6687 F1_18: 0.7086 | F1_bin: 0.8778 F1_9: 0.7270 F1_avg: 0.8024
Epoch  42 | Loss: 0.1712 | Acc18: 0.6718 F1_18: 0.7120 | F1_bin: 0.8748 F1_9: 0.7322 F1_avg: 0.8035
Epoch  43 | Loss: 0.1779 | Acc18: 0.6749 F1_18: 0.7106 | F1_bin: 0.8759 F1_9: 0.7302 F1_avg: 0.8030
Epoch 00044: reducing learning rate of group 0 to 3.1250e-05.
Epoch  44 | Loss: 0.1677 | Acc18: 0.6779 F1_18: 0.7133 | F1_bin: 0.8825 F1_9: 0.7304 F1_avg: 0.8064
Epoch  45 | Loss: 0.1651 | Acc18: 0.6736 F1_18: 0.7090 | F1_bin: 0.8810 F1_9: 0.7264 F1_avg: 0.8037
Epoch  46 | Loss: 0.1647 | Acc18: 0.6699 F1_18: 0.7065 | F1_bin: 0.8725 F1_9: 0.7234 F1_avg: 0.7979
Epoch  47 | Loss: 0.1585 | Acc18: 0.6693 F1_18: 0.7073 | F1_bin: 0.8769 F1_9: 0.7247 F1_avg: 0.8008
Epoch 00048: reducing learning rate of group 0 to 1.5625e-05.
Epoch  48 | Loss: 0.1555 | Acc18: 0.6706 F1_18: 0.7085 | F1_bin: 0.8750 F1_9: 0.7257 F1_avg: 0.8004
Epoch  49 | Loss: 0.1531 | Acc18: 0.6736 F1_18: 0.7087 | F1_bin: 0.8761 F1_9: 0.7254 F1_avg: 0.8007
Epoch  50 | Loss: 0.1561 | Acc18: 0.6755 F1_18: 0.7118 | F1_bin: 0.8781 F1_9: 0.7282 F1_avg: 0.8031
Epoch  51 | Loss: 0.1593 | Acc18: 0.6761 F1_18: 0.7111 | F1_bin: 0.8837 F1_9: 0.7293 F1_avg: 0.8065
Epoch 00052: reducing learning rate of group 0 to 7.8125e-06.
Epoch  52 | Loss: 0.1543 | Acc18: 0.6749 F1_18: 0.7106 | F1_bin: 0.8777 F1_9: 0.7300 F1_avg: 0.8038
Early stopping.

=== Fold 4 ===
Epoch   1 | Loss: 1.9819 | Acc18: 0.4508 F1_18: 0.4648 | F1_bin: 0.7745 F1_9: 0.5052 F1_avg: 0.6398
Epoch   2 | Loss: 1.4637 | Acc18: 0.5412 F1_18: 0.5592 | F1_bin: 0.8109 F1_9: 0.5857 F1_avg: 0.6983
Epoch   3 | Loss: 1.2773 | Acc18: 0.5726 F1_18: 0.6091 | F1_bin: 0.8200 F1_9: 0.6358 F1_avg: 0.7279
Epoch   4 | Loss: 1.1741 | Acc18: 0.5271 F1_18: 0.5498 | F1_bin: 0.8066 F1_9: 0.5642 F1_avg: 0.6854
Epoch   5 | Loss: 1.1120 | Acc18: 0.5879 F1_18: 0.6184 | F1_bin: 0.8328 F1_9: 0.6455 F1_avg: 0.7391
Epoch   6 | Loss: 1.0408 | Acc18: 0.6132 F1_18: 0.6436 | F1_bin: 0.8473 F1_9: 0.6638 F1_avg: 0.7556
Epoch   7 | Loss: 0.9880 | Acc18: 0.6107 F1_18: 0.6411 | F1_bin: 0.8210 F1_9: 0.6698 F1_avg: 0.7454
Epoch   8 | Loss: 0.9514 | Acc18: 0.5984 F1_18: 0.6398 | F1_bin: 0.8396 F1_9: 0.6546 F1_avg: 0.7471
Epoch   9 | Loss: 0.9301 | Acc18: 0.6156 F1_18: 0.6548 | F1_bin: 0.8317 F1_9: 0.6802 F1_avg: 0.7560
Epoch  10 | Loss: 0.8744 | Acc18: 0.6199 F1_18: 0.6593 | F1_bin: 0.8499 F1_9: 0.6788 F1_avg: 0.7644
Epoch  11 | Loss: 0.8503 | Acc18: 0.6089 F1_18: 0.6479 | F1_bin: 0.8355 F1_9: 0.6714 F1_avg: 0.7534
Epoch  12 | Loss: 0.8176 | Acc18: 0.6267 F1_18: 0.6628 | F1_bin: 0.8453 F1_9: 0.6900 F1_avg: 0.7676
Epoch  13 | Loss: 0.8005 | Acc18: 0.6199 F1_18: 0.6660 | F1_bin: 0.8351 F1_9: 0.6897 F1_avg: 0.7624
Epoch  14 | Loss: 0.7648 | Acc18: 0.6292 F1_18: 0.6769 | F1_bin: 0.8614 F1_9: 0.6865 F1_avg: 0.7739
Epoch  15 | Loss: 0.7579 | Acc18: 0.6261 F1_18: 0.6723 | F1_bin: 0.8447 F1_9: 0.6911 F1_avg: 0.7679
Epoch  16 | Loss: 0.7194 | Acc18: 0.6378 F1_18: 0.6863 | F1_bin: 0.8557 F1_9: 0.7025 F1_avg: 0.7791
Epoch  17 | Loss: 0.6923 | Acc18: 0.6304 F1_18: 0.6793 | F1_bin: 0.8302 F1_9: 0.7021 F1_avg: 0.7661
Epoch  18 | Loss: 0.6803 | Acc18: 0.6181 F1_18: 0.6643 | F1_bin: 0.8402 F1_9: 0.6809 F1_avg: 0.7605
Epoch  19 | Loss: 0.6628 | Acc18: 0.6427 F1_18: 0.6861 | F1_bin: 0.8528 F1_9: 0.7096 F1_avg: 0.7812
Epoch  20 | Loss: 0.6357 | Acc18: 0.6371 F1_18: 0.6948 | F1_bin: 0.8415 F1_9: 0.7147 F1_avg: 0.7781
Epoch  21 | Loss: 0.6234 | Acc18: 0.6415 F1_18: 0.7017 | F1_bin: 0.8376 F1_9: 0.7269 F1_avg: 0.7823
Epoch  22 | Loss: 0.6083 | Acc18: 0.6248 F1_18: 0.6710 | F1_bin: 0.8418 F1_9: 0.6895 F1_avg: 0.7657
Epoch  23 | Loss: 0.5736 | Acc18: 0.6427 F1_18: 0.6916 | F1_bin: 0.8627 F1_9: 0.7019 F1_avg: 0.7823
Epoch  24 | Loss: 0.5657 | Acc18: 0.6298 F1_18: 0.6756 | F1_bin: 0.8524 F1_9: 0.6956 F1_avg: 0.7740
Epoch 00025: reducing learning rate of group 0 to 5.0000e-04.
Epoch  25 | Loss: 0.5579 | Acc18: 0.6378 F1_18: 0.6818 | F1_bin: 0.8382 F1_9: 0.7075 F1_avg: 0.7729
Epoch  26 | Loss: 0.4655 | Acc18: 0.6544 F1_18: 0.7111 | F1_bin: 0.8627 F1_9: 0.7277 F1_avg: 0.7952
Epoch  27 | Loss: 0.4212 | Acc18: 0.6335 F1_18: 0.6852 | F1_bin: 0.8503 F1_9: 0.7054 F1_avg: 0.7778
Epoch  28 | Loss: 0.3910 | Acc18: 0.6482 F1_18: 0.7045 | F1_bin: 0.8524 F1_9: 0.7199 F1_avg: 0.7861
Epoch  29 | Loss: 0.3773 | Acc18: 0.6415 F1_18: 0.6997 | F1_bin: 0.8474 F1_9: 0.7148 F1_avg: 0.7811
Epoch  30 | Loss: 0.3628 | Acc18: 0.6494 F1_18: 0.7112 | F1_bin: 0.8600 F1_9: 0.7220 F1_avg: 0.7910
Epoch  31 | Loss: 0.3303 | Acc18: 0.6384 F1_18: 0.6890 | F1_bin: 0.8574 F1_9: 0.7081 F1_avg: 0.7827
Epoch  32 | Loss: 0.3311 | Acc18: 0.6384 F1_18: 0.6925 | F1_bin: 0.8510 F1_9: 0.7042 F1_avg: 0.7776
Epoch  33 | Loss: 0.3179 | Acc18: 0.6494 F1_18: 0.6943 | F1_bin: 0.8630 F1_9: 0.7114 F1_avg: 0.7872
Epoch 00034: reducing learning rate of group 0 to 2.5000e-04.
Epoch  34 | Loss: 0.3055 | Acc18: 0.6513 F1_18: 0.7009 | F1_bin: 0.8574 F1_9: 0.7160 F1_avg: 0.7867
Epoch  35 | Loss: 0.2613 | Acc18: 0.6599 F1_18: 0.7065 | F1_bin: 0.8632 F1_9: 0.7249 F1_avg: 0.7941
Epoch  36 | Loss: 0.2353 | Acc18: 0.6482 F1_18: 0.6988 | F1_bin: 0.8635 F1_9: 0.7124 F1_avg: 0.7880
Epoch  37 | Loss: 0.2291 | Acc18: 0.6482 F1_18: 0.7068 | F1_bin: 0.8526 F1_9: 0.7240 F1_avg: 0.7883
Epoch 00038: reducing learning rate of group 0 to 1.2500e-04.
Epoch  38 | Loss: 0.2181 | Acc18: 0.6531 F1_18: 0.7030 | F1_bin: 0.8539 F1_9: 0.7179 F1_avg: 0.7859
Epoch  39 | Loss: 0.1892 | Acc18: 0.6488 F1_18: 0.7031 | F1_bin: 0.8576 F1_9: 0.7168 F1_avg: 0.7872
Epoch  40 | Loss: 0.1850 | Acc18: 0.6556 F1_18: 0.7066 | F1_bin: 0.8595 F1_9: 0.7227 F1_avg: 0.7911
Epoch  41 | Loss: 0.1743 | Acc18: 0.6550 F1_18: 0.7072 | F1_bin: 0.8627 F1_9: 0.7261 F1_avg: 0.7944
Epoch 00042: reducing learning rate of group 0 to 6.2500e-05.
Epoch  42 | Loss: 0.1730 | Acc18: 0.6427 F1_18: 0.6956 | F1_bin: 0.8575 F1_9: 0.7087 F1_avg: 0.7831
Epoch  43 | Loss: 0.1631 | Acc18: 0.6470 F1_18: 0.6976 | F1_bin: 0.8605 F1_9: 0.7129 F1_avg: 0.7867
Epoch  44 | Loss: 0.1572 | Acc18: 0.6519 F1_18: 0.7044 | F1_bin: 0.8643 F1_9: 0.7179 F1_avg: 0.7911
Epoch  45 | Loss: 0.1495 | Acc18: 0.6519 F1_18: 0.7012 | F1_bin: 0.8642 F1_9: 0.7177 F1_avg: 0.7910
Epoch 00046: reducing learning rate of group 0 to 3.1250e-05.
Epoch  46 | Loss: 0.1551 | Acc18: 0.6587 F1_18: 0.7104 | F1_bin: 0.8644 F1_9: 0.7250 F1_avg: 0.7947
Epoch  47 | Loss: 0.1497 | Acc18: 0.6544 F1_18: 0.7050 | F1_bin: 0.8644 F1_9: 0.7219 F1_avg: 0.7931
Epoch  48 | Loss: 0.1446 | Acc18: 0.6574 F1_18: 0.7086 | F1_bin: 0.8650 F1_9: 0.7231 F1_avg: 0.7941
Epoch  49 | Loss: 0.1327 | Acc18: 0.6531 F1_18: 0.7050 | F1_bin: 0.8625 F1_9: 0.7192 F1_avg: 0.7909
Epoch 00050: reducing learning rate of group 0 to 1.5625e-05.
Epoch  50 | Loss: 0.1442 | Acc18: 0.6568 F1_18: 0.7112 | F1_bin: 0.8620 F1_9: 0.7258 F1_avg: 0.7939
Epoch  51 | Loss: 0.1266 | Acc18: 0.6538 F1_18: 0.7068 | F1_bin: 0.8637 F1_9: 0.7199 F1_avg: 0.7918
Epoch  52 | Loss: 0.1329 | Acc18: 0.6507 F1_18: 0.7025 | F1_bin: 0.8601 F1_9: 0.7171 F1_avg: 0.7886
Epoch  53 | Loss: 0.1405 | Acc18: 0.6574 F1_18: 0.7087 | F1_bin: 0.8644 F1_9: 0.7231 F1_avg: 0.7938
Epoch 00054: reducing learning rate of group 0 to 7.8125e-06.
Epoch  54 | Loss: 0.1300 | Acc18: 0.6538 F1_18: 0.7062 | F1_bin: 0.8613 F1_9: 0.7207 F1_avg: 0.7910
Epoch  55 | Loss: 0.1264 | Acc18: 0.6531 F1_18: 0.7044 | F1_bin: 0.8625 F1_9: 0.7186 F1_avg: 0.7905
Epoch  56 | Loss: 0.1280 | Acc18: 0.6556 F1_18: 0.7099 | F1_bin: 0.8631 F1_9: 0.7239 F1_avg: 0.7935
Epoch  57 | Loss: 0.1360 | Acc18: 0.6519 F1_18: 0.7066 | F1_bin: 0.8595 F1_9: 0.7216 F1_avg: 0.7905
Epoch  58 | Loss: 0.1308 | Acc18: 0.6599 F1_18: 0.7118 | F1_bin: 0.8601 F1_9: 0.7273 F1_avg: 0.7937
Epoch  59 | Loss: 0.1241 | Acc18: 0.6519 F1_18: 0.7054 | F1_bin: 0.8607 F1_9: 0.7208 F1_avg: 0.7907
Epoch  60 | Loss: 0.1293 | Acc18: 0.6519 F1_18: 0.7045 | F1_bin: 0.8600 F1_9: 0.7188 F1_avg: 0.7894
Epoch  61 | Loss: 0.1268 | Acc18: 0.6538 F1_18: 0.7106 | F1_bin: 0.8612 F1_9: 0.7246 F1_avg: 0.7929
Epoch 00062: reducing learning rate of group 0 to 3.9063e-06.
Epoch  62 | Loss: 0.1309 | Acc18: 0.6513 F1_18: 0.7060 | F1_bin: 0.8600 F1_9: 0.7210 F1_avg: 0.7905
Epoch  63 | Loss: 0.1260 | Acc18: 0.6482 F1_18: 0.7042 | F1_bin: 0.8568 F1_9: 0.7195 F1_avg: 0.7882
Epoch  64 | Loss: 0.1333 | Acc18: 0.6550 F1_18: 0.7091 | F1_bin: 0.8631 F1_9: 0.7242 F1_avg: 0.7937
Epoch  65 | Loss: 0.1311 | Acc18: 0.6581 F1_18: 0.7119 | F1_bin: 0.8619 F1_9: 0.7274 F1_avg: 0.7947
Epoch  66 | Loss: 0.1271 | Acc18: 0.6568 F1_18: 0.7083 | F1_bin: 0.8632 F1_9: 0.7228 F1_avg: 0.7930
Epoch  67 | Loss: 0.1262 | Acc18: 0.6544 F1_18: 0.7107 | F1_bin: 0.8618 F1_9: 0.7240 F1_avg: 0.7929
Epoch  68 | Loss: 0.1294 | Acc18: 0.6544 F1_18: 0.7077 | F1_bin: 0.8601 F1_9: 0.7223 F1_avg: 0.7912
Epoch  69 | Loss: 0.1273 | Acc18: 0.6599 F1_18: 0.7125 | F1_bin: 0.8614 F1_9: 0.7279 F1_avg: 0.7946
Epoch  70 | Loss: 0.1258 | Acc18: 0.6550 F1_18: 0.7061 | F1_bin: 0.8588 F1_9: 0.7209 F1_avg: 0.7899
Epoch  71 | Loss: 0.1252 | Acc18: 0.6574 F1_18: 0.7088 | F1_bin: 0.8583 F1_9: 0.7242 F1_avg: 0.7912
Epoch  72 | Loss: 0.1254 | Acc18: 0.6544 F1_18: 0.7078 | F1_bin: 0.8612 F1_9: 0.7211 F1_avg: 0.7911
Epoch 00073: reducing learning rate of group 0 to 1.9531e-06.
Epoch  73 | Loss: 0.1246 | Acc18: 0.6513 F1_18: 0.7063 | F1_bin: 0.8581 F1_9: 0.7217 F1_avg: 0.7899
Epoch  74 | Loss: 0.1250 | Acc18: 0.6513 F1_18: 0.7017 | F1_bin: 0.8594 F1_9: 0.7168 F1_avg: 0.7881
Epoch  75 | Loss: 0.1299 | Acc18: 0.6501 F1_18: 0.7043 | F1_bin: 0.8564 F1_9: 0.7184 F1_avg: 0.7874
Epoch  76 | Loss: 0.1282 | Acc18: 0.6556 F1_18: 0.7097 | F1_bin: 0.8595 F1_9: 0.7249 F1_avg: 0.7922
Epoch 00077: reducing learning rate of group 0 to 9.7656e-07.
Epoch  77 | Loss: 0.1212 | Acc18: 0.6568 F1_18: 0.7077 | F1_bin: 0.8607 F1_9: 0.7204 F1_avg: 0.7905
Epoch  78 | Loss: 0.1188 | Acc18: 0.6519 F1_18: 0.7064 | F1_bin: 0.8625 F1_9: 0.7192 F1_avg: 0.7909
Epoch  79 | Loss: 0.1277 | Acc18: 0.6513 F1_18: 0.7052 | F1_bin: 0.8569 F1_9: 0.7223 F1_avg: 0.7896
Epoch  80 | Loss: 0.1232 | Acc18: 0.6556 F1_18: 0.7073 | F1_bin: 0.8612 F1_9: 0.7232 F1_avg: 0.7922
Epoch 00081: reducing learning rate of group 0 to 4.8828e-07.
Epoch  81 | Loss: 0.1203 | Acc18: 0.6538 F1_18: 0.7037 | F1_bin: 0.8577 F1_9: 0.7197 F1_avg: 0.7887
Epoch  82 | Loss: 0.1270 | Acc18: 0.6587 F1_18: 0.7106 | F1_bin: 0.8630 F1_9: 0.7246 F1_avg: 0.7938
Epoch  83 | Loss: 0.1265 | Acc18: 0.6501 F1_18: 0.7034 | F1_bin: 0.8545 F1_9: 0.7206 F1_avg: 0.7875
Epoch  84 | Loss: 0.1313 | Acc18: 0.6525 F1_18: 0.7061 | F1_bin: 0.8594 F1_9: 0.7200 F1_avg: 0.7897
Epoch 00085: reducing learning rate of group 0 to 2.4414e-07.
Epoch  85 | Loss: 0.1279 | Acc18: 0.6544 F1_18: 0.7055 | F1_bin: 0.8607 F1_9: 0.7209 F1_avg: 0.7908
Epoch  86 | Loss: 0.1245 | Acc18: 0.6525 F1_18: 0.7059 | F1_bin: 0.8606 F1_9: 0.7196 F1_avg: 0.7901
Epoch  87 | Loss: 0.1191 | Acc18: 0.6568 F1_18: 0.7080 | F1_bin: 0.8595 F1_9: 0.7224 F1_avg: 0.7909
Epoch  88 | Loss: 0.1153 | Acc18: 0.6550 F1_18: 0.7052 | F1_bin: 0.8601 F1_9: 0.7191 F1_avg: 0.7896
Epoch 00089: reducing learning rate of group 0 to 1.2207e-07.
Epoch  89 | Loss: 0.1319 | Acc18: 0.6538 F1_18: 0.7088 | F1_bin: 0.8594 F1_9: 0.7255 F1_avg: 0.7925
Early stopping.

=== Fold 5 ===
Epoch   1 | Loss: 1.8528 | Acc18: 0.4434 F1_18: 0.4614 | F1_bin: 0.7657 F1_9: 0.5005 F1_avg: 0.6331
Epoch   2 | Loss: 1.3818 | Acc18: 0.4391 F1_18: 0.4761 | F1_bin: 0.7152 F1_9: 0.5194 F1_avg: 0.6173
Epoch   3 | Loss: 1.2170 | Acc18: 0.5131 F1_18: 0.5367 | F1_bin: 0.8052 F1_9: 0.5833 F1_avg: 0.6943
Epoch   4 | Loss: 1.1234 | Acc18: 0.5407 F1_18: 0.5645 | F1_bin: 0.8166 F1_9: 0.5963 F1_avg: 0.7065
Epoch   5 | Loss: 1.0660 | Acc18: 0.5523 F1_18: 0.5774 | F1_bin: 0.8185 F1_9: 0.6120 F1_avg: 0.7153
Epoch   6 | Loss: 1.0160 | Acc18: 0.5731 F1_18: 0.5965 | F1_bin: 0.8415 F1_9: 0.6287 F1_avg: 0.7351
Epoch   7 | Loss: 0.9700 | Acc18: 0.5786 F1_18: 0.6220 | F1_bin: 0.8165 F1_9: 0.6393 F1_avg: 0.7279
Epoch   8 | Loss: 0.9317 | Acc18: 0.5743 F1_18: 0.6069 | F1_bin: 0.8294 F1_9: 0.6414 F1_avg: 0.7354
Epoch   9 | Loss: 0.8923 | Acc18: 0.5939 F1_18: 0.6256 | F1_bin: 0.8383 F1_9: 0.6483 F1_avg: 0.7433
Epoch  10 | Loss: 0.8752 | Acc18: 0.5688 F1_18: 0.6097 | F1_bin: 0.8416 F1_9: 0.6490 F1_avg: 0.7453
Epoch  11 | Loss: 0.8338 | Acc18: 0.5890 F1_18: 0.6311 | F1_bin: 0.8360 F1_9: 0.6637 F1_avg: 0.7498
Epoch  12 | Loss: 0.8114 | Acc18: 0.5890 F1_18: 0.6340 | F1_bin: 0.8379 F1_9: 0.6641 F1_avg: 0.7510
Epoch  13 | Loss: 0.7841 | Acc18: 0.5694 F1_18: 0.6124 | F1_bin: 0.8320 F1_9: 0.6397 F1_avg: 0.7359
Epoch  14 | Loss: 0.7593 | Acc18: 0.6043 F1_18: 0.6470 | F1_bin: 0.8411 F1_9: 0.6725 F1_avg: 0.7568
Epoch  15 | Loss: 0.7221 | Acc18: 0.6024 F1_18: 0.6414 | F1_bin: 0.8425 F1_9: 0.6661 F1_avg: 0.7543
Epoch  16 | Loss: 0.7087 | Acc18: 0.5859 F1_18: 0.6302 | F1_bin: 0.8317 F1_9: 0.6636 F1_avg: 0.7476
Epoch  17 | Loss: 0.6883 | Acc18: 0.6067 F1_18: 0.6387 | F1_bin: 0.8487 F1_9: 0.6649 F1_avg: 0.7568
Epoch 00018: reducing learning rate of group 0 to 5.0000e-04.
Epoch  18 | Loss: 0.6805 | Acc18: 0.5859 F1_18: 0.6284 | F1_bin: 0.8327 F1_9: 0.6540 F1_avg: 0.7433
Epoch  19 | Loss: 0.5976 | Acc18: 0.5890 F1_18: 0.6322 | F1_bin: 0.8403 F1_9: 0.6626 F1_avg: 0.7515
Epoch  20 | Loss: 0.5512 | Acc18: 0.6135 F1_18: 0.6554 | F1_bin: 0.8583 F1_9: 0.6763 F1_avg: 0.7673
Epoch  21 | Loss: 0.5174 | Acc18: 0.5927 F1_18: 0.6411 | F1_bin: 0.8420 F1_9: 0.6612 F1_avg: 0.7516
Epoch  22 | Loss: 0.4885 | Acc18: 0.5982 F1_18: 0.6492 | F1_bin: 0.8491 F1_9: 0.6761 F1_avg: 0.7626
Epoch  23 | Loss: 0.4863 | Acc18: 0.5945 F1_18: 0.6512 | F1_bin: 0.8447 F1_9: 0.6781 F1_avg: 0.7614
Epoch  24 | Loss: 0.4778 | Acc18: 0.6055 F1_18: 0.6567 | F1_bin: 0.8525 F1_9: 0.6813 F1_avg: 0.7669
Epoch  25 | Loss: 0.4619 | Acc18: 0.6086 F1_18: 0.6580 | F1_bin: 0.8506 F1_9: 0.6846 F1_avg: 0.7676
Epoch  26 | Loss: 0.4371 | Acc18: 0.5920 F1_18: 0.6400 | F1_bin: 0.8441 F1_9: 0.6653 F1_avg: 0.7547
Epoch  27 | Loss: 0.4304 | Acc18: 0.5994 F1_18: 0.6501 | F1_bin: 0.8539 F1_9: 0.6766 F1_avg: 0.7653
Epoch  28 | Loss: 0.4153 | Acc18: 0.5994 F1_18: 0.6479 | F1_bin: 0.8495 F1_9: 0.6748 F1_avg: 0.7622
Epoch  29 | Loss: 0.3930 | Acc18: 0.6067 F1_18: 0.6656 | F1_bin: 0.8481 F1_9: 0.6909 F1_avg: 0.7695
Epoch  30 | Loss: 0.3770 | Acc18: 0.5933 F1_18: 0.6462 | F1_bin: 0.8403 F1_9: 0.6739 F1_avg: 0.7571
Epoch  31 | Loss: 0.3629 | Acc18: 0.5994 F1_18: 0.6531 | F1_bin: 0.8470 F1_9: 0.6818 F1_avg: 0.7644
Epoch  32 | Loss: 0.3538 | Acc18: 0.5859 F1_18: 0.6404 | F1_bin: 0.8456 F1_9: 0.6679 F1_avg: 0.7568
Epoch 00033: reducing learning rate of group 0 to 2.5000e-04.
Epoch  33 | Loss: 0.3361 | Acc18: 0.5939 F1_18: 0.6506 | F1_bin: 0.8452 F1_9: 0.6751 F1_avg: 0.7602
Epoch  34 | Loss: 0.3030 | Acc18: 0.6043 F1_18: 0.6578 | F1_bin: 0.8553 F1_9: 0.6841 F1_avg: 0.7697
Epoch  35 | Loss: 0.2615 | Acc18: 0.6067 F1_18: 0.6570 | F1_bin: 0.8538 F1_9: 0.6851 F1_avg: 0.7695
Epoch  36 | Loss: 0.2474 | Acc18: 0.6086 F1_18: 0.6617 | F1_bin: 0.8598 F1_9: 0.6870 F1_avg: 0.7734
Epoch 00037: reducing learning rate of group 0 to 1.2500e-04.
Epoch  37 | Loss: 0.2393 | Acc18: 0.5976 F1_18: 0.6465 | F1_bin: 0.8499 F1_9: 0.6735 F1_avg: 0.7617
Epoch  38 | Loss: 0.2122 | Acc18: 0.6092 F1_18: 0.6613 | F1_bin: 0.8564 F1_9: 0.6872 F1_avg: 0.7718
Epoch  39 | Loss: 0.2029 | Acc18: 0.6055 F1_18: 0.6585 | F1_bin: 0.8487 F1_9: 0.6816 F1_avg: 0.7652
Epoch  40 | Loss: 0.2022 | Acc18: 0.6104 F1_18: 0.6613 | F1_bin: 0.8571 F1_9: 0.6861 F1_avg: 0.7716
Epoch 00041: reducing learning rate of group 0 to 6.2500e-05.
Epoch  41 | Loss: 0.1947 | Acc18: 0.6024 F1_18: 0.6559 | F1_bin: 0.8526 F1_9: 0.6811 F1_avg: 0.7668
Epoch  42 | Loss: 0.1827 | Acc18: 0.6006 F1_18: 0.6552 | F1_bin: 0.8513 F1_9: 0.6790 F1_avg: 0.7651
Epoch  43 | Loss: 0.1771 | Acc18: 0.6110 F1_18: 0.6597 | F1_bin: 0.8563 F1_9: 0.6849 F1_avg: 0.7706
Epoch  44 | Loss: 0.1653 | Acc18: 0.6073 F1_18: 0.6606 | F1_bin: 0.8573 F1_9: 0.6855 F1_avg: 0.7714
Epoch 00045: reducing learning rate of group 0 to 3.1250e-05.
Epoch  45 | Loss: 0.1677 | Acc18: 0.6110 F1_18: 0.6629 | F1_bin: 0.8619 F1_9: 0.6874 F1_avg: 0.7747
Epoch  46 | Loss: 0.1668 | Acc18: 0.6055 F1_18: 0.6593 | F1_bin: 0.8547 F1_9: 0.6840 F1_avg: 0.7694
Epoch  47 | Loss: 0.1616 | Acc18: 0.6086 F1_18: 0.6623 | F1_bin: 0.8546 F1_9: 0.6859 F1_avg: 0.7703
Epoch  48 | Loss: 0.1610 | Acc18: 0.6061 F1_18: 0.6593 | F1_bin: 0.8513 F1_9: 0.6831 F1_avg: 0.7672
Epoch 00049: reducing learning rate of group 0 to 1.5625e-05.
Epoch  49 | Loss: 0.1528 | Acc18: 0.6031 F1_18: 0.6587 | F1_bin: 0.8532 F1_9: 0.6847 F1_avg: 0.7690
Early stopping.
"""
