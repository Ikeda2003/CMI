#ModelVariant_LSTMGRU 特徴量に角速度を追加、thmはIMUに含まれないので削除　IMUonlyモデル
#モデル保存条件をf1_18からf1_avgに変更
# CV=0.794
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

# ========= 設定 =========
RAW_CSV = "train.csv"
SAVE_DIR = "train_test_18class_35"
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
    best_f1_avg, patience_counter = 0,0

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
Epoch   1 | Loss: 1.9931 | Acc18: 0.3131 F1_18: 0.3048 | F1_bin: 0.6130 F1_9: 0.3650 F1_avg: 0.4890
Epoch   2 | Loss: 1.4969 | Acc18: 0.5123 F1_18: 0.5175 | F1_bin: 0.7837 F1_9: 0.5737 F1_avg: 0.6787
Epoch   3 | Loss: 1.3266 | Acc18: 0.5319 F1_18: 0.5766 | F1_bin: 0.8137 F1_9: 0.6199 F1_avg: 0.7168
Epoch   4 | Loss: 1.2121 | Acc18: 0.5699 F1_18: 0.6149 | F1_bin: 0.8351 F1_9: 0.6342 F1_avg: 0.7347
Epoch   5 | Loss: 1.1424 | Acc18: 0.5888 F1_18: 0.6175 | F1_bin: 0.8321 F1_9: 0.6598 F1_avg: 0.7459
Epoch   6 | Loss: 1.0743 | Acc18: 0.6054 F1_18: 0.6477 | F1_bin: 0.8477 F1_9: 0.6639 F1_avg: 0.7558
Epoch   7 | Loss: 1.0211 | Acc18: 0.5809 F1_18: 0.5939 | F1_bin: 0.8565 F1_9: 0.6325 F1_avg: 0.7445
Epoch   8 | Loss: 0.9901 | Acc18: 0.6385 F1_18: 0.6792 | F1_bin: 0.8644 F1_9: 0.6948 F1_avg: 0.7796
Epoch   9 | Loss: 0.9523 | Acc18: 0.6183 F1_18: 0.6660 | F1_bin: 0.8595 F1_9: 0.6900 F1_avg: 0.7748
Epoch  10 | Loss: 0.9088 | Acc18: 0.6483 F1_18: 0.6837 | F1_bin: 0.8821 F1_9: 0.7090 F1_avg: 0.7955
Epoch  11 | Loss: 0.8831 | Acc18: 0.6489 F1_18: 0.7068 | F1_bin: 0.8659 F1_9: 0.7153 F1_avg: 0.7906
Epoch  12 | Loss: 0.8448 | Acc18: 0.6495 F1_18: 0.6852 | F1_bin: 0.8829 F1_9: 0.7088 F1_avg: 0.7959
Epoch  13 | Loss: 0.8202 | Acc18: 0.6287 F1_18: 0.6687 | F1_bin: 0.8735 F1_9: 0.6815 F1_avg: 0.7775
Epoch  14 | Loss: 0.7977 | Acc18: 0.6452 F1_18: 0.6840 | F1_bin: 0.8672 F1_9: 0.7071 F1_avg: 0.7872
Epoch 00015: reducing learning rate of group 0 to 5.0000e-04.
Epoch  15 | Loss: 0.7586 | Acc18: 0.6464 F1_18: 0.6796 | F1_bin: 0.8886 F1_9: 0.6998 F1_avg: 0.7942
Epoch  16 | Loss: 0.6631 | Acc18: 0.6728 F1_18: 0.7196 | F1_bin: 0.8848 F1_9: 0.7345 F1_avg: 0.8096
Epoch  17 | Loss: 0.6129 | Acc18: 0.6667 F1_18: 0.7189 | F1_bin: 0.8823 F1_9: 0.7273 F1_avg: 0.8048
Epoch  18 | Loss: 0.5785 | Acc18: 0.6808 F1_18: 0.7137 | F1_bin: 0.8856 F1_9: 0.7328 F1_avg: 0.8092
Epoch  19 | Loss: 0.5550 | Acc18: 0.6722 F1_18: 0.7183 | F1_bin: 0.8872 F1_9: 0.7318 F1_avg: 0.8095
Epoch 00020: reducing learning rate of group 0 to 2.5000e-04.
Epoch  20 | Loss: 0.5253 | Acc18: 0.6599 F1_18: 0.7103 | F1_bin: 0.8855 F1_9: 0.7199 F1_avg: 0.8027
Epoch  21 | Loss: 0.4608 | Acc18: 0.6826 F1_18: 0.7364 | F1_bin: 0.8899 F1_9: 0.7531 F1_avg: 0.8215
Epoch  22 | Loss: 0.4337 | Acc18: 0.6752 F1_18: 0.7243 | F1_bin: 0.8875 F1_9: 0.7390 F1_avg: 0.8133
Epoch  23 | Loss: 0.4066 | Acc18: 0.6808 F1_18: 0.7292 | F1_bin: 0.8889 F1_9: 0.7423 F1_avg: 0.8156
Epoch  24 | Loss: 0.3872 | Acc18: 0.6795 F1_18: 0.7259 | F1_bin: 0.8907 F1_9: 0.7452 F1_avg: 0.8179
Epoch 00025: reducing learning rate of group 0 to 1.2500e-04.
Epoch  25 | Loss: 0.3866 | Acc18: 0.6795 F1_18: 0.7292 | F1_bin: 0.8881 F1_9: 0.7388 F1_avg: 0.8134
Epoch  26 | Loss: 0.3340 | Acc18: 0.6900 F1_18: 0.7378 | F1_bin: 0.9006 F1_9: 0.7463 F1_avg: 0.8234
Epoch  27 | Loss: 0.3250 | Acc18: 0.6844 F1_18: 0.7309 | F1_bin: 0.8953 F1_9: 0.7416 F1_avg: 0.8184
Epoch  28 | Loss: 0.3174 | Acc18: 0.6893 F1_18: 0.7423 | F1_bin: 0.8957 F1_9: 0.7502 F1_avg: 0.8230
Epoch  29 | Loss: 0.3049 | Acc18: 0.6887 F1_18: 0.7333 | F1_bin: 0.8932 F1_9: 0.7454 F1_avg: 0.8193
Epoch  30 | Loss: 0.3081 | Acc18: 0.6887 F1_18: 0.7389 | F1_bin: 0.8955 F1_9: 0.7488 F1_avg: 0.8221
Epoch  31 | Loss: 0.2874 | Acc18: 0.6820 F1_18: 0.7319 | F1_bin: 0.8902 F1_9: 0.7434 F1_avg: 0.8168
Epoch 00032: reducing learning rate of group 0 to 6.2500e-05.
Epoch  32 | Loss: 0.2711 | Acc18: 0.6844 F1_18: 0.7340 | F1_bin: 0.8979 F1_9: 0.7435 F1_avg: 0.8207
Epoch  33 | Loss: 0.2622 | Acc18: 0.6924 F1_18: 0.7423 | F1_bin: 0.8986 F1_9: 0.7516 F1_avg: 0.8251
Epoch  34 | Loss: 0.2523 | Acc18: 0.6906 F1_18: 0.7374 | F1_bin: 0.8979 F1_9: 0.7491 F1_avg: 0.8235
Epoch  35 | Loss: 0.2503 | Acc18: 0.7004 F1_18: 0.7481 | F1_bin: 0.8980 F1_9: 0.7596 F1_avg: 0.8288
Epoch  36 | Loss: 0.2449 | Acc18: 0.6973 F1_18: 0.7450 | F1_bin: 0.9004 F1_9: 0.7546 F1_avg: 0.8275
Epoch  37 | Loss: 0.2348 | Acc18: 0.6893 F1_18: 0.7391 | F1_bin: 0.8893 F1_9: 0.7547 F1_avg: 0.8220
Epoch  38 | Loss: 0.2328 | Acc18: 0.6912 F1_18: 0.7406 | F1_bin: 0.8949 F1_9: 0.7502 F1_avg: 0.8225
Epoch 00039: reducing learning rate of group 0 to 3.1250e-05.
Epoch  39 | Loss: 0.2344 | Acc18: 0.6863 F1_18: 0.7364 | F1_bin: 0.8942 F1_9: 0.7474 F1_avg: 0.8208
Epoch  40 | Loss: 0.2139 | Acc18: 0.6930 F1_18: 0.7400 | F1_bin: 0.8980 F1_9: 0.7512 F1_avg: 0.8246
Epoch  41 | Loss: 0.2134 | Acc18: 0.6912 F1_18: 0.7383 | F1_bin: 0.8954 F1_9: 0.7484 F1_avg: 0.8219
Epoch  42 | Loss: 0.2143 | Acc18: 0.6893 F1_18: 0.7353 | F1_bin: 0.8967 F1_9: 0.7476 F1_avg: 0.8221
Epoch 00043: reducing learning rate of group 0 to 1.5625e-05.
Epoch  43 | Loss: 0.2124 | Acc18: 0.6912 F1_18: 0.7376 | F1_bin: 0.8928 F1_9: 0.7502 F1_avg: 0.8215
Epoch  44 | Loss: 0.2138 | Acc18: 0.6930 F1_18: 0.7411 | F1_bin: 0.8930 F1_9: 0.7519 F1_avg: 0.8225
Epoch  45 | Loss: 0.2099 | Acc18: 0.6906 F1_18: 0.7395 | F1_bin: 0.8911 F1_9: 0.7516 F1_avg: 0.8213
Epoch  46 | Loss: 0.2128 | Acc18: 0.6863 F1_18: 0.7362 | F1_bin: 0.8940 F1_9: 0.7473 F1_avg: 0.8206
Epoch 00047: reducing learning rate of group 0 to 7.8125e-06.
Epoch  47 | Loss: 0.1997 | Acc18: 0.6863 F1_18: 0.7331 | F1_bin: 0.8904 F1_9: 0.7450 F1_avg: 0.8177
Epoch  48 | Loss: 0.2113 | Acc18: 0.6942 F1_18: 0.7400 | F1_bin: 0.8972 F1_9: 0.7517 F1_avg: 0.8245
Epoch  49 | Loss: 0.1958 | Acc18: 0.6893 F1_18: 0.7375 | F1_bin: 0.8928 F1_9: 0.7475 F1_avg: 0.8202
Epoch  50 | Loss: 0.1951 | Acc18: 0.6900 F1_18: 0.7391 | F1_bin: 0.8910 F1_9: 0.7523 F1_avg: 0.8216
Epoch 00051: reducing learning rate of group 0 to 3.9063e-06.
Epoch  51 | Loss: 0.2008 | Acc18: 0.6924 F1_18: 0.7404 | F1_bin: 0.8981 F1_9: 0.7497 F1_avg: 0.8239
Epoch  52 | Loss: 0.1948 | Acc18: 0.6906 F1_18: 0.7382 | F1_bin: 0.8942 F1_9: 0.7484 F1_avg: 0.8213
Epoch  53 | Loss: 0.2020 | Acc18: 0.6893 F1_18: 0.7373 | F1_bin: 0.8961 F1_9: 0.7477 F1_avg: 0.8219
Epoch  54 | Loss: 0.1977 | Acc18: 0.6930 F1_18: 0.7394 | F1_bin: 0.8956 F1_9: 0.7515 F1_avg: 0.8235
Epoch 00055: reducing learning rate of group 0 to 1.9531e-06.
Epoch  55 | Loss: 0.1988 | Acc18: 0.6881 F1_18: 0.7356 | F1_bin: 0.8966 F1_9: 0.7468 F1_avg: 0.8217
Early stopping.

=== Fold 2 ===
Epoch   1 | Loss: 1.9255 | Acc18: 0.3863 F1_18: 0.4028 | F1_bin: 0.7512 F1_9: 0.4505 F1_avg: 0.6008
Epoch   2 | Loss: 1.4196 | Acc18: 0.5279 F1_18: 0.5612 | F1_bin: 0.8146 F1_9: 0.5943 F1_avg: 0.7044
Epoch   3 | Loss: 1.2510 | Acc18: 0.5463 F1_18: 0.5975 | F1_bin: 0.8184 F1_9: 0.6216 F1_avg: 0.7200
Epoch   4 | Loss: 1.1778 | Acc18: 0.5438 F1_18: 0.5749 | F1_bin: 0.8224 F1_9: 0.6117 F1_avg: 0.7170
Epoch   5 | Loss: 1.0888 | Acc18: 0.5536 F1_18: 0.5904 | F1_bin: 0.8101 F1_9: 0.6371 F1_avg: 0.7236
Epoch   6 | Loss: 1.0413 | Acc18: 0.5690 F1_18: 0.6173 | F1_bin: 0.8062 F1_9: 0.6528 F1_avg: 0.7295
Epoch   7 | Loss: 0.9854 | Acc18: 0.5782 F1_18: 0.6285 | F1_bin: 0.8380 F1_9: 0.6520 F1_avg: 0.7450
Epoch   8 | Loss: 0.9413 | Acc18: 0.5782 F1_18: 0.6191 | F1_bin: 0.8283 F1_9: 0.6486 F1_avg: 0.7384
Epoch   9 | Loss: 0.9002 | Acc18: 0.6033 F1_18: 0.6528 | F1_bin: 0.8326 F1_9: 0.6865 F1_avg: 0.7596
Epoch  10 | Loss: 0.8817 | Acc18: 0.5917 F1_18: 0.6436 | F1_bin: 0.8420 F1_9: 0.6681 F1_avg: 0.7551
Epoch  11 | Loss: 0.8405 | Acc18: 0.5966 F1_18: 0.6500 | F1_bin: 0.8526 F1_9: 0.6695 F1_avg: 0.7611
Epoch  12 | Loss: 0.8092 | Acc18: 0.5966 F1_18: 0.6401 | F1_bin: 0.8369 F1_9: 0.6692 F1_avg: 0.7530
Epoch 00013: reducing learning rate of group 0 to 5.0000e-04.
Epoch  13 | Loss: 0.7885 | Acc18: 0.6033 F1_18: 0.6426 | F1_bin: 0.8428 F1_9: 0.6755 F1_avg: 0.7591
Epoch  14 | Loss: 0.6905 | Acc18: 0.6107 F1_18: 0.6628 | F1_bin: 0.8308 F1_9: 0.6897 F1_avg: 0.7602
Epoch  15 | Loss: 0.6430 | Acc18: 0.6033 F1_18: 0.6539 | F1_bin: 0.8380 F1_9: 0.6767 F1_avg: 0.7573
Epoch  16 | Loss: 0.6293 | Acc18: 0.6242 F1_18: 0.6741 | F1_bin: 0.8458 F1_9: 0.7017 F1_avg: 0.7737
Epoch  17 | Loss: 0.6031 | Acc18: 0.6058 F1_18: 0.6551 | F1_bin: 0.8341 F1_9: 0.6803 F1_avg: 0.7572
Epoch  18 | Loss: 0.5810 | Acc18: 0.6193 F1_18: 0.6712 | F1_bin: 0.8344 F1_9: 0.6958 F1_avg: 0.7651
Epoch  19 | Loss: 0.5733 | Acc18: 0.6150 F1_18: 0.6741 | F1_bin: 0.8349 F1_9: 0.6943 F1_avg: 0.7646
Epoch  20 | Loss: 0.5496 | Acc18: 0.6088 F1_18: 0.6749 | F1_bin: 0.8374 F1_9: 0.6902 F1_avg: 0.7638
Epoch  21 | Loss: 0.5192 | Acc18: 0.6186 F1_18: 0.6797 | F1_bin: 0.8395 F1_9: 0.6964 F1_avg: 0.7680
Epoch  22 | Loss: 0.5057 | Acc18: 0.6045 F1_18: 0.6572 | F1_bin: 0.8362 F1_9: 0.6833 F1_avg: 0.7598
Epoch  23 | Loss: 0.4847 | Acc18: 0.6107 F1_18: 0.6776 | F1_bin: 0.8318 F1_9: 0.6952 F1_avg: 0.7635
Epoch  24 | Loss: 0.4814 | Acc18: 0.6107 F1_18: 0.6599 | F1_bin: 0.8356 F1_9: 0.6812 F1_avg: 0.7584
Epoch 00025: reducing learning rate of group 0 to 2.5000e-04.
Epoch  25 | Loss: 0.4546 | Acc18: 0.6156 F1_18: 0.6568 | F1_bin: 0.8388 F1_9: 0.6793 F1_avg: 0.7591
Epoch  26 | Loss: 0.4008 | Acc18: 0.6266 F1_18: 0.6885 | F1_bin: 0.8448 F1_9: 0.7058 F1_avg: 0.7753
Epoch  27 | Loss: 0.3649 | Acc18: 0.6180 F1_18: 0.6750 | F1_bin: 0.8350 F1_9: 0.6958 F1_avg: 0.7654
Epoch  28 | Loss: 0.3551 | Acc18: 0.6260 F1_18: 0.6780 | F1_bin: 0.8477 F1_9: 0.6957 F1_avg: 0.7717
Epoch  29 | Loss: 0.3358 | Acc18: 0.6248 F1_18: 0.6804 | F1_bin: 0.8405 F1_9: 0.6982 F1_avg: 0.7693
Epoch 00030: reducing learning rate of group 0 to 1.2500e-04.
Epoch  30 | Loss: 0.3258 | Acc18: 0.6217 F1_18: 0.6793 | F1_bin: 0.8394 F1_9: 0.6991 F1_avg: 0.7692
Epoch  31 | Loss: 0.2943 | Acc18: 0.6248 F1_18: 0.6815 | F1_bin: 0.8527 F1_9: 0.6950 F1_avg: 0.7739
Epoch  32 | Loss: 0.2782 | Acc18: 0.6297 F1_18: 0.6863 | F1_bin: 0.8472 F1_9: 0.7043 F1_avg: 0.7757
Epoch  33 | Loss: 0.2619 | Acc18: 0.6162 F1_18: 0.6675 | F1_bin: 0.8398 F1_9: 0.6876 F1_avg: 0.7637
Epoch 00034: reducing learning rate of group 0 to 6.2500e-05.
Epoch  34 | Loss: 0.2574 | Acc18: 0.6223 F1_18: 0.6742 | F1_bin: 0.8508 F1_9: 0.6917 F1_avg: 0.7712
Epoch  35 | Loss: 0.2406 | Acc18: 0.6229 F1_18: 0.6792 | F1_bin: 0.8496 F1_9: 0.6943 F1_avg: 0.7719
Epoch  36 | Loss: 0.2396 | Acc18: 0.6309 F1_18: 0.6846 | F1_bin: 0.8508 F1_9: 0.7028 F1_avg: 0.7768
Epoch  37 | Loss: 0.2301 | Acc18: 0.6217 F1_18: 0.6763 | F1_bin: 0.8470 F1_9: 0.6927 F1_avg: 0.7699
Epoch 00038: reducing learning rate of group 0 to 3.1250e-05.
Epoch  38 | Loss: 0.2234 | Acc18: 0.6303 F1_18: 0.6822 | F1_bin: 0.8533 F1_9: 0.6972 F1_avg: 0.7752
Epoch  39 | Loss: 0.2263 | Acc18: 0.6309 F1_18: 0.6845 | F1_bin: 0.8525 F1_9: 0.6977 F1_avg: 0.7751
Epoch  40 | Loss: 0.2175 | Acc18: 0.6309 F1_18: 0.6839 | F1_bin: 0.8514 F1_9: 0.7006 F1_avg: 0.7760
Epoch  41 | Loss: 0.2051 | Acc18: 0.6291 F1_18: 0.6820 | F1_bin: 0.8495 F1_9: 0.6994 F1_avg: 0.7744
Epoch 00042: reducing learning rate of group 0 to 1.5625e-05.
Epoch  42 | Loss: 0.2233 | Acc18: 0.6248 F1_18: 0.6775 | F1_bin: 0.8525 F1_9: 0.6935 F1_avg: 0.7730
Epoch  43 | Loss: 0.2140 | Acc18: 0.6254 F1_18: 0.6799 | F1_bin: 0.8483 F1_9: 0.6978 F1_avg: 0.7730
Epoch  44 | Loss: 0.2083 | Acc18: 0.6260 F1_18: 0.6789 | F1_bin: 0.8475 F1_9: 0.6959 F1_avg: 0.7717
Epoch  45 | Loss: 0.2050 | Acc18: 0.6327 F1_18: 0.6843 | F1_bin: 0.8531 F1_9: 0.7012 F1_avg: 0.7772
Epoch 00046: reducing learning rate of group 0 to 7.8125e-06.
Epoch  46 | Loss: 0.2031 | Acc18: 0.6266 F1_18: 0.6791 | F1_bin: 0.8488 F1_9: 0.6966 F1_avg: 0.7727
Epoch  47 | Loss: 0.1992 | Acc18: 0.6303 F1_18: 0.6844 | F1_bin: 0.8525 F1_9: 0.7004 F1_avg: 0.7764
Epoch  48 | Loss: 0.1998 | Acc18: 0.6327 F1_18: 0.6858 | F1_bin: 0.8544 F1_9: 0.7009 F1_avg: 0.7776
Epoch  49 | Loss: 0.2023 | Acc18: 0.6260 F1_18: 0.6790 | F1_bin: 0.8468 F1_9: 0.6964 F1_avg: 0.7716
Epoch 00050: reducing learning rate of group 0 to 3.9063e-06.
Epoch  50 | Loss: 0.1959 | Acc18: 0.6297 F1_18: 0.6832 | F1_bin: 0.8531 F1_9: 0.7006 F1_avg: 0.7769
Epoch  51 | Loss: 0.2023 | Acc18: 0.6297 F1_18: 0.6836 | F1_bin: 0.8519 F1_9: 0.6999 F1_avg: 0.7759
Epoch  52 | Loss: 0.1975 | Acc18: 0.6266 F1_18: 0.6801 | F1_bin: 0.8513 F1_9: 0.6965 F1_avg: 0.7739
Epoch  53 | Loss: 0.1894 | Acc18: 0.6272 F1_18: 0.6806 | F1_bin: 0.8519 F1_9: 0.6971 F1_avg: 0.7745
Epoch 00054: reducing learning rate of group 0 to 1.9531e-06.
Epoch  54 | Loss: 0.1945 | Acc18: 0.6272 F1_18: 0.6805 | F1_bin: 0.8531 F1_9: 0.6976 F1_avg: 0.7753
Epoch  55 | Loss: 0.1871 | Acc18: 0.6278 F1_18: 0.6810 | F1_bin: 0.8525 F1_9: 0.6971 F1_avg: 0.7748
Epoch  56 | Loss: 0.1972 | Acc18: 0.6254 F1_18: 0.6786 | F1_bin: 0.8506 F1_9: 0.6965 F1_avg: 0.7735
Epoch  57 | Loss: 0.1976 | Acc18: 0.6297 F1_18: 0.6826 | F1_bin: 0.8519 F1_9: 0.6989 F1_avg: 0.7754
Epoch 00058: reducing learning rate of group 0 to 9.7656e-07.
Epoch  58 | Loss: 0.1953 | Acc18: 0.6242 F1_18: 0.6772 | F1_bin: 0.8525 F1_9: 0.6921 F1_avg: 0.7723
Epoch  59 | Loss: 0.1894 | Acc18: 0.6297 F1_18: 0.6825 | F1_bin: 0.8519 F1_9: 0.6990 F1_avg: 0.7754
Epoch  60 | Loss: 0.1846 | Acc18: 0.6284 F1_18: 0.6814 | F1_bin: 0.8513 F1_9: 0.6980 F1_avg: 0.7746
Epoch  61 | Loss: 0.1978 | Acc18: 0.6278 F1_18: 0.6815 | F1_bin: 0.8495 F1_9: 0.6993 F1_avg: 0.7744
Epoch 00062: reducing learning rate of group 0 to 4.8828e-07.
Epoch  62 | Loss: 0.1943 | Acc18: 0.6260 F1_18: 0.6796 | F1_bin: 0.8519 F1_9: 0.6972 F1_avg: 0.7745
Epoch  63 | Loss: 0.1830 | Acc18: 0.6260 F1_18: 0.6787 | F1_bin: 0.8519 F1_9: 0.6951 F1_avg: 0.7735
Epoch  64 | Loss: 0.1983 | Acc18: 0.6315 F1_18: 0.6822 | F1_bin: 0.8525 F1_9: 0.7008 F1_avg: 0.7767
Epoch  65 | Loss: 0.2003 | Acc18: 0.6248 F1_18: 0.6781 | F1_bin: 0.8471 F1_9: 0.6955 F1_avg: 0.7713
Epoch 00066: reducing learning rate of group 0 to 2.4414e-07.
Epoch  66 | Loss: 0.1942 | Acc18: 0.6260 F1_18: 0.6790 | F1_bin: 0.8489 F1_9: 0.6952 F1_avg: 0.7720
Epoch  67 | Loss: 0.1922 | Acc18: 0.6235 F1_18: 0.6760 | F1_bin: 0.8494 F1_9: 0.6946 F1_avg: 0.7720
Epoch  68 | Loss: 0.1988 | Acc18: 0.6242 F1_18: 0.6771 | F1_bin: 0.8495 F1_9: 0.6955 F1_avg: 0.7725
Early stopping.

=== Fold 3 ===
Epoch   1 | Loss: 1.8893 | Acc18: 0.4634 F1_18: 0.4859 | F1_bin: 0.7350 F1_9: 0.5284 F1_avg: 0.6317
Epoch   2 | Loss: 1.4328 | Acc18: 0.5396 F1_18: 0.5555 | F1_bin: 0.8137 F1_9: 0.5912 F1_avg: 0.7025
Epoch   3 | Loss: 1.2697 | Acc18: 0.5556 F1_18: 0.5694 | F1_bin: 0.8205 F1_9: 0.5953 F1_avg: 0.7079
Epoch   4 | Loss: 1.1696 | Acc18: 0.5753 F1_18: 0.5900 | F1_bin: 0.8207 F1_9: 0.6301 F1_avg: 0.7254
Epoch   5 | Loss: 1.1064 | Acc18: 0.6066 F1_18: 0.6269 | F1_bin: 0.8401 F1_9: 0.6607 F1_avg: 0.7504
Epoch   6 | Loss: 1.0489 | Acc18: 0.5888 F1_18: 0.6104 | F1_bin: 0.8265 F1_9: 0.6416 F1_avg: 0.7340
Epoch   7 | Loss: 1.0008 | Acc18: 0.6165 F1_18: 0.6198 | F1_bin: 0.8535 F1_9: 0.6529 F1_avg: 0.7532
Epoch   8 | Loss: 0.9416 | Acc18: 0.6103 F1_18: 0.6430 | F1_bin: 0.8466 F1_9: 0.6654 F1_avg: 0.7560
Epoch   9 | Loss: 0.9332 | Acc18: 0.6325 F1_18: 0.6540 | F1_bin: 0.8402 F1_9: 0.6898 F1_avg: 0.7650
Epoch  10 | Loss: 0.8814 | Acc18: 0.6349 F1_18: 0.6773 | F1_bin: 0.8575 F1_9: 0.6999 F1_avg: 0.7787
Epoch  11 | Loss: 0.8602 | Acc18: 0.6226 F1_18: 0.6384 | F1_bin: 0.8365 F1_9: 0.6724 F1_avg: 0.7545
Epoch  12 | Loss: 0.8439 | Acc18: 0.6128 F1_18: 0.6470 | F1_bin: 0.8702 F1_9: 0.6762 F1_avg: 0.7732
Epoch  13 | Loss: 0.8128 | Acc18: 0.6398 F1_18: 0.6668 | F1_bin: 0.8607 F1_9: 0.6958 F1_avg: 0.7783
Epoch 00014: reducing learning rate of group 0 to 5.0000e-04.
Epoch  14 | Loss: 0.7892 | Acc18: 0.6478 F1_18: 0.6768 | F1_bin: 0.8408 F1_9: 0.7124 F1_avg: 0.7766
Epoch  15 | Loss: 0.6889 | Acc18: 0.6681 F1_18: 0.7046 | F1_bin: 0.8692 F1_9: 0.7221 F1_avg: 0.7956
Epoch  16 | Loss: 0.6307 | Acc18: 0.6681 F1_18: 0.6948 | F1_bin: 0.8662 F1_9: 0.7233 F1_avg: 0.7947
Epoch  17 | Loss: 0.6167 | Acc18: 0.6699 F1_18: 0.6844 | F1_bin: 0.8778 F1_9: 0.7169 F1_avg: 0.7973
Epoch  18 | Loss: 0.5972 | Acc18: 0.6583 F1_18: 0.6896 | F1_bin: 0.8781 F1_9: 0.7117 F1_avg: 0.7949
Epoch  19 | Loss: 0.5608 | Acc18: 0.6779 F1_18: 0.7195 | F1_bin: 0.8685 F1_9: 0.7380 F1_avg: 0.8033
Epoch  20 | Loss: 0.5505 | Acc18: 0.6669 F1_18: 0.6950 | F1_bin: 0.8586 F1_9: 0.7232 F1_avg: 0.7909
Epoch  21 | Loss: 0.5348 | Acc18: 0.6589 F1_18: 0.6959 | F1_bin: 0.8492 F1_9: 0.7232 F1_avg: 0.7862
Epoch  22 | Loss: 0.5281 | Acc18: 0.6583 F1_18: 0.6997 | F1_bin: 0.8624 F1_9: 0.7219 F1_avg: 0.7921
Epoch 00023: reducing learning rate of group 0 to 2.5000e-04.
Epoch  23 | Loss: 0.4899 | Acc18: 0.6601 F1_18: 0.6952 | F1_bin: 0.8530 F1_9: 0.7218 F1_avg: 0.7874
Epoch  24 | Loss: 0.4595 | Acc18: 0.6656 F1_18: 0.7004 | F1_bin: 0.8776 F1_9: 0.7231 F1_avg: 0.8004
Epoch  25 | Loss: 0.4136 | Acc18: 0.6644 F1_18: 0.6912 | F1_bin: 0.8684 F1_9: 0.7197 F1_avg: 0.7941
Epoch  26 | Loss: 0.3918 | Acc18: 0.6742 F1_18: 0.7068 | F1_bin: 0.8734 F1_9: 0.7294 F1_avg: 0.8014
Epoch 00027: reducing learning rate of group 0 to 1.2500e-04.
Epoch  27 | Loss: 0.3848 | Acc18: 0.6761 F1_18: 0.7038 | F1_bin: 0.8757 F1_9: 0.7269 F1_avg: 0.8013
Epoch  28 | Loss: 0.3442 | Acc18: 0.6718 F1_18: 0.7110 | F1_bin: 0.8727 F1_9: 0.7281 F1_avg: 0.8004
Epoch  29 | Loss: 0.3350 | Acc18: 0.6742 F1_18: 0.7092 | F1_bin: 0.8817 F1_9: 0.7285 F1_avg: 0.8051
Epoch  30 | Loss: 0.3154 | Acc18: 0.6736 F1_18: 0.7140 | F1_bin: 0.8742 F1_9: 0.7295 F1_avg: 0.8019
Epoch 00031: reducing learning rate of group 0 to 6.2500e-05.
Epoch  31 | Loss: 0.3185 | Acc18: 0.6730 F1_18: 0.7124 | F1_bin: 0.8734 F1_9: 0.7324 F1_avg: 0.8029
Epoch  32 | Loss: 0.2967 | Acc18: 0.6718 F1_18: 0.7109 | F1_bin: 0.8712 F1_9: 0.7315 F1_avg: 0.8014
Epoch  33 | Loss: 0.2893 | Acc18: 0.6755 F1_18: 0.7141 | F1_bin: 0.8743 F1_9: 0.7329 F1_avg: 0.8036
Epoch  34 | Loss: 0.2834 | Acc18: 0.6755 F1_18: 0.7113 | F1_bin: 0.8682 F1_9: 0.7350 F1_avg: 0.8016
Epoch 00035: reducing learning rate of group 0 to 3.1250e-05.
Epoch  35 | Loss: 0.2793 | Acc18: 0.6773 F1_18: 0.7109 | F1_bin: 0.8731 F1_9: 0.7365 F1_avg: 0.8048
Epoch  36 | Loss: 0.2766 | Acc18: 0.6712 F1_18: 0.7050 | F1_bin: 0.8766 F1_9: 0.7281 F1_avg: 0.8024
Epoch  37 | Loss: 0.2761 | Acc18: 0.6724 F1_18: 0.7099 | F1_bin: 0.8779 F1_9: 0.7302 F1_avg: 0.8041
Epoch  38 | Loss: 0.2662 | Acc18: 0.6718 F1_18: 0.7078 | F1_bin: 0.8790 F1_9: 0.7283 F1_avg: 0.8037
Epoch 00039: reducing learning rate of group 0 to 1.5625e-05.
Epoch  39 | Loss: 0.2696 | Acc18: 0.6742 F1_18: 0.7102 | F1_bin: 0.8767 F1_9: 0.7323 F1_avg: 0.8045
Epoch  40 | Loss: 0.2532 | Acc18: 0.6730 F1_18: 0.7133 | F1_bin: 0.8731 F1_9: 0.7333 F1_avg: 0.8032
Epoch  41 | Loss: 0.2586 | Acc18: 0.6706 F1_18: 0.7107 | F1_bin: 0.8749 F1_9: 0.7315 F1_avg: 0.8032
Epoch  42 | Loss: 0.2525 | Acc18: 0.6736 F1_18: 0.7072 | F1_bin: 0.8779 F1_9: 0.7306 F1_avg: 0.8043
Epoch 00043: reducing learning rate of group 0 to 7.8125e-06.
Epoch  43 | Loss: 0.2553 | Acc18: 0.6742 F1_18: 0.7156 | F1_bin: 0.8785 F1_9: 0.7339 F1_avg: 0.8062
Epoch  44 | Loss: 0.2527 | Acc18: 0.6706 F1_18: 0.7099 | F1_bin: 0.8759 F1_9: 0.7305 F1_avg: 0.8032
Epoch  45 | Loss: 0.2598 | Acc18: 0.6761 F1_18: 0.7144 | F1_bin: 0.8799 F1_9: 0.7322 F1_avg: 0.8060
Epoch  46 | Loss: 0.2462 | Acc18: 0.6712 F1_18: 0.7087 | F1_bin: 0.8760 F1_9: 0.7282 F1_avg: 0.8021
Epoch 00047: reducing learning rate of group 0 to 3.9063e-06.
Epoch  47 | Loss: 0.2403 | Acc18: 0.6712 F1_18: 0.7096 | F1_bin: 0.8766 F1_9: 0.7307 F1_avg: 0.8037
Epoch  48 | Loss: 0.2524 | Acc18: 0.6730 F1_18: 0.7138 | F1_bin: 0.8761 F1_9: 0.7325 F1_avg: 0.8043
Epoch  49 | Loss: 0.2450 | Acc18: 0.6693 F1_18: 0.7074 | F1_bin: 0.8767 F1_9: 0.7281 F1_avg: 0.8024
Epoch  50 | Loss: 0.2480 | Acc18: 0.6675 F1_18: 0.7030 | F1_bin: 0.8785 F1_9: 0.7240 F1_avg: 0.8012
Epoch 00051: reducing learning rate of group 0 to 1.9531e-06.
Epoch  51 | Loss: 0.2445 | Acc18: 0.6730 F1_18: 0.7125 | F1_bin: 0.8767 F1_9: 0.7331 F1_avg: 0.8049
Epoch  52 | Loss: 0.2477 | Acc18: 0.6724 F1_18: 0.7121 | F1_bin: 0.8767 F1_9: 0.7315 F1_avg: 0.8041
Epoch  53 | Loss: 0.2438 | Acc18: 0.6749 F1_18: 0.7146 | F1_bin: 0.8743 F1_9: 0.7360 F1_avg: 0.8051
Epoch  54 | Loss: 0.2445 | Acc18: 0.6755 F1_18: 0.7161 | F1_bin: 0.8779 F1_9: 0.7342 F1_avg: 0.8060
Epoch 00055: reducing learning rate of group 0 to 9.7656e-07.
Epoch  55 | Loss: 0.2356 | Acc18: 0.6742 F1_18: 0.7102 | F1_bin: 0.8761 F1_9: 0.7333 F1_avg: 0.8047
Epoch  56 | Loss: 0.2494 | Acc18: 0.6724 F1_18: 0.7137 | F1_bin: 0.8742 F1_9: 0.7353 F1_avg: 0.8048
Epoch  57 | Loss: 0.2521 | Acc18: 0.6736 F1_18: 0.7101 | F1_bin: 0.8780 F1_9: 0.7305 F1_avg: 0.8042
Epoch  58 | Loss: 0.2469 | Acc18: 0.6706 F1_18: 0.7073 | F1_bin: 0.8767 F1_9: 0.7286 F1_avg: 0.8026
Epoch 00059: reducing learning rate of group 0 to 4.8828e-07.
Epoch  59 | Loss: 0.2412 | Acc18: 0.6749 F1_18: 0.7122 | F1_bin: 0.8766 F1_9: 0.7351 F1_avg: 0.8058
Epoch  60 | Loss: 0.2478 | Acc18: 0.6755 F1_18: 0.7139 | F1_bin: 0.8725 F1_9: 0.7372 F1_avg: 0.8048
Epoch  61 | Loss: 0.2479 | Acc18: 0.6724 F1_18: 0.7080 | F1_bin: 0.8779 F1_9: 0.7281 F1_avg: 0.8030
Epoch  62 | Loss: 0.2446 | Acc18: 0.6749 F1_18: 0.7147 | F1_bin: 0.8798 F1_9: 0.7331 F1_avg: 0.8064
Epoch 00063: reducing learning rate of group 0 to 2.4414e-07.
Epoch  63 | Loss: 0.2431 | Acc18: 0.6730 F1_18: 0.7124 | F1_bin: 0.8761 F1_9: 0.7329 F1_avg: 0.8045
Epoch  64 | Loss: 0.2502 | Acc18: 0.6730 F1_18: 0.7103 | F1_bin: 0.8779 F1_9: 0.7314 F1_avg: 0.8046
Epoch  65 | Loss: 0.2398 | Acc18: 0.6742 F1_18: 0.7112 | F1_bin: 0.8797 F1_9: 0.7310 F1_avg: 0.8054
Epoch  66 | Loss: 0.2464 | Acc18: 0.6718 F1_18: 0.7100 | F1_bin: 0.8767 F1_9: 0.7327 F1_avg: 0.8047
Epoch 00067: reducing learning rate of group 0 to 1.2207e-07.
Epoch  67 | Loss: 0.2515 | Acc18: 0.6761 F1_18: 0.7132 | F1_bin: 0.8767 F1_9: 0.7344 F1_avg: 0.8055
Epoch  68 | Loss: 0.2498 | Acc18: 0.6712 F1_18: 0.7087 | F1_bin: 0.8748 F1_9: 0.7315 F1_avg: 0.8032
Epoch  69 | Loss: 0.2435 | Acc18: 0.6724 F1_18: 0.7130 | F1_bin: 0.8779 F1_9: 0.7335 F1_avg: 0.8057
Epoch  70 | Loss: 0.2457 | Acc18: 0.6724 F1_18: 0.7101 | F1_bin: 0.8778 F1_9: 0.7302 F1_avg: 0.8040
Epoch 00071: reducing learning rate of group 0 to 6.1035e-08.
Epoch  71 | Loss: 0.2407 | Acc18: 0.6730 F1_18: 0.7127 | F1_bin: 0.8779 F1_9: 0.7303 F1_avg: 0.8041
Epoch  72 | Loss: 0.2498 | Acc18: 0.6724 F1_18: 0.7057 | F1_bin: 0.8791 F1_9: 0.7274 F1_avg: 0.8033
Epoch  73 | Loss: 0.2448 | Acc18: 0.6706 F1_18: 0.7089 | F1_bin: 0.8767 F1_9: 0.7291 F1_avg: 0.8029
Epoch  74 | Loss: 0.2445 | Acc18: 0.6755 F1_18: 0.7121 | F1_bin: 0.8779 F1_9: 0.7330 F1_avg: 0.8054
Epoch 00075: reducing learning rate of group 0 to 3.0518e-08.
Epoch  75 | Loss: 0.2444 | Acc18: 0.6755 F1_18: 0.7125 | F1_bin: 0.8779 F1_9: 0.7351 F1_avg: 0.8065
Epoch  76 | Loss: 0.2481 | Acc18: 0.6718 F1_18: 0.7105 | F1_bin: 0.8804 F1_9: 0.7299 F1_avg: 0.8051
Epoch  77 | Loss: 0.2478 | Acc18: 0.6724 F1_18: 0.7092 | F1_bin: 0.8767 F1_9: 0.7291 F1_avg: 0.8029
Epoch  78 | Loss: 0.2340 | Acc18: 0.6798 F1_18: 0.7173 | F1_bin: 0.8768 F1_9: 0.7388 F1_avg: 0.8078
Epoch 00079: reducing learning rate of group 0 to 1.5259e-08.
Epoch  79 | Loss: 0.2443 | Acc18: 0.6730 F1_18: 0.7120 | F1_bin: 0.8755 F1_9: 0.7315 F1_avg: 0.8035
Epoch  80 | Loss: 0.2482 | Acc18: 0.6699 F1_18: 0.7084 | F1_bin: 0.8754 F1_9: 0.7294 F1_avg: 0.8024
Epoch  81 | Loss: 0.2463 | Acc18: 0.6742 F1_18: 0.7141 | F1_bin: 0.8755 F1_9: 0.7327 F1_avg: 0.8041
Epoch  82 | Loss: 0.2466 | Acc18: 0.6706 F1_18: 0.7096 | F1_bin: 0.8779 F1_9: 0.7291 F1_avg: 0.8035
Epoch  83 | Loss: 0.2424 | Acc18: 0.6736 F1_18: 0.7105 | F1_bin: 0.8768 F1_9: 0.7303 F1_avg: 0.8035
Epoch  84 | Loss: 0.2421 | Acc18: 0.6724 F1_18: 0.7107 | F1_bin: 0.8779 F1_9: 0.7320 F1_avg: 0.8049
Epoch  85 | Loss: 0.2389 | Acc18: 0.6712 F1_18: 0.7075 | F1_bin: 0.8779 F1_9: 0.7282 F1_avg: 0.8030
Epoch  86 | Loss: 0.2421 | Acc18: 0.6730 F1_18: 0.7089 | F1_bin: 0.8785 F1_9: 0.7312 F1_avg: 0.8048
Epoch  87 | Loss: 0.2466 | Acc18: 0.6749 F1_18: 0.7129 | F1_bin: 0.8786 F1_9: 0.7335 F1_avg: 0.8060
Epoch  88 | Loss: 0.2446 | Acc18: 0.6706 F1_18: 0.7090 | F1_bin: 0.8779 F1_9: 0.7282 F1_avg: 0.8030
Epoch  89 | Loss: 0.2355 | Acc18: 0.6730 F1_18: 0.7097 | F1_bin: 0.8760 F1_9: 0.7302 F1_avg: 0.8031
Epoch  90 | Loss: 0.2447 | Acc18: 0.6712 F1_18: 0.7079 | F1_bin: 0.8767 F1_9: 0.7298 F1_avg: 0.8032
Epoch  91 | Loss: 0.2422 | Acc18: 0.6724 F1_18: 0.7112 | F1_bin: 0.8785 F1_9: 0.7312 F1_avg: 0.8048
Epoch  92 | Loss: 0.2423 | Acc18: 0.6724 F1_18: 0.7124 | F1_bin: 0.8760 F1_9: 0.7334 F1_avg: 0.8047
Epoch  93 | Loss: 0.2456 | Acc18: 0.6755 F1_18: 0.7129 | F1_bin: 0.8791 F1_9: 0.7319 F1_avg: 0.8055
Epoch  94 | Loss: 0.2535 | Acc18: 0.6699 F1_18: 0.7074 | F1_bin: 0.8760 F1_9: 0.7281 F1_avg: 0.8021
Epoch  95 | Loss: 0.2514 | Acc18: 0.6724 F1_18: 0.7106 | F1_bin: 0.8767 F1_9: 0.7304 F1_avg: 0.8036
Epoch  96 | Loss: 0.2476 | Acc18: 0.6730 F1_18: 0.7065 | F1_bin: 0.8791 F1_9: 0.7285 F1_avg: 0.8038
Epoch  97 | Loss: 0.2405 | Acc18: 0.6724 F1_18: 0.7096 | F1_bin: 0.8780 F1_9: 0.7301 F1_avg: 0.8041
Epoch  98 | Loss: 0.2518 | Acc18: 0.6706 F1_18: 0.7034 | F1_bin: 0.8772 F1_9: 0.7282 F1_avg: 0.8027
Early stopping.

=== Fold 4 ===
Epoch   1 | Loss: 1.9138 | Acc18: 0.4717 F1_18: 0.4785 | F1_bin: 0.7884 F1_9: 0.5221 F1_avg: 0.6552
Epoch   2 | Loss: 1.4188 | Acc18: 0.5074 F1_18: 0.5320 | F1_bin: 0.8029 F1_9: 0.5529 F1_avg: 0.6779
Epoch   3 | Loss: 1.2580 | Acc18: 0.5713 F1_18: 0.6003 | F1_bin: 0.8136 F1_9: 0.6388 F1_avg: 0.7262
Epoch   4 | Loss: 1.1610 | Acc18: 0.5726 F1_18: 0.5911 | F1_bin: 0.8154 F1_9: 0.6149 F1_avg: 0.7151
Epoch   5 | Loss: 1.0779 | Acc18: 0.5843 F1_18: 0.6127 | F1_bin: 0.8042 F1_9: 0.6361 F1_avg: 0.7202
Epoch   6 | Loss: 1.0257 | Acc18: 0.5990 F1_18: 0.6031 | F1_bin: 0.8240 F1_9: 0.6345 F1_avg: 0.7293
Epoch   7 | Loss: 0.9875 | Acc18: 0.6255 F1_18: 0.6653 | F1_bin: 0.8444 F1_9: 0.6809 F1_avg: 0.7626
Epoch   8 | Loss: 0.9435 | Acc18: 0.6076 F1_18: 0.6422 | F1_bin: 0.8327 F1_9: 0.6578 F1_avg: 0.7453
Epoch   9 | Loss: 0.9087 | Acc18: 0.6310 F1_18: 0.6684 | F1_bin: 0.8308 F1_9: 0.6869 F1_avg: 0.7589
Epoch  10 | Loss: 0.8797 | Acc18: 0.6212 F1_18: 0.6642 | F1_bin: 0.8423 F1_9: 0.6856 F1_avg: 0.7639
Epoch  11 | Loss: 0.8384 | Acc18: 0.6058 F1_18: 0.6471 | F1_bin: 0.8258 F1_9: 0.6729 F1_avg: 0.7493
Epoch  12 | Loss: 0.8075 | Acc18: 0.6224 F1_18: 0.6793 | F1_bin: 0.8362 F1_9: 0.6987 F1_avg: 0.7675
Epoch  13 | Loss: 0.7889 | Acc18: 0.6230 F1_18: 0.6661 | F1_bin: 0.8424 F1_9: 0.6839 F1_avg: 0.7632
Epoch  14 | Loss: 0.7493 | Acc18: 0.6039 F1_18: 0.6579 | F1_bin: 0.8278 F1_9: 0.6776 F1_avg: 0.7527
Epoch  15 | Loss: 0.7399 | Acc18: 0.6039 F1_18: 0.6580 | F1_bin: 0.8309 F1_9: 0.6710 F1_avg: 0.7509
Epoch 00016: reducing learning rate of group 0 to 5.0000e-04.
Epoch  16 | Loss: 0.7191 | Acc18: 0.6273 F1_18: 0.6779 | F1_bin: 0.8462 F1_9: 0.6940 F1_avg: 0.7701
Epoch  17 | Loss: 0.6194 | Acc18: 0.6568 F1_18: 0.7087 | F1_bin: 0.8591 F1_9: 0.7179 F1_avg: 0.7885
Epoch  18 | Loss: 0.5789 | Acc18: 0.6396 F1_18: 0.6911 | F1_bin: 0.8499 F1_9: 0.7062 F1_avg: 0.7781
Epoch  19 | Loss: 0.5458 | Acc18: 0.6451 F1_18: 0.6982 | F1_bin: 0.8542 F1_9: 0.7062 F1_avg: 0.7802
Epoch  20 | Loss: 0.5233 | Acc18: 0.6562 F1_18: 0.7105 | F1_bin: 0.8622 F1_9: 0.7193 F1_avg: 0.7907
Epoch  21 | Loss: 0.5007 | Acc18: 0.6433 F1_18: 0.6942 | F1_bin: 0.8643 F1_9: 0.7064 F1_avg: 0.7854
Epoch  22 | Loss: 0.4725 | Acc18: 0.6458 F1_18: 0.7021 | F1_bin: 0.8450 F1_9: 0.7144 F1_avg: 0.7797
Epoch  23 | Loss: 0.4647 | Acc18: 0.6470 F1_18: 0.6927 | F1_bin: 0.8481 F1_9: 0.7111 F1_avg: 0.7796
Epoch 00024: reducing learning rate of group 0 to 2.5000e-04.
Epoch  24 | Loss: 0.4557 | Acc18: 0.6439 F1_18: 0.6920 | F1_bin: 0.8475 F1_9: 0.7114 F1_avg: 0.7795
Epoch  25 | Loss: 0.3971 | Acc18: 0.6550 F1_18: 0.7063 | F1_bin: 0.8604 F1_9: 0.7161 F1_avg: 0.7882
Epoch  26 | Loss: 0.3620 | Acc18: 0.6538 F1_18: 0.7107 | F1_bin: 0.8579 F1_9: 0.7252 F1_avg: 0.7915
Epoch  27 | Loss: 0.3490 | Acc18: 0.6482 F1_18: 0.6978 | F1_bin: 0.8542 F1_9: 0.7133 F1_avg: 0.7837
Epoch  28 | Loss: 0.3425 | Acc18: 0.6556 F1_18: 0.7078 | F1_bin: 0.8651 F1_9: 0.7180 F1_avg: 0.7916
Epoch  29 | Loss: 0.3259 | Acc18: 0.6488 F1_18: 0.7006 | F1_bin: 0.8548 F1_9: 0.7147 F1_avg: 0.7847
Epoch 00030: reducing learning rate of group 0 to 1.2500e-04.
Epoch  30 | Loss: 0.3196 | Acc18: 0.6439 F1_18: 0.6981 | F1_bin: 0.8499 F1_9: 0.7112 F1_avg: 0.7806
Epoch  31 | Loss: 0.2820 | Acc18: 0.6605 F1_18: 0.7201 | F1_bin: 0.8688 F1_9: 0.7261 F1_avg: 0.7974
Epoch  32 | Loss: 0.2644 | Acc18: 0.6544 F1_18: 0.7088 | F1_bin: 0.8555 F1_9: 0.7227 F1_avg: 0.7891
Epoch  33 | Loss: 0.2604 | Acc18: 0.6581 F1_18: 0.7154 | F1_bin: 0.8627 F1_9: 0.7280 F1_avg: 0.7953
Epoch  34 | Loss: 0.2523 | Acc18: 0.6519 F1_18: 0.7091 | F1_bin: 0.8615 F1_9: 0.7211 F1_avg: 0.7913
Epoch 00035: reducing learning rate of group 0 to 6.2500e-05.
Epoch  35 | Loss: 0.2458 | Acc18: 0.6513 F1_18: 0.7056 | F1_bin: 0.8596 F1_9: 0.7179 F1_avg: 0.7887
Epoch  36 | Loss: 0.2225 | Acc18: 0.6556 F1_18: 0.7115 | F1_bin: 0.8578 F1_9: 0.7237 F1_avg: 0.7907
Epoch  37 | Loss: 0.2218 | Acc18: 0.6531 F1_18: 0.7067 | F1_bin: 0.8645 F1_9: 0.7164 F1_avg: 0.7905
Epoch  38 | Loss: 0.2246 | Acc18: 0.6525 F1_18: 0.7099 | F1_bin: 0.8638 F1_9: 0.7201 F1_avg: 0.7919
Epoch 00039: reducing learning rate of group 0 to 3.1250e-05.
Epoch  39 | Loss: 0.2219 | Acc18: 0.6482 F1_18: 0.7010 | F1_bin: 0.8590 F1_9: 0.7119 F1_avg: 0.7854
Epoch  40 | Loss: 0.2115 | Acc18: 0.6482 F1_18: 0.7061 | F1_bin: 0.8572 F1_9: 0.7183 F1_avg: 0.7878
Epoch  41 | Loss: 0.2090 | Acc18: 0.6482 F1_18: 0.7043 | F1_bin: 0.8584 F1_9: 0.7178 F1_avg: 0.7881
Epoch  42 | Loss: 0.1956 | Acc18: 0.6488 F1_18: 0.7050 | F1_bin: 0.8573 F1_9: 0.7199 F1_avg: 0.7886
Epoch 00043: reducing learning rate of group 0 to 1.5625e-05.
Epoch  43 | Loss: 0.1987 | Acc18: 0.6519 F1_18: 0.7085 | F1_bin: 0.8590 F1_9: 0.7219 F1_avg: 0.7905
Epoch  44 | Loss: 0.1953 | Acc18: 0.6525 F1_18: 0.7096 | F1_bin: 0.8584 F1_9: 0.7224 F1_avg: 0.7904
Epoch  45 | Loss: 0.1928 | Acc18: 0.6531 F1_18: 0.7090 | F1_bin: 0.8603 F1_9: 0.7214 F1_avg: 0.7908
Epoch  46 | Loss: 0.1947 | Acc18: 0.6513 F1_18: 0.7092 | F1_bin: 0.8609 F1_9: 0.7207 F1_avg: 0.7908
Epoch 00047: reducing learning rate of group 0 to 7.8125e-06.
Epoch  47 | Loss: 0.1865 | Acc18: 0.6513 F1_18: 0.7071 | F1_bin: 0.8645 F1_9: 0.7202 F1_avg: 0.7924
Epoch  48 | Loss: 0.1823 | Acc18: 0.6464 F1_18: 0.7018 | F1_bin: 0.8579 F1_9: 0.7149 F1_avg: 0.7864
Epoch  49 | Loss: 0.1885 | Acc18: 0.6525 F1_18: 0.7079 | F1_bin: 0.8652 F1_9: 0.7211 F1_avg: 0.7931
Epoch  50 | Loss: 0.1893 | Acc18: 0.6531 F1_18: 0.7087 | F1_bin: 0.8621 F1_9: 0.7235 F1_avg: 0.7928
Epoch 00051: reducing learning rate of group 0 to 3.9063e-06.
Epoch  51 | Loss: 0.1922 | Acc18: 0.6513 F1_18: 0.7071 | F1_bin: 0.8579 F1_9: 0.7214 F1_avg: 0.7896
Early stopping.

=== Fold 5 ===
Epoch   1 | Loss: 1.9139 | Acc18: 0.4391 F1_18: 0.4713 | F1_bin: 0.7549 F1_9: 0.5103 F1_avg: 0.6326
Epoch   2 | Loss: 1.4061 | Acc18: 0.5254 F1_18: 0.5668 | F1_bin: 0.8112 F1_9: 0.5958 F1_avg: 0.7035
Epoch   3 | Loss: 1.2404 | Acc18: 0.5309 F1_18: 0.5658 | F1_bin: 0.8043 F1_9: 0.6008 F1_avg: 0.7025
Epoch   4 | Loss: 1.1459 | Acc18: 0.5627 F1_18: 0.5869 | F1_bin: 0.8131 F1_9: 0.6283 F1_avg: 0.7207
Epoch   5 | Loss: 1.0879 | Acc18: 0.5517 F1_18: 0.5803 | F1_bin: 0.8043 F1_9: 0.6133 F1_avg: 0.7088
Epoch   6 | Loss: 1.0289 | Acc18: 0.5688 F1_18: 0.6149 | F1_bin: 0.8222 F1_9: 0.6410 F1_avg: 0.7316
Epoch   7 | Loss: 0.9707 | Acc18: 0.5657 F1_18: 0.6040 | F1_bin: 0.8319 F1_9: 0.6306 F1_avg: 0.7313
Epoch   8 | Loss: 0.9358 | Acc18: 0.5523 F1_18: 0.5873 | F1_bin: 0.8024 F1_9: 0.6085 F1_avg: 0.7055
Epoch   9 | Loss: 0.9049 | Acc18: 0.5872 F1_18: 0.6155 | F1_bin: 0.8302 F1_9: 0.6489 F1_avg: 0.7396
Epoch  10 | Loss: 0.8735 | Acc18: 0.5927 F1_18: 0.6295 | F1_bin: 0.8389 F1_9: 0.6590 F1_avg: 0.7489
Epoch  11 | Loss: 0.8414 | Acc18: 0.5951 F1_18: 0.6179 | F1_bin: 0.8304 F1_9: 0.6596 F1_avg: 0.7450
Epoch  12 | Loss: 0.8029 | Acc18: 0.5872 F1_18: 0.6305 | F1_bin: 0.8196 F1_9: 0.6615 F1_avg: 0.7405
Epoch  13 | Loss: 0.7701 | Acc18: 0.5994 F1_18: 0.6400 | F1_bin: 0.8446 F1_9: 0.6672 F1_avg: 0.7559
Epoch  14 | Loss: 0.7569 | Acc18: 0.6061 F1_18: 0.6445 | F1_bin: 0.8352 F1_9: 0.6704 F1_avg: 0.7528
Epoch  15 | Loss: 0.7299 | Acc18: 0.5725 F1_18: 0.6175 | F1_bin: 0.8311 F1_9: 0.6504 F1_avg: 0.7408
Epoch  16 | Loss: 0.7022 | Acc18: 0.5963 F1_18: 0.6389 | F1_bin: 0.8429 F1_9: 0.6619 F1_avg: 0.7524
Epoch  17 | Loss: 0.6634 | Acc18: 0.5976 F1_18: 0.6390 | F1_bin: 0.8437 F1_9: 0.6688 F1_avg: 0.7563
Epoch  18 | Loss: 0.6461 | Acc18: 0.5982 F1_18: 0.6448 | F1_bin: 0.8423 F1_9: 0.6692 F1_avg: 0.7558
Epoch  19 | Loss: 0.6434 | Acc18: 0.5884 F1_18: 0.6302 | F1_bin: 0.8059 F1_9: 0.6551 F1_avg: 0.7305
Epoch  20 | Loss: 0.6163 | Acc18: 0.6049 F1_18: 0.6439 | F1_bin: 0.8316 F1_9: 0.6684 F1_avg: 0.7500
Epoch  21 | Loss: 0.6134 | Acc18: 0.5768 F1_18: 0.6332 | F1_bin: 0.8029 F1_9: 0.6584 F1_avg: 0.7307
Epoch  22 | Loss: 0.5908 | Acc18: 0.6000 F1_18: 0.6516 | F1_bin: 0.8359 F1_9: 0.6765 F1_avg: 0.7562
Epoch  23 | Loss: 0.5716 | Acc18: 0.5841 F1_18: 0.6337 | F1_bin: 0.8214 F1_9: 0.6605 F1_avg: 0.7409
Epoch  24 | Loss: 0.5598 | Acc18: 0.5963 F1_18: 0.6453 | F1_bin: 0.8484 F1_9: 0.6708 F1_avg: 0.7596
Epoch  25 | Loss: 0.5353 | Acc18: 0.5976 F1_18: 0.6426 | F1_bin: 0.8496 F1_9: 0.6684 F1_avg: 0.7590
Epoch 00026: reducing learning rate of group 0 to 5.0000e-04.
Epoch  26 | Loss: 0.5320 | Acc18: 0.6012 F1_18: 0.6476 | F1_bin: 0.8444 F1_9: 0.6731 F1_avg: 0.7587
Epoch  27 | Loss: 0.4263 | Acc18: 0.6104 F1_18: 0.6640 | F1_bin: 0.8495 F1_9: 0.6892 F1_avg: 0.7693
Epoch  28 | Loss: 0.3805 | Acc18: 0.6116 F1_18: 0.6580 | F1_bin: 0.8546 F1_9: 0.6781 F1_avg: 0.7664
Epoch  29 | Loss: 0.3681 | Acc18: 0.6043 F1_18: 0.6497 | F1_bin: 0.8445 F1_9: 0.6751 F1_avg: 0.7598
Epoch  30 | Loss: 0.3535 | Acc18: 0.6080 F1_18: 0.6611 | F1_bin: 0.8447 F1_9: 0.6852 F1_avg: 0.7650
Epoch 00031: reducing learning rate of group 0 to 2.5000e-04.
Epoch  31 | Loss: 0.3244 | Acc18: 0.6110 F1_18: 0.6584 | F1_bin: 0.8480 F1_9: 0.6805 F1_avg: 0.7643
Epoch  32 | Loss: 0.2902 | Acc18: 0.6135 F1_18: 0.6505 | F1_bin: 0.8501 F1_9: 0.6813 F1_avg: 0.7657
Epoch  33 | Loss: 0.2708 | Acc18: 0.6092 F1_18: 0.6577 | F1_bin: 0.8485 F1_9: 0.6813 F1_avg: 0.7649
Epoch  34 | Loss: 0.2493 | Acc18: 0.6037 F1_18: 0.6562 | F1_bin: 0.8485 F1_9: 0.6802 F1_avg: 0.7643
Epoch 00035: reducing learning rate of group 0 to 1.2500e-04.
Epoch  35 | Loss: 0.2426 | Acc18: 0.5988 F1_18: 0.6486 | F1_bin: 0.8423 F1_9: 0.6732 F1_avg: 0.7577
Epoch  36 | Loss: 0.2180 | Acc18: 0.6171 F1_18: 0.6643 | F1_bin: 0.8506 F1_9: 0.6884 F1_avg: 0.7695
Epoch  37 | Loss: 0.2131 | Acc18: 0.6183 F1_18: 0.6659 | F1_bin: 0.8507 F1_9: 0.6932 F1_avg: 0.7719
Epoch  38 | Loss: 0.2038 | Acc18: 0.6159 F1_18: 0.6648 | F1_bin: 0.8462 F1_9: 0.6905 F1_avg: 0.7684
Epoch  39 | Loss: 0.2014 | Acc18: 0.6214 F1_18: 0.6637 | F1_bin: 0.8516 F1_9: 0.6916 F1_avg: 0.7716
Epoch  40 | Loss: 0.1847 | Acc18: 0.6153 F1_18: 0.6624 | F1_bin: 0.8427 F1_9: 0.6908 F1_avg: 0.7667
Epoch 00041: reducing learning rate of group 0 to 6.2500e-05.
Epoch  41 | Loss: 0.1869 | Acc18: 0.6171 F1_18: 0.6658 | F1_bin: 0.8441 F1_9: 0.6909 F1_avg: 0.7675
Epoch  42 | Loss: 0.1831 | Acc18: 0.6147 F1_18: 0.6571 | F1_bin: 0.8469 F1_9: 0.6837 F1_avg: 0.7653
Epoch  43 | Loss: 0.1740 | Acc18: 0.6171 F1_18: 0.6614 | F1_bin: 0.8481 F1_9: 0.6910 F1_avg: 0.7695
Epoch  44 | Loss: 0.1703 | Acc18: 0.6214 F1_18: 0.6663 | F1_bin: 0.8534 F1_9: 0.6958 F1_avg: 0.7746
Epoch  45 | Loss: 0.1633 | Acc18: 0.6214 F1_18: 0.6651 | F1_bin: 0.8493 F1_9: 0.6905 F1_avg: 0.7699
Epoch  46 | Loss: 0.1572 | Acc18: 0.6147 F1_18: 0.6630 | F1_bin: 0.8429 F1_9: 0.6882 F1_avg: 0.7656
Epoch  47 | Loss: 0.1638 | Acc18: 0.6159 F1_18: 0.6653 | F1_bin: 0.8456 F1_9: 0.6911 F1_avg: 0.7683
Epoch 00048: reducing learning rate of group 0 to 3.1250e-05.
Epoch  48 | Loss: 0.1515 | Acc18: 0.6061 F1_18: 0.6556 | F1_bin: 0.8413 F1_9: 0.6818 F1_avg: 0.7615
Epoch  49 | Loss: 0.1502 | Acc18: 0.6135 F1_18: 0.6638 | F1_bin: 0.8449 F1_9: 0.6899 F1_avg: 0.7674
Epoch  50 | Loss: 0.1422 | Acc18: 0.6153 F1_18: 0.6640 | F1_bin: 0.8473 F1_9: 0.6900 F1_avg: 0.7687
Epoch  51 | Loss: 0.1441 | Acc18: 0.6104 F1_18: 0.6586 | F1_bin: 0.8443 F1_9: 0.6864 F1_avg: 0.7654
Epoch 00052: reducing learning rate of group 0 to 1.5625e-05.
Epoch  52 | Loss: 0.1534 | Acc18: 0.6153 F1_18: 0.6656 | F1_bin: 0.8456 F1_9: 0.6938 F1_avg: 0.7697
Epoch  53 | Loss: 0.1420 | Acc18: 0.6086 F1_18: 0.6567 | F1_bin: 0.8426 F1_9: 0.6827 F1_avg: 0.7627
Epoch  54 | Loss: 0.1361 | Acc18: 0.6159 F1_18: 0.6599 | F1_bin: 0.8481 F1_9: 0.6888 F1_avg: 0.7684
Epoch  55 | Loss: 0.1418 | Acc18: 0.6147 F1_18: 0.6624 | F1_bin: 0.8473 F1_9: 0.6888 F1_avg: 0.7681
Epoch 00056: reducing learning rate of group 0 to 7.8125e-06.
Epoch  56 | Loss: 0.1392 | Acc18: 0.6147 F1_18: 0.6600 | F1_bin: 0.8473 F1_9: 0.6890 F1_avg: 0.7681
Epoch  57 | Loss: 0.1356 | Acc18: 0.6159 F1_18: 0.6613 | F1_bin: 0.8468 F1_9: 0.6903 F1_avg: 0.7686
Epoch  58 | Loss: 0.1319 | Acc18: 0.6122 F1_18: 0.6574 | F1_bin: 0.8468 F1_9: 0.6852 F1_avg: 0.7660
Epoch  59 | Loss: 0.1361 | Acc18: 0.6128 F1_18: 0.6614 | F1_bin: 0.8473 F1_9: 0.6878 F1_avg: 0.7676
Epoch 00060: reducing learning rate of group 0 to 3.9063e-06.
Epoch  60 | Loss: 0.1401 | Acc18: 0.6141 F1_18: 0.6636 | F1_bin: 0.8472 F1_9: 0.6922 F1_avg: 0.7697
Epoch  61 | Loss: 0.1370 | Acc18: 0.6190 F1_18: 0.6656 | F1_bin: 0.8503 F1_9: 0.6927 F1_avg: 0.7715
Epoch  62 | Loss: 0.1404 | Acc18: 0.6141 F1_18: 0.6604 | F1_bin: 0.8477 F1_9: 0.6897 F1_avg: 0.7687
Epoch  63 | Loss: 0.1378 | Acc18: 0.6147 F1_18: 0.6630 | F1_bin: 0.8492 F1_9: 0.6908 F1_avg: 0.7700
Epoch 00064: reducing learning rate of group 0 to 1.9531e-06.
Epoch  64 | Loss: 0.1279 | Acc18: 0.6116 F1_18: 0.6576 | F1_bin: 0.8487 F1_9: 0.6844 F1_avg: 0.7665
Early stopping.
"""
