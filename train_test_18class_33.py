#ModelVariant_LSTMGRU_TinyCNN 特徴量に角速度を追加　allモデル
# CV=0.846
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
SAVE_DIR = "train_test_18class_33"
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

    best_f1, patience_counter = 0, 0
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

        if f1_18 > best_f1:
            best_f1, patience_counter = f1_18, 0
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
Epoch   1 | Loss: 1.7091 | Acc18: 0.5282 F1_18: 0.5517 | F1_bin: 0.8401 F1_9: 0.5880 F1_avg: 0.7141
Epoch   2 | Loss: 1.1921 | Acc18: 0.6023 F1_18: 0.6205 | F1_bin: 0.8638 F1_9: 0.6522 F1_avg: 0.7580
Epoch   3 | Loss: 1.0020 | Acc18: 0.6605 F1_18: 0.6642 | F1_bin: 0.9111 F1_9: 0.6904 F1_avg: 0.8007
Epoch   4 | Loss: 0.8981 | Acc18: 0.6765 F1_18: 0.6921 | F1_bin: 0.9050 F1_9: 0.7073 F1_avg: 0.8062
Epoch   5 | Loss: 0.8122 | Acc18: 0.7010 F1_18: 0.7267 | F1_bin: 0.9111 F1_9: 0.7423 F1_avg: 0.8267
Epoch   6 | Loss: 0.7716 | Acc18: 0.7089 F1_18: 0.7103 | F1_bin: 0.9180 F1_9: 0.7326 F1_avg: 0.8253
Epoch   7 | Loss: 0.7147 | Acc18: 0.7108 F1_18: 0.7361 | F1_bin: 0.9019 F1_9: 0.7613 F1_avg: 0.8316
Epoch   8 | Loss: 0.6800 | Acc18: 0.7114 F1_18: 0.7283 | F1_bin: 0.9312 F1_9: 0.7389 F1_avg: 0.8351
Epoch   9 | Loss: 0.6183 | Acc18: 0.7151 F1_18: 0.7291 | F1_bin: 0.9169 F1_9: 0.7599 F1_avg: 0.8384
Epoch  10 | Loss: 0.6034 | Acc18: 0.7298 F1_18: 0.7520 | F1_bin: 0.9331 F1_9: 0.7636 F1_avg: 0.8483
Epoch  11 | Loss: 0.5875 | Acc18: 0.7402 F1_18: 0.7608 | F1_bin: 0.9302 F1_9: 0.7738 F1_avg: 0.8520
Epoch  12 | Loss: 0.5298 | Acc18: 0.7273 F1_18: 0.7507 | F1_bin: 0.9221 F1_9: 0.7675 F1_avg: 0.8448
Epoch  13 | Loss: 0.5069 | Acc18: 0.7420 F1_18: 0.7645 | F1_bin: 0.9258 F1_9: 0.7807 F1_avg: 0.8533
Epoch  14 | Loss: 0.4813 | Acc18: 0.7414 F1_18: 0.7673 | F1_bin: 0.9215 F1_9: 0.7826 F1_avg: 0.8520
Epoch  15 | Loss: 0.4747 | Acc18: 0.7316 F1_18: 0.7605 | F1_bin: 0.9348 F1_9: 0.7675 F1_avg: 0.8511
Epoch  16 | Loss: 0.4429 | Acc18: 0.7475 F1_18: 0.7872 | F1_bin: 0.9245 F1_9: 0.7926 F1_avg: 0.8585
Epoch  17 | Loss: 0.4506 | Acc18: 0.7500 F1_18: 0.7762 | F1_bin: 0.9294 F1_9: 0.7900 F1_avg: 0.8597
Epoch  18 | Loss: 0.4147 | Acc18: 0.7525 F1_18: 0.7778 | F1_bin: 0.9306 F1_9: 0.7854 F1_avg: 0.8580
Epoch  19 | Loss: 0.3963 | Acc18: 0.7525 F1_18: 0.7712 | F1_bin: 0.9336 F1_9: 0.7756 F1_avg: 0.8546
Epoch 00020: reducing learning rate of group 0 to 5.0000e-04.
Epoch  20 | Loss: 0.3929 | Acc18: 0.7531 F1_18: 0.7862 | F1_bin: 0.9282 F1_9: 0.7946 F1_avg: 0.8614
Epoch  21 | Loss: 0.3190 | Acc18: 0.7837 F1_18: 0.8087 | F1_bin: 0.9440 F1_9: 0.8175 F1_avg: 0.8808
Epoch  22 | Loss: 0.2760 | Acc18: 0.7770 F1_18: 0.7981 | F1_bin: 0.9422 F1_9: 0.8067 F1_avg: 0.8745
Epoch  23 | Loss: 0.2616 | Acc18: 0.7690 F1_18: 0.8011 | F1_bin: 0.9368 F1_9: 0.8050 F1_avg: 0.8709
Epoch  24 | Loss: 0.2430 | Acc18: 0.7806 F1_18: 0.8068 | F1_bin: 0.9427 F1_9: 0.8113 F1_avg: 0.8770
Epoch 00025: reducing learning rate of group 0 to 2.5000e-04.
Epoch  25 | Loss: 0.2452 | Acc18: 0.7770 F1_18: 0.8042 | F1_bin: 0.9410 F1_9: 0.8090 F1_avg: 0.8750
Epoch  26 | Loss: 0.2048 | Acc18: 0.7855 F1_18: 0.8127 | F1_bin: 0.9423 F1_9: 0.8164 F1_avg: 0.8793
Epoch  27 | Loss: 0.1920 | Acc18: 0.7819 F1_18: 0.8046 | F1_bin: 0.9446 F1_9: 0.8101 F1_avg: 0.8773
Epoch  28 | Loss: 0.1810 | Acc18: 0.7794 F1_18: 0.8036 | F1_bin: 0.9459 F1_9: 0.8097 F1_avg: 0.8778
Epoch  29 | Loss: 0.1727 | Acc18: 0.7806 F1_18: 0.8064 | F1_bin: 0.9465 F1_9: 0.8118 F1_avg: 0.8792
Epoch 00030: reducing learning rate of group 0 to 1.2500e-04.
Epoch  30 | Loss: 0.1711 | Acc18: 0.7843 F1_18: 0.8073 | F1_bin: 0.9452 F1_9: 0.8140 F1_avg: 0.8796
Epoch  31 | Loss: 0.1527 | Acc18: 0.7806 F1_18: 0.8076 | F1_bin: 0.9440 F1_9: 0.8126 F1_avg: 0.8783
Epoch  32 | Loss: 0.1448 | Acc18: 0.7819 F1_18: 0.8040 | F1_bin: 0.9435 F1_9: 0.8085 F1_avg: 0.8760
Epoch  33 | Loss: 0.1347 | Acc18: 0.7831 F1_18: 0.8100 | F1_bin: 0.9471 F1_9: 0.8120 F1_avg: 0.8795
Epoch 00034: reducing learning rate of group 0 to 6.2500e-05.
Epoch  34 | Loss: 0.1351 | Acc18: 0.7862 F1_18: 0.8109 | F1_bin: 0.9459 F1_9: 0.8156 F1_avg: 0.8807
Epoch  35 | Loss: 0.1290 | Acc18: 0.7794 F1_18: 0.8049 | F1_bin: 0.9429 F1_9: 0.8104 F1_avg: 0.8767
Epoch  36 | Loss: 0.1246 | Acc18: 0.7812 F1_18: 0.8094 | F1_bin: 0.9416 F1_9: 0.8147 F1_avg: 0.8781
Epoch  37 | Loss: 0.1172 | Acc18: 0.7788 F1_18: 0.8041 | F1_bin: 0.9459 F1_9: 0.8095 F1_avg: 0.8777
Epoch 00038: reducing learning rate of group 0 to 3.1250e-05.
Epoch  38 | Loss: 0.1159 | Acc18: 0.7788 F1_18: 0.8025 | F1_bin: 0.9422 F1_9: 0.8094 F1_avg: 0.8758
Epoch  39 | Loss: 0.1138 | Acc18: 0.7855 F1_18: 0.8114 | F1_bin: 0.9471 F1_9: 0.8157 F1_avg: 0.8814
Epoch  40 | Loss: 0.1109 | Acc18: 0.7812 F1_18: 0.8087 | F1_bin: 0.9435 F1_9: 0.8152 F1_avg: 0.8793
Epoch  41 | Loss: 0.1072 | Acc18: 0.7800 F1_18: 0.8067 | F1_bin: 0.9447 F1_9: 0.8126 F1_avg: 0.8786
Epoch 00042: reducing learning rate of group 0 to 1.5625e-05.
Epoch  42 | Loss: 0.1127 | Acc18: 0.7831 F1_18: 0.8076 | F1_bin: 0.9447 F1_9: 0.8125 F1_avg: 0.8786
Epoch  43 | Loss: 0.1105 | Acc18: 0.7794 F1_18: 0.8056 | F1_bin: 0.9447 F1_9: 0.8104 F1_avg: 0.8775
Epoch  44 | Loss: 0.1084 | Acc18: 0.7825 F1_18: 0.8094 | F1_bin: 0.9465 F1_9: 0.8124 F1_avg: 0.8795
Epoch  45 | Loss: 0.1022 | Acc18: 0.7782 F1_18: 0.8009 | F1_bin: 0.9428 F1_9: 0.8076 F1_avg: 0.8752
Epoch 00046: reducing learning rate of group 0 to 7.8125e-06.
Epoch  46 | Loss: 0.1107 | Acc18: 0.7837 F1_18: 0.8098 | F1_bin: 0.9434 F1_9: 0.8143 F1_avg: 0.8789
Early stopping.

=== Fold 2 ===
Epoch   1 | Loss: 1.6110 | Acc18: 0.5402 F1_18: 0.5659 | F1_bin: 0.8232 F1_9: 0.6033 F1_avg: 0.7133
Epoch   2 | Loss: 1.0792 | Acc18: 0.5549 F1_18: 0.5730 | F1_bin: 0.8406 F1_9: 0.6063 F1_avg: 0.7234
Epoch   3 | Loss: 0.9242 | Acc18: 0.6150 F1_18: 0.6345 | F1_bin: 0.8728 F1_9: 0.6619 F1_avg: 0.7674
Epoch   4 | Loss: 0.8225 | Acc18: 0.5990 F1_18: 0.6253 | F1_bin: 0.8729 F1_9: 0.6511 F1_avg: 0.7620
Epoch   5 | Loss: 0.7445 | Acc18: 0.6186 F1_18: 0.6476 | F1_bin: 0.8559 F1_9: 0.6718 F1_avg: 0.7639
Epoch   6 | Loss: 0.7028 | Acc18: 0.6076 F1_18: 0.6463 | F1_bin: 0.8638 F1_9: 0.6716 F1_avg: 0.7677
Epoch   7 | Loss: 0.6426 | Acc18: 0.6242 F1_18: 0.6497 | F1_bin: 0.8642 F1_9: 0.6786 F1_avg: 0.7714
Epoch   8 | Loss: 0.6158 | Acc18: 0.6389 F1_18: 0.6731 | F1_bin: 0.8705 F1_9: 0.6932 F1_avg: 0.7819
Epoch   9 | Loss: 0.5861 | Acc18: 0.6266 F1_18: 0.6591 | F1_bin: 0.8655 F1_9: 0.6843 F1_avg: 0.7749
Epoch  10 | Loss: 0.5434 | Acc18: 0.6585 F1_18: 0.6944 | F1_bin: 0.8761 F1_9: 0.7147 F1_avg: 0.7954
Epoch  11 | Loss: 0.5093 | Acc18: 0.6548 F1_18: 0.6805 | F1_bin: 0.8669 F1_9: 0.7045 F1_avg: 0.7857
Epoch  12 | Loss: 0.5154 | Acc18: 0.6407 F1_18: 0.6715 | F1_bin: 0.8687 F1_9: 0.6970 F1_avg: 0.7828
Epoch  13 | Loss: 0.4831 | Acc18: 0.6248 F1_18: 0.6648 | F1_bin: 0.8777 F1_9: 0.6843 F1_avg: 0.7810
Epoch 00014: reducing learning rate of group 0 to 5.0000e-04.
Epoch  14 | Loss: 0.4533 | Acc18: 0.6493 F1_18: 0.6924 | F1_bin: 0.8682 F1_9: 0.7057 F1_avg: 0.7869
Epoch  15 | Loss: 0.3653 | Acc18: 0.6842 F1_18: 0.7217 | F1_bin: 0.8816 F1_9: 0.7398 F1_avg: 0.8107
Epoch  16 | Loss: 0.3296 | Acc18: 0.6812 F1_18: 0.7182 | F1_bin: 0.8859 F1_9: 0.7364 F1_avg: 0.8112
Epoch  17 | Loss: 0.3097 | Acc18: 0.6720 F1_18: 0.7116 | F1_bin: 0.8797 F1_9: 0.7261 F1_avg: 0.8029
Epoch  18 | Loss: 0.2905 | Acc18: 0.6800 F1_18: 0.7160 | F1_bin: 0.8761 F1_9: 0.7358 F1_avg: 0.8060
Epoch 00019: reducing learning rate of group 0 to 2.5000e-04.
Epoch  19 | Loss: 0.2841 | Acc18: 0.6683 F1_18: 0.7072 | F1_bin: 0.8725 F1_9: 0.7227 F1_avg: 0.7976
Epoch  20 | Loss: 0.2485 | Acc18: 0.6867 F1_18: 0.7233 | F1_bin: 0.8755 F1_9: 0.7408 F1_avg: 0.8082
Epoch  21 | Loss: 0.2369 | Acc18: 0.6849 F1_18: 0.7220 | F1_bin: 0.8768 F1_9: 0.7385 F1_avg: 0.8076
Epoch  22 | Loss: 0.2113 | Acc18: 0.6720 F1_18: 0.7144 | F1_bin: 0.8822 F1_9: 0.7293 F1_avg: 0.8058
Epoch  23 | Loss: 0.2136 | Acc18: 0.6769 F1_18: 0.7201 | F1_bin: 0.8767 F1_9: 0.7355 F1_avg: 0.8061
Epoch 00024: reducing learning rate of group 0 to 1.2500e-04.
Epoch  24 | Loss: 0.2031 | Acc18: 0.6812 F1_18: 0.7229 | F1_bin: 0.8846 F1_9: 0.7377 F1_avg: 0.8112
Epoch  25 | Loss: 0.1825 | Acc18: 0.6800 F1_18: 0.7218 | F1_bin: 0.8810 F1_9: 0.7342 F1_avg: 0.8076
Epoch  26 | Loss: 0.1706 | Acc18: 0.6720 F1_18: 0.7152 | F1_bin: 0.8829 F1_9: 0.7279 F1_avg: 0.8054
Epoch  27 | Loss: 0.1676 | Acc18: 0.6750 F1_18: 0.7193 | F1_bin: 0.8767 F1_9: 0.7345 F1_avg: 0.8056
Epoch 00028: reducing learning rate of group 0 to 6.2500e-05.
Epoch  28 | Loss: 0.1570 | Acc18: 0.6757 F1_18: 0.7201 | F1_bin: 0.8834 F1_9: 0.7321 F1_avg: 0.8078
Epoch  29 | Loss: 0.1537 | Acc18: 0.6781 F1_18: 0.7202 | F1_bin: 0.8786 F1_9: 0.7347 F1_avg: 0.8066
Epoch  30 | Loss: 0.1490 | Acc18: 0.6793 F1_18: 0.7217 | F1_bin: 0.8835 F1_9: 0.7349 F1_avg: 0.8092
Epoch  31 | Loss: 0.1482 | Acc18: 0.6842 F1_18: 0.7258 | F1_bin: 0.8872 F1_9: 0.7377 F1_avg: 0.8124
Epoch  32 | Loss: 0.1487 | Acc18: 0.6867 F1_18: 0.7249 | F1_bin: 0.8872 F1_9: 0.7375 F1_avg: 0.8123
Epoch  33 | Loss: 0.1400 | Acc18: 0.6800 F1_18: 0.7206 | F1_bin: 0.8823 F1_9: 0.7348 F1_avg: 0.8085
Epoch  34 | Loss: 0.1291 | Acc18: 0.6812 F1_18: 0.7201 | F1_bin: 0.8866 F1_9: 0.7330 F1_avg: 0.8098
Epoch 00035: reducing learning rate of group 0 to 3.1250e-05.
Epoch  35 | Loss: 0.1320 | Acc18: 0.6781 F1_18: 0.7181 | F1_bin: 0.8853 F1_9: 0.7323 F1_avg: 0.8088
Epoch  36 | Loss: 0.1257 | Acc18: 0.6781 F1_18: 0.7196 | F1_bin: 0.8877 F1_9: 0.7330 F1_avg: 0.8103
Epoch  37 | Loss: 0.1287 | Acc18: 0.6830 F1_18: 0.7214 | F1_bin: 0.8871 F1_9: 0.7366 F1_avg: 0.8119
Epoch  38 | Loss: 0.1215 | Acc18: 0.6800 F1_18: 0.7187 | F1_bin: 0.8878 F1_9: 0.7313 F1_avg: 0.8095
Epoch 00039: reducing learning rate of group 0 to 1.5625e-05.
Epoch  39 | Loss: 0.1194 | Acc18: 0.6830 F1_18: 0.7226 | F1_bin: 0.8889 F1_9: 0.7353 F1_avg: 0.8121
Epoch  40 | Loss: 0.1211 | Acc18: 0.6775 F1_18: 0.7156 | F1_bin: 0.8859 F1_9: 0.7306 F1_avg: 0.8082
Epoch  41 | Loss: 0.1254 | Acc18: 0.6806 F1_18: 0.7212 | F1_bin: 0.8877 F1_9: 0.7337 F1_avg: 0.8107
Epoch  42 | Loss: 0.1185 | Acc18: 0.6818 F1_18: 0.7202 | F1_bin: 0.8890 F1_9: 0.7323 F1_avg: 0.8107
Epoch 00043: reducing learning rate of group 0 to 7.8125e-06.
Epoch  43 | Loss: 0.1180 | Acc18: 0.6806 F1_18: 0.7237 | F1_bin: 0.8896 F1_9: 0.7361 F1_avg: 0.8129
Epoch  44 | Loss: 0.1123 | Acc18: 0.6800 F1_18: 0.7212 | F1_bin: 0.8895 F1_9: 0.7345 F1_avg: 0.8120
Epoch  45 | Loss: 0.1183 | Acc18: 0.6818 F1_18: 0.7220 | F1_bin: 0.8914 F1_9: 0.7350 F1_avg: 0.8132
Epoch  46 | Loss: 0.1157 | Acc18: 0.6830 F1_18: 0.7231 | F1_bin: 0.8921 F1_9: 0.7357 F1_avg: 0.8139
Epoch 00047: reducing learning rate of group 0 to 3.9063e-06.
Epoch  47 | Loss: 0.1136 | Acc18: 0.6818 F1_18: 0.7218 | F1_bin: 0.8878 F1_9: 0.7340 F1_avg: 0.8109
Epoch  48 | Loss: 0.1156 | Acc18: 0.6812 F1_18: 0.7210 | F1_bin: 0.8884 F1_9: 0.7328 F1_avg: 0.8106
Epoch  49 | Loss: 0.1147 | Acc18: 0.6793 F1_18: 0.7196 | F1_bin: 0.8908 F1_9: 0.7317 F1_avg: 0.8113
Epoch  50 | Loss: 0.1175 | Acc18: 0.6793 F1_18: 0.7221 | F1_bin: 0.8890 F1_9: 0.7339 F1_avg: 0.8114
Epoch 00051: reducing learning rate of group 0 to 1.9531e-06.
Epoch  51 | Loss: 0.1082 | Acc18: 0.6781 F1_18: 0.7209 | F1_bin: 0.8901 F1_9: 0.7328 F1_avg: 0.8114
Early stopping.

=== Fold 3 ===
Epoch   1 | Loss: 1.7977 | Acc18: 0.4794 F1_18: 0.5144 | F1_bin: 0.8199 F1_9: 0.5450 F1_avg: 0.6824
Epoch   2 | Loss: 1.2653 | Acc18: 0.5691 F1_18: 0.5902 | F1_bin: 0.8589 F1_9: 0.6124 F1_avg: 0.7357
Epoch   3 | Loss: 1.0584 | Acc18: 0.6478 F1_18: 0.6603 | F1_bin: 0.8793 F1_9: 0.6829 F1_avg: 0.7811
Epoch   4 | Loss: 0.9621 | Acc18: 0.6742 F1_18: 0.6804 | F1_bin: 0.8873 F1_9: 0.7031 F1_avg: 0.7952
Epoch   5 | Loss: 0.8764 | Acc18: 0.6349 F1_18: 0.6576 | F1_bin: 0.8734 F1_9: 0.6712 F1_avg: 0.7723
Epoch   6 | Loss: 0.8125 | Acc18: 0.6939 F1_18: 0.7121 | F1_bin: 0.8913 F1_9: 0.7338 F1_avg: 0.8126
Epoch   7 | Loss: 0.7606 | Acc18: 0.6546 F1_18: 0.6670 | F1_bin: 0.8961 F1_9: 0.6972 F1_avg: 0.7967
Epoch   8 | Loss: 0.7049 | Acc18: 0.6699 F1_18: 0.6744 | F1_bin: 0.8939 F1_9: 0.7000 F1_avg: 0.7970
Epoch   9 | Loss: 0.6727 | Acc18: 0.6988 F1_18: 0.7074 | F1_bin: 0.9054 F1_9: 0.7253 F1_avg: 0.8154
Epoch  10 | Loss: 0.6412 | Acc18: 0.7105 F1_18: 0.7177 | F1_bin: 0.8980 F1_9: 0.7427 F1_avg: 0.8203
Epoch  11 | Loss: 0.5941 | Acc18: 0.6982 F1_18: 0.6985 | F1_bin: 0.8972 F1_9: 0.7357 F1_avg: 0.8165
Epoch  12 | Loss: 0.5731 | Acc18: 0.7074 F1_18: 0.7111 | F1_bin: 0.9045 F1_9: 0.7271 F1_avg: 0.8158
Epoch  13 | Loss: 0.5628 | Acc18: 0.7105 F1_18: 0.7041 | F1_bin: 0.9071 F1_9: 0.7358 F1_avg: 0.8214
Epoch 00014: reducing learning rate of group 0 to 5.0000e-04.
Epoch  14 | Loss: 0.5352 | Acc18: 0.7087 F1_18: 0.7024 | F1_bin: 0.8974 F1_9: 0.7356 F1_avg: 0.8165
Epoch  15 | Loss: 0.4377 | Acc18: 0.7283 F1_18: 0.7208 | F1_bin: 0.9084 F1_9: 0.7481 F1_avg: 0.8283
Epoch  16 | Loss: 0.3867 | Acc18: 0.7326 F1_18: 0.7415 | F1_bin: 0.9064 F1_9: 0.7638 F1_avg: 0.8351
Epoch  17 | Loss: 0.3705 | Acc18: 0.7449 F1_18: 0.7479 | F1_bin: 0.9118 F1_9: 0.7745 F1_avg: 0.8431
Epoch  18 | Loss: 0.3547 | Acc18: 0.7265 F1_18: 0.7334 | F1_bin: 0.9132 F1_9: 0.7548 F1_avg: 0.8340
Epoch  19 | Loss: 0.3391 | Acc18: 0.7382 F1_18: 0.7462 | F1_bin: 0.9141 F1_9: 0.7672 F1_avg: 0.8407
Epoch  20 | Loss: 0.3260 | Acc18: 0.7216 F1_18: 0.7172 | F1_bin: 0.9102 F1_9: 0.7464 F1_avg: 0.8283
Epoch  21 | Loss: 0.3230 | Acc18: 0.7474 F1_18: 0.7510 | F1_bin: 0.9154 F1_9: 0.7740 F1_avg: 0.8447
Epoch  22 | Loss: 0.2881 | Acc18: 0.7474 F1_18: 0.7503 | F1_bin: 0.9170 F1_9: 0.7749 F1_avg: 0.8459
Epoch  23 | Loss: 0.2804 | Acc18: 0.7394 F1_18: 0.7459 | F1_bin: 0.9146 F1_9: 0.7674 F1_avg: 0.8410
Epoch  24 | Loss: 0.2645 | Acc18: 0.7443 F1_18: 0.7557 | F1_bin: 0.9183 F1_9: 0.7723 F1_avg: 0.8453
Epoch  25 | Loss: 0.2498 | Acc18: 0.7382 F1_18: 0.7380 | F1_bin: 0.9172 F1_9: 0.7653 F1_avg: 0.8412
Epoch  26 | Loss: 0.2504 | Acc18: 0.7394 F1_18: 0.7522 | F1_bin: 0.9178 F1_9: 0.7722 F1_avg: 0.8450
Epoch  27 | Loss: 0.2410 | Acc18: 0.7388 F1_18: 0.7446 | F1_bin: 0.9118 F1_9: 0.7723 F1_avg: 0.8420
Epoch 00028: reducing learning rate of group 0 to 2.5000e-04.
Epoch  28 | Loss: 0.2419 | Acc18: 0.7259 F1_18: 0.7413 | F1_bin: 0.9087 F1_9: 0.7637 F1_avg: 0.8362
Epoch  29 | Loss: 0.1981 | Acc18: 0.7523 F1_18: 0.7587 | F1_bin: 0.9153 F1_9: 0.7815 F1_avg: 0.8484
Epoch  30 | Loss: 0.1682 | Acc18: 0.7529 F1_18: 0.7543 | F1_bin: 0.9152 F1_9: 0.7795 F1_avg: 0.8473
Epoch  31 | Loss: 0.1614 | Acc18: 0.7474 F1_18: 0.7541 | F1_bin: 0.9166 F1_9: 0.7801 F1_avg: 0.8483
Epoch  32 | Loss: 0.1535 | Acc18: 0.7376 F1_18: 0.7453 | F1_bin: 0.9152 F1_9: 0.7651 F1_avg: 0.8401
Epoch 00033: reducing learning rate of group 0 to 1.2500e-04.
Epoch  33 | Loss: 0.1396 | Acc18: 0.7462 F1_18: 0.7581 | F1_bin: 0.9146 F1_9: 0.7760 F1_avg: 0.8453
Epoch  34 | Loss: 0.1255 | Acc18: 0.7505 F1_18: 0.7575 | F1_bin: 0.9134 F1_9: 0.7804 F1_avg: 0.8469
Epoch  35 | Loss: 0.1186 | Acc18: 0.7560 F1_18: 0.7629 | F1_bin: 0.9172 F1_9: 0.7851 F1_avg: 0.8512
Epoch  36 | Loss: 0.1094 | Acc18: 0.7498 F1_18: 0.7553 | F1_bin: 0.9153 F1_9: 0.7759 F1_avg: 0.8456
Epoch  37 | Loss: 0.1003 | Acc18: 0.7480 F1_18: 0.7552 | F1_bin: 0.9127 F1_9: 0.7780 F1_avg: 0.8454
Epoch  38 | Loss: 0.0971 | Acc18: 0.7621 F1_18: 0.7721 | F1_bin: 0.9203 F1_9: 0.7928 F1_avg: 0.8565
Epoch  39 | Loss: 0.0999 | Acc18: 0.7548 F1_18: 0.7538 | F1_bin: 0.9209 F1_9: 0.7797 F1_avg: 0.8503
Epoch  40 | Loss: 0.0991 | Acc18: 0.7529 F1_18: 0.7584 | F1_bin: 0.9215 F1_9: 0.7792 F1_avg: 0.8503
Epoch  41 | Loss: 0.0932 | Acc18: 0.7541 F1_18: 0.7561 | F1_bin: 0.9159 F1_9: 0.7784 F1_avg: 0.8472
Epoch 00042: reducing learning rate of group 0 to 6.2500e-05.
Epoch  42 | Loss: 0.0949 | Acc18: 0.7511 F1_18: 0.7555 | F1_bin: 0.9195 F1_9: 0.7759 F1_avg: 0.8477
Epoch  43 | Loss: 0.0848 | Acc18: 0.7449 F1_18: 0.7444 | F1_bin: 0.9147 F1_9: 0.7686 F1_avg: 0.8416
Epoch  44 | Loss: 0.0755 | Acc18: 0.7486 F1_18: 0.7527 | F1_bin: 0.9159 F1_9: 0.7732 F1_avg: 0.8446
Epoch  45 | Loss: 0.0747 | Acc18: 0.7535 F1_18: 0.7585 | F1_bin: 0.9153 F1_9: 0.7778 F1_avg: 0.8466
Epoch 00046: reducing learning rate of group 0 to 3.1250e-05.
Epoch  46 | Loss: 0.0778 | Acc18: 0.7554 F1_18: 0.7624 | F1_bin: 0.9184 F1_9: 0.7840 F1_avg: 0.8512
Epoch  47 | Loss: 0.0739 | Acc18: 0.7529 F1_18: 0.7515 | F1_bin: 0.9127 F1_9: 0.7795 F1_avg: 0.8461
Epoch  48 | Loss: 0.0709 | Acc18: 0.7578 F1_18: 0.7616 | F1_bin: 0.9183 F1_9: 0.7860 F1_avg: 0.8521
Epoch  49 | Loss: 0.0683 | Acc18: 0.7560 F1_18: 0.7630 | F1_bin: 0.9178 F1_9: 0.7839 F1_avg: 0.8508
Epoch 00050: reducing learning rate of group 0 to 1.5625e-05.
Epoch  50 | Loss: 0.0645 | Acc18: 0.7578 F1_18: 0.7618 | F1_bin: 0.9158 F1_9: 0.7848 F1_avg: 0.8503
Epoch  51 | Loss: 0.0625 | Acc18: 0.7585 F1_18: 0.7601 | F1_bin: 0.9185 F1_9: 0.7820 F1_avg: 0.8503
Epoch  52 | Loss: 0.0598 | Acc18: 0.7548 F1_18: 0.7607 | F1_bin: 0.9172 F1_9: 0.7835 F1_avg: 0.8504
Epoch  53 | Loss: 0.0685 | Acc18: 0.7554 F1_18: 0.7577 | F1_bin: 0.9191 F1_9: 0.7826 F1_avg: 0.8509
Epoch 00054: reducing learning rate of group 0 to 7.8125e-06.
Epoch  54 | Loss: 0.0618 | Acc18: 0.7535 F1_18: 0.7548 | F1_bin: 0.9166 F1_9: 0.7795 F1_avg: 0.8481
Epoch  55 | Loss: 0.0633 | Acc18: 0.7566 F1_18: 0.7565 | F1_bin: 0.9158 F1_9: 0.7809 F1_avg: 0.8484
Epoch  56 | Loss: 0.0626 | Acc18: 0.7554 F1_18: 0.7629 | F1_bin: 0.9159 F1_9: 0.7832 F1_avg: 0.8495
Epoch  57 | Loss: 0.0609 | Acc18: 0.7529 F1_18: 0.7604 | F1_bin: 0.9135 F1_9: 0.7836 F1_avg: 0.8485
Epoch 00058: reducing learning rate of group 0 to 3.9063e-06.
Epoch  58 | Loss: 0.0645 | Acc18: 0.7523 F1_18: 0.7575 | F1_bin: 0.9153 F1_9: 0.7790 F1_avg: 0.8471
Early stopping.

=== Fold 4 ===
Epoch   1 | Loss: 1.7395 | Acc18: 0.5369 F1_18: 0.5410 | F1_bin: 0.8461 F1_9: 0.5558 F1_avg: 0.7010
Epoch   2 | Loss: 1.1228 | Acc18: 0.5640 F1_18: 0.5935 | F1_bin: 0.8493 F1_9: 0.6103 F1_avg: 0.7298
Epoch   3 | Loss: 0.9619 | Acc18: 0.6408 F1_18: 0.6623 | F1_bin: 0.8713 F1_9: 0.6833 F1_avg: 0.7773
Epoch   4 | Loss: 0.8735 | Acc18: 0.6593 F1_18: 0.6597 | F1_bin: 0.8978 F1_9: 0.6767 F1_avg: 0.7872
Epoch   5 | Loss: 0.7911 | Acc18: 0.6562 F1_18: 0.6649 | F1_bin: 0.8942 F1_9: 0.6718 F1_avg: 0.7830
Epoch   6 | Loss: 0.7363 | Acc18: 0.6538 F1_18: 0.6775 | F1_bin: 0.8908 F1_9: 0.6871 F1_avg: 0.7889
Epoch   7 | Loss: 0.6994 | Acc18: 0.6765 F1_18: 0.6892 | F1_bin: 0.9101 F1_9: 0.6988 F1_avg: 0.8045
Epoch   8 | Loss: 0.6442 | Acc18: 0.6710 F1_18: 0.6845 | F1_bin: 0.8942 F1_9: 0.7162 F1_avg: 0.8052
Epoch   9 | Loss: 0.6004 | Acc18: 0.6925 F1_18: 0.7128 | F1_bin: 0.8936 F1_9: 0.7206 F1_avg: 0.8071
Epoch  10 | Loss: 0.5781 | Acc18: 0.6956 F1_18: 0.7294 | F1_bin: 0.9026 F1_9: 0.7423 F1_avg: 0.8225
Epoch  11 | Loss: 0.5435 | Acc18: 0.7060 F1_18: 0.7313 | F1_bin: 0.9021 F1_9: 0.7484 F1_avg: 0.8252
Epoch  12 | Loss: 0.5306 | Acc18: 0.7097 F1_18: 0.7452 | F1_bin: 0.9169 F1_9: 0.7479 F1_avg: 0.8324
Epoch  13 | Loss: 0.4907 | Acc18: 0.7005 F1_18: 0.7338 | F1_bin: 0.9102 F1_9: 0.7463 F1_avg: 0.8282
Epoch  14 | Loss: 0.4727 | Acc18: 0.6999 F1_18: 0.7343 | F1_bin: 0.9040 F1_9: 0.7465 F1_avg: 0.8253
Epoch  15 | Loss: 0.4567 | Acc18: 0.7140 F1_18: 0.7436 | F1_bin: 0.9076 F1_9: 0.7554 F1_avg: 0.8315
Epoch 00016: reducing learning rate of group 0 to 5.0000e-04.
Epoch  16 | Loss: 0.4337 | Acc18: 0.7011 F1_18: 0.7226 | F1_bin: 0.9089 F1_9: 0.7409 F1_avg: 0.8249
Epoch  17 | Loss: 0.3552 | Acc18: 0.7232 F1_18: 0.7527 | F1_bin: 0.9132 F1_9: 0.7641 F1_avg: 0.8386
Epoch  18 | Loss: 0.3085 | Acc18: 0.7140 F1_18: 0.7462 | F1_bin: 0.9059 F1_9: 0.7587 F1_avg: 0.8323
Epoch  19 | Loss: 0.3035 | Acc18: 0.7171 F1_18: 0.7448 | F1_bin: 0.9139 F1_9: 0.7596 F1_avg: 0.8368
Epoch  20 | Loss: 0.2952 | Acc18: 0.7263 F1_18: 0.7629 | F1_bin: 0.9222 F1_9: 0.7681 F1_avg: 0.8451
Epoch  21 | Loss: 0.2671 | Acc18: 0.7269 F1_18: 0.7532 | F1_bin: 0.9181 F1_9: 0.7660 F1_avg: 0.8421
Epoch  22 | Loss: 0.2671 | Acc18: 0.7306 F1_18: 0.7640 | F1_bin: 0.9139 F1_9: 0.7711 F1_avg: 0.8425
Epoch  23 | Loss: 0.2437 | Acc18: 0.7220 F1_18: 0.7517 | F1_bin: 0.9114 F1_9: 0.7653 F1_avg: 0.8384
Epoch  24 | Loss: 0.2358 | Acc18: 0.7239 F1_18: 0.7509 | F1_bin: 0.9194 F1_9: 0.7618 F1_avg: 0.8406
Epoch  25 | Loss: 0.2257 | Acc18: 0.7288 F1_18: 0.7625 | F1_bin: 0.9157 F1_9: 0.7749 F1_avg: 0.8453
Epoch  26 | Loss: 0.2205 | Acc18: 0.7386 F1_18: 0.7719 | F1_bin: 0.9261 F1_9: 0.7777 F1_avg: 0.8519
Epoch  27 | Loss: 0.2039 | Acc18: 0.7276 F1_18: 0.7575 | F1_bin: 0.9187 F1_9: 0.7672 F1_avg: 0.8429
Epoch  28 | Loss: 0.2060 | Acc18: 0.7060 F1_18: 0.7503 | F1_bin: 0.9145 F1_9: 0.7591 F1_avg: 0.8368
Epoch  29 | Loss: 0.2002 | Acc18: 0.7073 F1_18: 0.7461 | F1_bin: 0.9120 F1_9: 0.7637 F1_avg: 0.8379
Epoch 00030: reducing learning rate of group 0 to 2.5000e-04.
Epoch  30 | Loss: 0.1907 | Acc18: 0.7153 F1_18: 0.7531 | F1_bin: 0.9181 F1_9: 0.7622 F1_avg: 0.8401
Epoch  31 | Loss: 0.1470 | Acc18: 0.7343 F1_18: 0.7632 | F1_bin: 0.9168 F1_9: 0.7752 F1_avg: 0.8460
Epoch  32 | Loss: 0.1292 | Acc18: 0.7232 F1_18: 0.7580 | F1_bin: 0.9181 F1_9: 0.7702 F1_avg: 0.8442
Epoch  33 | Loss: 0.1280 | Acc18: 0.7368 F1_18: 0.7724 | F1_bin: 0.9188 F1_9: 0.7834 F1_avg: 0.8511
Epoch  34 | Loss: 0.1211 | Acc18: 0.7269 F1_18: 0.7631 | F1_bin: 0.9175 F1_9: 0.7744 F1_avg: 0.8460
Epoch  35 | Loss: 0.1107 | Acc18: 0.7337 F1_18: 0.7701 | F1_bin: 0.9237 F1_9: 0.7785 F1_avg: 0.8511
Epoch  36 | Loss: 0.1124 | Acc18: 0.7245 F1_18: 0.7638 | F1_bin: 0.9206 F1_9: 0.7705 F1_avg: 0.8455
Epoch 00037: reducing learning rate of group 0 to 1.2500e-04.
Epoch  37 | Loss: 0.1089 | Acc18: 0.7325 F1_18: 0.7656 | F1_bin: 0.9267 F1_9: 0.7725 F1_avg: 0.8496
Epoch  38 | Loss: 0.0895 | Acc18: 0.7435 F1_18: 0.7736 | F1_bin: 0.9280 F1_9: 0.7822 F1_avg: 0.8551
Epoch  39 | Loss: 0.0882 | Acc18: 0.7435 F1_18: 0.7827 | F1_bin: 0.9255 F1_9: 0.7896 F1_avg: 0.8575
Epoch  40 | Loss: 0.0794 | Acc18: 0.7405 F1_18: 0.7744 | F1_bin: 0.9286 F1_9: 0.7829 F1_avg: 0.8557
Epoch  41 | Loss: 0.0771 | Acc18: 0.7368 F1_18: 0.7713 | F1_bin: 0.9279 F1_9: 0.7775 F1_avg: 0.8527
Epoch  42 | Loss: 0.0779 | Acc18: 0.7331 F1_18: 0.7651 | F1_bin: 0.9199 F1_9: 0.7739 F1_avg: 0.8469
Epoch 00043: reducing learning rate of group 0 to 6.2500e-05.
Epoch  43 | Loss: 0.0746 | Acc18: 0.7349 F1_18: 0.7679 | F1_bin: 0.9224 F1_9: 0.7726 F1_avg: 0.8475
Epoch  44 | Loss: 0.0693 | Acc18: 0.7392 F1_18: 0.7761 | F1_bin: 0.9267 F1_9: 0.7845 F1_avg: 0.8556
Epoch  45 | Loss: 0.0678 | Acc18: 0.7423 F1_18: 0.7770 | F1_bin: 0.9273 F1_9: 0.7836 F1_avg: 0.8555
Epoch  46 | Loss: 0.0641 | Acc18: 0.7355 F1_18: 0.7707 | F1_bin: 0.9242 F1_9: 0.7794 F1_avg: 0.8518
Epoch 00047: reducing learning rate of group 0 to 3.1250e-05.
Epoch  47 | Loss: 0.0652 | Acc18: 0.7417 F1_18: 0.7744 | F1_bin: 0.9255 F1_9: 0.7809 F1_avg: 0.8532
Epoch  48 | Loss: 0.0643 | Acc18: 0.7368 F1_18: 0.7727 | F1_bin: 0.9267 F1_9: 0.7771 F1_avg: 0.8519
Epoch  49 | Loss: 0.0594 | Acc18: 0.7392 F1_18: 0.7751 | F1_bin: 0.9249 F1_9: 0.7841 F1_avg: 0.8545
Epoch  50 | Loss: 0.0619 | Acc18: 0.7392 F1_18: 0.7732 | F1_bin: 0.9267 F1_9: 0.7799 F1_avg: 0.8533
Epoch 00051: reducing learning rate of group 0 to 1.5625e-05.
Epoch  51 | Loss: 0.0585 | Acc18: 0.7392 F1_18: 0.7751 | F1_bin: 0.9273 F1_9: 0.7827 F1_avg: 0.8550
Epoch  52 | Loss: 0.0558 | Acc18: 0.7362 F1_18: 0.7731 | F1_bin: 0.9224 F1_9: 0.7826 F1_avg: 0.8525
Epoch  53 | Loss: 0.0522 | Acc18: 0.7399 F1_18: 0.7718 | F1_bin: 0.9249 F1_9: 0.7822 F1_avg: 0.8535
Epoch  54 | Loss: 0.0553 | Acc18: 0.7380 F1_18: 0.7739 | F1_bin: 0.9273 F1_9: 0.7803 F1_avg: 0.8538
Epoch 00055: reducing learning rate of group 0 to 7.8125e-06.
Epoch  55 | Loss: 0.0541 | Acc18: 0.7306 F1_18: 0.7656 | F1_bin: 0.9224 F1_9: 0.7739 F1_avg: 0.8482
Epoch  56 | Loss: 0.0575 | Acc18: 0.7399 F1_18: 0.7733 | F1_bin: 0.9255 F1_9: 0.7800 F1_avg: 0.8528
Epoch  57 | Loss: 0.0526 | Acc18: 0.7368 F1_18: 0.7701 | F1_bin: 0.9237 F1_9: 0.7818 F1_avg: 0.8527
Epoch  58 | Loss: 0.0583 | Acc18: 0.7368 F1_18: 0.7725 | F1_bin: 0.9249 F1_9: 0.7819 F1_avg: 0.8534
Epoch 00059: reducing learning rate of group 0 to 3.9063e-06.
Epoch  59 | Loss: 0.0543 | Acc18: 0.7380 F1_18: 0.7715 | F1_bin: 0.9273 F1_9: 0.7785 F1_avg: 0.8529
Early stopping.

=== Fold 5 ===
Epoch   1 | Loss: 1.6754 | Acc18: 0.5168 F1_18: 0.5353 | F1_bin: 0.8351 F1_9: 0.5603 F1_avg: 0.6977
Epoch   2 | Loss: 1.1155 | Acc18: 0.5615 F1_18: 0.5689 | F1_bin: 0.8694 F1_9: 0.5881 F1_avg: 0.7287
Epoch   3 | Loss: 0.9647 | Acc18: 0.6080 F1_18: 0.6282 | F1_bin: 0.8752 F1_9: 0.6426 F1_avg: 0.7589
Epoch   4 | Loss: 0.8738 | Acc18: 0.6135 F1_18: 0.6242 | F1_bin: 0.9007 F1_9: 0.6432 F1_avg: 0.7720
Epoch   5 | Loss: 0.7920 | Acc18: 0.6642 F1_18: 0.6805 | F1_bin: 0.8983 F1_9: 0.6991 F1_avg: 0.7987
Epoch   6 | Loss: 0.7338 | Acc18: 0.6465 F1_18: 0.6662 | F1_bin: 0.8946 F1_9: 0.6909 F1_avg: 0.7928
Epoch   7 | Loss: 0.6736 | Acc18: 0.6593 F1_18: 0.6888 | F1_bin: 0.9109 F1_9: 0.6984 F1_avg: 0.8046
Epoch   8 | Loss: 0.6298 | Acc18: 0.6758 F1_18: 0.6988 | F1_bin: 0.8960 F1_9: 0.7156 F1_avg: 0.8058
Epoch   9 | Loss: 0.5996 | Acc18: 0.6881 F1_18: 0.7084 | F1_bin: 0.8936 F1_9: 0.7321 F1_avg: 0.8128
Epoch  10 | Loss: 0.5608 | Acc18: 0.6801 F1_18: 0.7079 | F1_bin: 0.8965 F1_9: 0.7242 F1_avg: 0.8103
Epoch  11 | Loss: 0.5410 | Acc18: 0.6654 F1_18: 0.6969 | F1_bin: 0.8945 F1_9: 0.7098 F1_avg: 0.8022
Epoch  12 | Loss: 0.5014 | Acc18: 0.6936 F1_18: 0.7124 | F1_bin: 0.9033 F1_9: 0.7253 F1_avg: 0.8143
Epoch  13 | Loss: 0.4908 | Acc18: 0.6930 F1_18: 0.7215 | F1_bin: 0.9103 F1_9: 0.7334 F1_avg: 0.8219
Epoch  14 | Loss: 0.4801 | Acc18: 0.6899 F1_18: 0.7263 | F1_bin: 0.9026 F1_9: 0.7340 F1_avg: 0.8183
Epoch  15 | Loss: 0.4451 | Acc18: 0.6765 F1_18: 0.7150 | F1_bin: 0.9032 F1_9: 0.7186 F1_avg: 0.8109
Epoch  16 | Loss: 0.4284 | Acc18: 0.6936 F1_18: 0.7170 | F1_bin: 0.9076 F1_9: 0.7312 F1_avg: 0.8194
Epoch  17 | Loss: 0.3898 | Acc18: 0.6844 F1_18: 0.7039 | F1_bin: 0.9112 F1_9: 0.7218 F1_avg: 0.8165
Epoch 00018: reducing learning rate of group 0 to 5.0000e-04.
Epoch  18 | Loss: 0.3831 | Acc18: 0.6832 F1_18: 0.7177 | F1_bin: 0.8911 F1_9: 0.7347 F1_avg: 0.8129
Epoch  19 | Loss: 0.3165 | Acc18: 0.7058 F1_18: 0.7382 | F1_bin: 0.9094 F1_9: 0.7517 F1_avg: 0.8306
Epoch  20 | Loss: 0.2806 | Acc18: 0.7254 F1_18: 0.7548 | F1_bin: 0.9190 F1_9: 0.7689 F1_avg: 0.8439
Epoch  21 | Loss: 0.2612 | Acc18: 0.7101 F1_18: 0.7381 | F1_bin: 0.9147 F1_9: 0.7548 F1_avg: 0.8348
Epoch  22 | Loss: 0.2411 | Acc18: 0.7107 F1_18: 0.7436 | F1_bin: 0.9118 F1_9: 0.7568 F1_avg: 0.8343
Epoch  23 | Loss: 0.2364 | Acc18: 0.7095 F1_18: 0.7437 | F1_bin: 0.9112 F1_9: 0.7542 F1_avg: 0.8327
Epoch 00024: reducing learning rate of group 0 to 2.5000e-04.
Epoch  24 | Loss: 0.2272 | Acc18: 0.7003 F1_18: 0.7263 | F1_bin: 0.9042 F1_9: 0.7460 F1_avg: 0.8251
Epoch  25 | Loss: 0.1996 | Acc18: 0.7034 F1_18: 0.7324 | F1_bin: 0.9130 F1_9: 0.7441 F1_avg: 0.8285
Epoch  26 | Loss: 0.1726 | Acc18: 0.7138 F1_18: 0.7435 | F1_bin: 0.9111 F1_9: 0.7584 F1_avg: 0.8348
Epoch  27 | Loss: 0.1641 | Acc18: 0.7260 F1_18: 0.7488 | F1_bin: 0.9148 F1_9: 0.7646 F1_avg: 0.8397
Epoch 00028: reducing learning rate of group 0 to 1.2500e-04.
Epoch  28 | Loss: 0.1575 | Acc18: 0.7119 F1_18: 0.7507 | F1_bin: 0.9160 F1_9: 0.7611 F1_avg: 0.8386
Epoch  29 | Loss: 0.1463 | Acc18: 0.7095 F1_18: 0.7390 | F1_bin: 0.9100 F1_9: 0.7561 F1_avg: 0.8330
Epoch  30 | Loss: 0.1403 | Acc18: 0.7107 F1_18: 0.7436 | F1_bin: 0.9130 F1_9: 0.7571 F1_avg: 0.8350
Epoch  31 | Loss: 0.1290 | Acc18: 0.7113 F1_18: 0.7431 | F1_bin: 0.9099 F1_9: 0.7566 F1_avg: 0.8333
Epoch 00032: reducing learning rate of group 0 to 6.2500e-05.
Epoch  32 | Loss: 0.1318 | Acc18: 0.7156 F1_18: 0.7426 | F1_bin: 0.9136 F1_9: 0.7582 F1_avg: 0.8359
Epoch  33 | Loss: 0.1208 | Acc18: 0.7058 F1_18: 0.7374 | F1_bin: 0.9136 F1_9: 0.7522 F1_avg: 0.8329
Epoch  34 | Loss: 0.1126 | Acc18: 0.7119 F1_18: 0.7407 | F1_bin: 0.9118 F1_9: 0.7589 F1_avg: 0.8353
Epoch  35 | Loss: 0.1094 | Acc18: 0.7119 F1_18: 0.7391 | F1_bin: 0.9166 F1_9: 0.7558 F1_avg: 0.8362
Epoch 00036: reducing learning rate of group 0 to 3.1250e-05.
Epoch  36 | Loss: 0.1091 | Acc18: 0.7089 F1_18: 0.7376 | F1_bin: 0.9160 F1_9: 0.7516 F1_avg: 0.8338
Epoch  37 | Loss: 0.1130 | Acc18: 0.7119 F1_18: 0.7408 | F1_bin: 0.9160 F1_9: 0.7558 F1_avg: 0.8359
Epoch  38 | Loss: 0.1041 | Acc18: 0.7058 F1_18: 0.7352 | F1_bin: 0.9142 F1_9: 0.7499 F1_avg: 0.8321
Epoch  39 | Loss: 0.1065 | Acc18: 0.7131 F1_18: 0.7398 | F1_bin: 0.9179 F1_9: 0.7561 F1_avg: 0.8370
Epoch 00040: reducing learning rate of group 0 to 1.5625e-05.
Epoch  40 | Loss: 0.1052 | Acc18: 0.7101 F1_18: 0.7371 | F1_bin: 0.9124 F1_9: 0.7539 F1_avg: 0.8331
Early stopping.
"""
