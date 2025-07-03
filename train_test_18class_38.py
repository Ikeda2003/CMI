#ModelVariant_LSTMGRU 特徴量に角速度を追加。シーケンスをまたがないように修正　allモデル
#ミックスアップを追加(thmも含める)
#CV=0.85ぐらいで少し上昇傾向がある　
#CMI 2025 デモ提出 バージョン76　IMUonly+all(ミックスアップを両モデルに適用) LB=0.83
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
SAVE_DIR = "train_test_18class_38"
PAD_PERCENTILE = 95
BATCH_SIZE = 64
EPOCHS = 100
LR_INIT = 1e-3
WD = 1e-4
PATIENCE = 20

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= データ準備・特徴量関数 =========
# （ここにあなたの feature_eng, remove_gravity_from_acc, calculate_angular_velocity_from_quat
#   calculate_angular_distance をそのまま貼り付け）
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
# ========= 各モジュール（TinyCNN, MultiScaleConv1d, SEBlock, ResidualSEBlock, AttentionLayer, MetaFeatureExtractor, GaussianNoise）
# （ここにあなたのクラスをそのまま貼り付け）
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


# ========= データロードと前処理 =========
df = pd.read_csv(RAW_CSV)
df = feature_eng(df)
df["gesture"] = df["gesture"].fillna("unknown")
le = LabelEncoder()
df["gesture_class"] = le.fit_transform(df["gesture"])

num_cols = df.select_dtypes(include=[np.number]).columns
#linear_acc系の特徴を除外した
imu_cols = [c for c in df.select_dtypes(include=np.number).columns
            if c.startswith(('acc_','rot_','acc_mag','rot_angle','acc_mag_jerk',
                             'rot_angle_vel','angular_vel','angular_dist','thm_')) and c!="gesture_class"]

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
    def __init__(self, imu_seqs, tof_seqs, y, mixup=False, alpha=0.4, num_classes=18):
        self.imu_seqs = torch.tensor(np.array(imu_seqs), dtype=torch.float32)
        self.tof_seqs = torch.tensor(np.array(tof_seqs), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.mixup = mixup
        self.alpha = alpha
        self.num_classes = num_classes

    def __len__(self):
        return len(self.imu_seqs)

    def __getitem__(self, i):
        xi_imu, xi_tof, yi = self.imu_seqs[i], self.tof_seqs[i], self.y[i]
        if self.mixup and np.random.rand() < 0.5:
            j = np.random.randint(0, len(self.imu_seqs))
            lam = np.random.beta(self.alpha, self.alpha)
            xj_imu, yj = self.imu_seqs[j], self.y[j]

            # IMU だけ mixup
            xi_imu = lam * xi_imu + (1 - lam) * xj_imu

            # ラベルも mixup
            yi_onehot = F.one_hot(yi, num_classes=self.num_classes).float()
            yj_onehot = F.one_hot(yj, num_classes=self.num_classes).float()
            yi = lam * yi_onehot + (1 - lam) * yj_onehot

            # TOF はそのまま
        else:
            yi = F.one_hot(yi, num_classes=self.num_classes).float()

        return xi_imu, xi_tof, yi


# ========= 学習コード（修正版） =========
kf = GroupKFold(n_splits=5)
seq_meta = df.drop_duplicates("sequence_id")[["sequence_id", "subject", "gesture_class"]]

for fold, (tr_idx, va_idx) in enumerate(kf.split(seq_meta, groups=seq_meta["subject"])):
    print(f"\n=== Fold {fold+1} ===")
    fold_dir = os.path.join(SAVE_DIR, f"fold{fold+1}")
    os.makedirs(fold_dir, exist_ok=True)

    train_ids = seq_meta.iloc[tr_idx]["sequence_id"].values
    val_ids = seq_meta.iloc[va_idx]["sequence_id"].values
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

    train_loader = DataLoader(
        TwoBranchDataset(X_imu_train, X_tof_train, y_train, mixup=True, alpha=0.4, num_classes=len(le.classes_)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TwoBranchDataset(X_imu_val, X_tof_val, y_val, mixup=False, num_classes=len(le.classes_)),
        batch_size=BATCH_SIZE
    )

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
            # soft label クロスエントロピー
            loss = torch.mean(torch.sum(-yb * F.log_softmax(logits, dim=1), dim=1))
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
                trues.extend(yb.argmax(1).numpy())  # one-hot からクラス番号に

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
