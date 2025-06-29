#LSTM+GRU+attention+noise モデルにTOF画像ブランチを追加
#CV=0.83
#CMI 2025 デモ提出 バージョン64　LB=75
#CMI 2025 デモ提出 バージョン67 IMUonly+ all 　LB=0.79

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

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
# ========= TinyCNN（in_channels=5に修正） =========
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




# ========= 設定 =========
RAW_CSV = "train.csv"
SAVE_DIR = "train_test_18class_29"
PAD_PERCENTILE = 95
BATCH_SIZE = 64
EPOCHS = 100
LR_INIT = 1e-3
WD = 1e-4
PATIENCE = 20

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= データ準備 =========
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
imu_cols = [c for c in num_cols if c.startswith(('acc_', 'rot_', 'acc_mag', 'rot_angle', 'acc_mag_jerk', 'rot_angle_vel', 'thm_'))]
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

    # ========= デバッグ付きモデル =========
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

        print(f"Epoch {epoch+1} | Loss:{total_loss/len(train_loader.dataset):.4f} | "
              f"Acc18:{acc_18:.4f}, F1_18:{f1_18:.4f} | "
              f"F1_bin:{f1_bin:.4f}, F1_9:{f1_9:.4f}, F1_avg:{f1_avg:.4f}")

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
Epoch 1 | Loss:1.7893 | Acc18:0.5527, F1_18:0.5649 | F1_bin:0.8570, F1_9:0.5913, F1_avg:0.7242
Epoch 2 | Loss:1.2323 | Acc18:0.6415, F1_18:0.6545 | F1_bin:0.8936, F1_9:0.6836, F1_avg:0.7886
Epoch 3 | Loss:1.0796 | Acc18:0.6293, F1_18:0.6467 | F1_bin:0.8719, F1_9:0.6765, F1_avg:0.7742
Epoch 4 | Loss:0.9694 | Acc18:0.6599, F1_18:0.6691 | F1_bin:0.8986, F1_9:0.6952, F1_avg:0.7969
Epoch 5 | Loss:0.9013 | Acc18:0.6575, F1_18:0.6789 | F1_bin:0.9077, F1_9:0.7012, F1_avg:0.8045
Epoch 6 | Loss:0.8338 | Acc18:0.6881, F1_18:0.6841 | F1_bin:0.9200, F1_9:0.7068, F1_avg:0.8134
Epoch 7 | Loss:0.7770 | Acc18:0.6795, F1_18:0.6918 | F1_bin:0.9204, F1_9:0.7128, F1_avg:0.8166
Epoch 8 | Loss:0.7373 | Acc18:0.7120, F1_18:0.7323 | F1_bin:0.9197, F1_9:0.7487, F1_avg:0.8342
Epoch 9 | Loss:0.6891 | Acc18:0.7132, F1_18:0.7311 | F1_bin:0.9367, F1_9:0.7386, F1_avg:0.8377
Epoch 10 | Loss:0.6786 | Acc18:0.7181, F1_18:0.7333 | F1_bin:0.9299, F1_9:0.7480, F1_avg:0.8390
Epoch 11 | Loss:0.6245 | Acc18:0.7034, F1_18:0.7094 | F1_bin:0.9256, F1_9:0.7232, F1_avg:0.8244
Epoch 12 | Loss:0.6259 | Acc18:0.7175, F1_18:0.7356 | F1_bin:0.9370, F1_9:0.7492, F1_avg:0.8431
Epoch 13 | Loss:0.5816 | Acc18:0.7138, F1_18:0.7374 | F1_bin:0.9409, F1_9:0.7458, F1_avg:0.8434
Epoch 14 | Loss:0.5670 | Acc18:0.7426, F1_18:0.7557 | F1_bin:0.9472, F1_9:0.7596, F1_avg:0.8534
Epoch 15 | Loss:0.5383 | Acc18:0.6752, F1_18:0.7034 | F1_bin:0.9225, F1_9:0.7195, F1_avg:0.8210
Epoch 16 | Loss:0.5209 | Acc18:0.7353, F1_18:0.7578 | F1_bin:0.9312, F1_9:0.7721, F1_avg:0.8517
Epoch 17 | Loss:0.5040 | Acc18:0.7402, F1_18:0.7587 | F1_bin:0.9347, F1_9:0.7685, F1_avg:0.8516
Epoch 18 | Loss:0.4719 | Acc18:0.7230, F1_18:0.7200 | F1_bin:0.9354, F1_9:0.7384, F1_avg:0.8369
Epoch 19 | Loss:0.4599 | Acc18:0.7475, F1_18:0.7645 | F1_bin:0.9441, F1_9:0.7696, F1_avg:0.8569
Epoch 20 | Loss:0.4357 | Acc18:0.7261, F1_18:0.7565 | F1_bin:0.9349, F1_9:0.7666, F1_avg:0.8507
Epoch 21 | Loss:0.4296 | Acc18:0.7279, F1_18:0.7502 | F1_bin:0.9337, F1_9:0.7577, F1_avg:0.8457
Epoch 22 | Loss:0.4085 | Acc18:0.7298, F1_18:0.7418 | F1_bin:0.9318, F1_9:0.7589, F1_avg:0.8453
Epoch 00023: reducing learning rate of group 0 to 5.0000e-04.
Epoch 23 | Loss:0.3819 | Acc18:0.7279, F1_18:0.7577 | F1_bin:0.9361, F1_9:0.7653, F1_avg:0.8507
Epoch 24 | Loss:0.3236 | Acc18:0.7537, F1_18:0.7659 | F1_bin:0.9429, F1_9:0.7755, F1_avg:0.8592
Epoch 25 | Loss:0.2885 | Acc18:0.7616, F1_18:0.7824 | F1_bin:0.9477, F1_9:0.7921, F1_avg:0.8699
Epoch 26 | Loss:0.2607 | Acc18:0.7580, F1_18:0.7748 | F1_bin:0.9441, F1_9:0.7816, F1_avg:0.8628
Epoch 27 | Loss:0.2516 | Acc18:0.7561, F1_18:0.7773 | F1_bin:0.9397, F1_9:0.7851, F1_avg:0.8624
Epoch 28 | Loss:0.2538 | Acc18:0.7420, F1_18:0.7581 | F1_bin:0.9422, F1_9:0.7674, F1_avg:0.8548
Epoch 00029: reducing learning rate of group 0 to 2.5000e-04.
Epoch 29 | Loss:0.2332 | Acc18:0.7531, F1_18:0.7713 | F1_bin:0.9409, F1_9:0.7825, F1_avg:0.8617
Epoch 30 | Loss:0.2023 | Acc18:0.7678, F1_18:0.7878 | F1_bin:0.9502, F1_9:0.7946, F1_avg:0.8724
Epoch 31 | Loss:0.1825 | Acc18:0.7623, F1_18:0.7781 | F1_bin:0.9489, F1_9:0.7886, F1_avg:0.8688
Epoch 32 | Loss:0.1717 | Acc18:0.7500, F1_18:0.7709 | F1_bin:0.9440, F1_9:0.7825, F1_avg:0.8633
Epoch 33 | Loss:0.1659 | Acc18:0.7604, F1_18:0.7772 | F1_bin:0.9434, F1_9:0.7901, F1_avg:0.8668
Epoch 00034: reducing learning rate of group 0 to 1.2500e-04.
Epoch 34 | Loss:0.1531 | Acc18:0.7635, F1_18:0.7869 | F1_bin:0.9446, F1_9:0.7955, F1_avg:0.8700
Epoch 35 | Loss:0.1443 | Acc18:0.7580, F1_18:0.7767 | F1_bin:0.9447, F1_9:0.7849, F1_avg:0.8648
Epoch 36 | Loss:0.1313 | Acc18:0.7610, F1_18:0.7830 | F1_bin:0.9477, F1_9:0.7922, F1_avg:0.8700
Epoch 37 | Loss:0.1303 | Acc18:0.7641, F1_18:0.7818 | F1_bin:0.9490, F1_9:0.7924, F1_avg:0.8707
Epoch 00038: reducing learning rate of group 0 to 6.2500e-05.
Epoch 38 | Loss:0.1253 | Acc18:0.7629, F1_18:0.7873 | F1_bin:0.9452, F1_9:0.7972, F1_avg:0.8712
Epoch 39 | Loss:0.1203 | Acc18:0.7610, F1_18:0.7842 | F1_bin:0.9453, F1_9:0.7934, F1_avg:0.8693
Epoch 40 | Loss:0.1189 | Acc18:0.7635, F1_18:0.7830 | F1_bin:0.9489, F1_9:0.7921, F1_avg:0.8705
Epoch 41 | Loss:0.1115 | Acc18:0.7641, F1_18:0.7854 | F1_bin:0.9453, F1_9:0.7958, F1_avg:0.8705
Epoch 00042: reducing learning rate of group 0 to 3.1250e-05.
Epoch 42 | Loss:0.1079 | Acc18:0.7653, F1_18:0.7842 | F1_bin:0.9520, F1_9:0.7948, F1_avg:0.8734
Epoch 43 | Loss:0.1095 | Acc18:0.7647, F1_18:0.7867 | F1_bin:0.9465, F1_9:0.7940, F1_avg:0.8703
Epoch 44 | Loss:0.1020 | Acc18:0.7629, F1_18:0.7859 | F1_bin:0.9489, F1_9:0.7937, F1_avg:0.8713
Epoch 45 | Loss:0.1053 | Acc18:0.7696, F1_18:0.7908 | F1_bin:0.9489, F1_9:0.7978, F1_avg:0.8734
Epoch 46 | Loss:0.1035 | Acc18:0.7647, F1_18:0.7862 | F1_bin:0.9477, F1_9:0.7929, F1_avg:0.8703
Epoch 47 | Loss:0.1011 | Acc18:0.7635, F1_18:0.7846 | F1_bin:0.9483, F1_9:0.7925, F1_avg:0.8704
Epoch 48 | Loss:0.0992 | Acc18:0.7598, F1_18:0.7821 | F1_bin:0.9489, F1_9:0.7883, F1_avg:0.8686
Epoch 49 | Loss:0.0966 | Acc18:0.7696, F1_18:0.7928 | F1_bin:0.9508, F1_9:0.7993, F1_avg:0.8751
Epoch 50 | Loss:0.0956 | Acc18:0.7665, F1_18:0.7857 | F1_bin:0.9489, F1_9:0.7938, F1_avg:0.8714
Epoch 51 | Loss:0.0928 | Acc18:0.7665, F1_18:0.7868 | F1_bin:0.9471, F1_9:0.7951, F1_avg:0.8711
Epoch 52 | Loss:0.0975 | Acc18:0.7672, F1_18:0.7853 | F1_bin:0.9502, F1_9:0.7931, F1_avg:0.8717
Epoch 00053: reducing learning rate of group 0 to 1.5625e-05.
Epoch 53 | Loss:0.0925 | Acc18:0.7672, F1_18:0.7885 | F1_bin:0.9502, F1_9:0.7948, F1_avg:0.8725
Epoch 54 | Loss:0.0973 | Acc18:0.7610, F1_18:0.7840 | F1_bin:0.9471, F1_9:0.7910, F1_avg:0.8690
Epoch 55 | Loss:0.0953 | Acc18:0.7598, F1_18:0.7815 | F1_bin:0.9471, F1_9:0.7897, F1_avg:0.8684
Epoch 56 | Loss:0.0882 | Acc18:0.7635, F1_18:0.7831 | F1_bin:0.9489, F1_9:0.7915, F1_avg:0.8702
Epoch 00057: reducing learning rate of group 0 to 7.8125e-06.
Epoch 57 | Loss:0.0901 | Acc18:0.7653, F1_18:0.7859 | F1_bin:0.9508, F1_9:0.7925, F1_avg:0.8716
Epoch 58 | Loss:0.0877 | Acc18:0.7629, F1_18:0.7835 | F1_bin:0.9496, F1_9:0.7906, F1_avg:0.8701
Epoch 59 | Loss:0.0905 | Acc18:0.7653, F1_18:0.7840 | F1_bin:0.9502, F1_9:0.7910, F1_avg:0.8706
Epoch 60 | Loss:0.0942 | Acc18:0.7641, F1_18:0.7869 | F1_bin:0.9490, F1_9:0.7936, F1_avg:0.8713
Epoch 00061: reducing learning rate of group 0 to 3.9063e-06.
Epoch 61 | Loss:0.0899 | Acc18:0.7647, F1_18:0.7857 | F1_bin:0.9502, F1_9:0.7908, F1_avg:0.8705
Epoch 62 | Loss:0.0910 | Acc18:0.7647, F1_18:0.7868 | F1_bin:0.9459, F1_9:0.7937, F1_avg:0.8698
Epoch 63 | Loss:0.0915 | Acc18:0.7629, F1_18:0.7839 | F1_bin:0.9489, F1_9:0.7916, F1_avg:0.8703
Epoch 64 | Loss:0.0873 | Acc18:0.7678, F1_18:0.7891 | F1_bin:0.9520, F1_9:0.7946, F1_avg:0.8733
Epoch 00065: reducing learning rate of group 0 to 1.9531e-06.
Epoch 65 | Loss:0.0830 | Acc18:0.7653, F1_18:0.7891 | F1_bin:0.9502, F1_9:0.7948, F1_avg:0.8725
Epoch 66 | Loss:0.0909 | Acc18:0.7629, F1_18:0.7832 | F1_bin:0.9477, F1_9:0.7926, F1_avg:0.8702
Epoch 67 | Loss:0.0881 | Acc18:0.7659, F1_18:0.7874 | F1_bin:0.9502, F1_9:0.7939, F1_avg:0.8721
Epoch 68 | Loss:0.0890 | Acc18:0.7653, F1_18:0.7841 | F1_bin:0.9508, F1_9:0.7910, F1_avg:0.8709
Epoch 00069: reducing learning rate of group 0 to 9.7656e-07.
Epoch 69 | Loss:0.0913 | Acc18:0.7665, F1_18:0.7872 | F1_bin:0.9514, F1_9:0.7939, F1_avg:0.8726
Early stopping.

=== Fold 2 ===
Epoch 1 | Loss:1.6757 | Acc18:0.4537, F1_18:0.4045 | F1_bin:0.7820, F1_9:0.4671, F1_avg:0.6245
Epoch 2 | Loss:1.1702 | Acc18:0.5254, F1_18:0.5313 | F1_bin:0.8516, F1_9:0.5857, F1_avg:0.7187
Epoch 3 | Loss:1.0070 | Acc18:0.5708, F1_18:0.5756 | F1_bin:0.8552, F1_9:0.6113, F1_avg:0.7332
Epoch 4 | Loss:0.9125 | Acc18:0.5727, F1_18:0.6031 | F1_bin:0.8611, F1_9:0.6267, F1_avg:0.7439
Epoch 5 | Loss:0.8186 | Acc18:0.5769, F1_18:0.6155 | F1_bin:0.8554, F1_9:0.6409, F1_avg:0.7481
Epoch 6 | Loss:0.7754 | Acc18:0.5825, F1_18:0.6163 | F1_bin:0.8546, F1_9:0.6292, F1_avg:0.7419
Epoch 7 | Loss:0.7137 | Acc18:0.5972, F1_18:0.6116 | F1_bin:0.8711, F1_9:0.6503, F1_avg:0.7607
Epoch 8 | Loss:0.6952 | Acc18:0.5947, F1_18:0.6159 | F1_bin:0.8644, F1_9:0.6518, F1_avg:0.7581
Epoch 9 | Loss:0.6456 | Acc18:0.6217, F1_18:0.6367 | F1_bin:0.8659, F1_9:0.6630, F1_avg:0.7645
Epoch 10 | Loss:0.6152 | Acc18:0.6002, F1_18:0.5967 | F1_bin:0.8703, F1_9:0.6359, F1_avg:0.7531
Epoch 11 | Loss:0.6066 | Acc18:0.6217, F1_18:0.6597 | F1_bin:0.8708, F1_9:0.6732, F1_avg:0.7720
Epoch 12 | Loss:0.5563 | Acc18:0.5904, F1_18:0.6073 | F1_bin:0.8448, F1_9:0.6254, F1_avg:0.7351
Epoch 13 | Loss:0.5434 | Acc18:0.5960, F1_18:0.6194 | F1_bin:0.8681, F1_9:0.6518, F1_avg:0.7599
Epoch 14 | Loss:0.5269 | Acc18:0.6291, F1_18:0.6607 | F1_bin:0.8721, F1_9:0.6745, F1_avg:0.7733
Epoch 15 | Loss:0.4967 | Acc18:0.6168, F1_18:0.6489 | F1_bin:0.8699, F1_9:0.6725, F1_avg:0.7712
Epoch 16 | Loss:0.4740 | Acc18:0.6193, F1_18:0.6513 | F1_bin:0.8744, F1_9:0.6721, F1_avg:0.7732
Epoch 17 | Loss:0.4486 | Acc18:0.6475, F1_18:0.6797 | F1_bin:0.8851, F1_9:0.6968, F1_avg:0.7910
Epoch 18 | Loss:0.4375 | Acc18:0.6284, F1_18:0.6619 | F1_bin:0.8769, F1_9:0.6829, F1_avg:0.7799
Epoch 19 | Loss:0.4310 | Acc18:0.6107, F1_18:0.6453 | F1_bin:0.8657, F1_9:0.6629, F1_avg:0.7643
Epoch 20 | Loss:0.4226 | Acc18:0.6376, F1_18:0.6631 | F1_bin:0.8790, F1_9:0.6853, F1_avg:0.7822
Epoch 00021: reducing learning rate of group 0 to 5.0000e-04.
Epoch 21 | Loss:0.3825 | Acc18:0.6272, F1_18:0.6453 | F1_bin:0.8876, F1_9:0.6766, F1_avg:0.7821
Epoch 22 | Loss:0.3205 | Acc18:0.6407, F1_18:0.6703 | F1_bin:0.8767, F1_9:0.6928, F1_avg:0.7848
Epoch 23 | Loss:0.2793 | Acc18:0.6432, F1_18:0.6724 | F1_bin:0.8840, F1_9:0.6951, F1_avg:0.7895
Epoch 24 | Loss:0.2547 | Acc18:0.6407, F1_18:0.6730 | F1_bin:0.8823, F1_9:0.6878, F1_avg:0.7851
Epoch 00025: reducing learning rate of group 0 to 2.5000e-04.
Epoch 25 | Loss:0.2526 | Acc18:0.6450, F1_18:0.6732 | F1_bin:0.8772, F1_9:0.6980, F1_avg:0.7876
Epoch 26 | Loss:0.2227 | Acc18:0.6530, F1_18:0.6860 | F1_bin:0.8832, F1_9:0.7056, F1_avg:0.7944
Epoch 27 | Loss:0.1983 | Acc18:0.6542, F1_18:0.6828 | F1_bin:0.8796, F1_9:0.7033, F1_avg:0.7915
Epoch 28 | Loss:0.1804 | Acc18:0.6468, F1_18:0.6783 | F1_bin:0.8822, F1_9:0.6957, F1_avg:0.7889
Epoch 29 | Loss:0.1864 | Acc18:0.6671, F1_18:0.6996 | F1_bin:0.8827, F1_9:0.7178, F1_avg:0.8002
Epoch 30 | Loss:0.1802 | Acc18:0.6475, F1_18:0.6856 | F1_bin:0.8802, F1_9:0.6996, F1_avg:0.7899
Epoch 31 | Loss:0.1655 | Acc18:0.6628, F1_18:0.6964 | F1_bin:0.8783, F1_9:0.7144, F1_avg:0.7963
Epoch 32 | Loss:0.1619 | Acc18:0.6493, F1_18:0.6837 | F1_bin:0.8802, F1_9:0.7007, F1_avg:0.7904
Epoch 00033: reducing learning rate of group 0 to 1.2500e-04.
Epoch 33 | Loss:0.1578 | Acc18:0.6511, F1_18:0.6788 | F1_bin:0.8784, F1_9:0.7006, F1_avg:0.7895
Epoch 34 | Loss:0.1382 | Acc18:0.6560, F1_18:0.6906 | F1_bin:0.8791, F1_9:0.7060, F1_avg:0.7925
Epoch 35 | Loss:0.1381 | Acc18:0.6646, F1_18:0.6967 | F1_bin:0.8820, F1_9:0.7141, F1_avg:0.7981
Epoch 36 | Loss:0.1270 | Acc18:0.6536, F1_18:0.6841 | F1_bin:0.8797, F1_9:0.7039, F1_avg:0.7918
Epoch 00037: reducing learning rate of group 0 to 6.2500e-05.
Epoch 37 | Loss:0.1279 | Acc18:0.6616, F1_18:0.6956 | F1_bin:0.8840, F1_9:0.7112, F1_avg:0.7976
Epoch 38 | Loss:0.1229 | Acc18:0.6591, F1_18:0.6932 | F1_bin:0.8858, F1_9:0.7108, F1_avg:0.7983
Epoch 39 | Loss:0.1189 | Acc18:0.6560, F1_18:0.6879 | F1_bin:0.8783, F1_9:0.7048, F1_avg:0.7915
Epoch 40 | Loss:0.1147 | Acc18:0.6665, F1_18:0.7012 | F1_bin:0.8840, F1_9:0.7142, F1_avg:0.7991
Epoch 41 | Loss:0.1150 | Acc18:0.6622, F1_18:0.6937 | F1_bin:0.8814, F1_9:0.7126, F1_avg:0.7970
Epoch 42 | Loss:0.1075 | Acc18:0.6622, F1_18:0.6966 | F1_bin:0.8827, F1_9:0.7094, F1_avg:0.7960
Epoch 43 | Loss:0.1051 | Acc18:0.6640, F1_18:0.6980 | F1_bin:0.8882, F1_9:0.7129, F1_avg:0.8006
Epoch 00044: reducing learning rate of group 0 to 3.1250e-05.
Epoch 44 | Loss:0.1066 | Acc18:0.6579, F1_18:0.6908 | F1_bin:0.8851, F1_9:0.7074, F1_avg:0.7963
Epoch 45 | Loss:0.0983 | Acc18:0.6628, F1_18:0.6940 | F1_bin:0.8851, F1_9:0.7113, F1_avg:0.7982
Epoch 46 | Loss:0.1002 | Acc18:0.6591, F1_18:0.6936 | F1_bin:0.8858, F1_9:0.7083, F1_avg:0.7970
Epoch 47 | Loss:0.0946 | Acc18:0.6616, F1_18:0.6939 | F1_bin:0.8845, F1_9:0.7109, F1_avg:0.7977
Epoch 00048: reducing learning rate of group 0 to 1.5625e-05.
Epoch 48 | Loss:0.0955 | Acc18:0.6579, F1_18:0.6913 | F1_bin:0.8839, F1_9:0.7048, F1_avg:0.7943
Epoch 49 | Loss:0.0950 | Acc18:0.6597, F1_18:0.6900 | F1_bin:0.8826, F1_9:0.7074, F1_avg:0.7950
Epoch 50 | Loss:0.0923 | Acc18:0.6585, F1_18:0.6896 | F1_bin:0.8809, F1_9:0.7069, F1_avg:0.7939
Epoch 51 | Loss:0.0919 | Acc18:0.6622, F1_18:0.6935 | F1_bin:0.8839, F1_9:0.7130, F1_avg:0.7985
Epoch 00052: reducing learning rate of group 0 to 7.8125e-06.
Epoch 52 | Loss:0.0991 | Acc18:0.6597, F1_18:0.6913 | F1_bin:0.8839, F1_9:0.7092, F1_avg:0.7966
Epoch 53 | Loss:0.0963 | Acc18:0.6573, F1_18:0.6888 | F1_bin:0.8845, F1_9:0.7042, F1_avg:0.7944
Epoch 54 | Loss:0.0941 | Acc18:0.6579, F1_18:0.6882 | F1_bin:0.8846, F1_9:0.7038, F1_avg:0.7942
Epoch 55 | Loss:0.0890 | Acc18:0.6597, F1_18:0.6892 | F1_bin:0.8833, F1_9:0.7069, F1_avg:0.7951
Epoch 00056: reducing learning rate of group 0 to 3.9063e-06.
Epoch 56 | Loss:0.0903 | Acc18:0.6628, F1_18:0.6922 | F1_bin:0.8876, F1_9:0.7082, F1_avg:0.7979
Epoch 57 | Loss:0.0906 | Acc18:0.6591, F1_18:0.6900 | F1_bin:0.8857, F1_9:0.7065, F1_avg:0.7961
Epoch 58 | Loss:0.0927 | Acc18:0.6597, F1_18:0.6910 | F1_bin:0.8815, F1_9:0.7099, F1_avg:0.7957
Epoch 59 | Loss:0.0866 | Acc18:0.6579, F1_18:0.6863 | F1_bin:0.8858, F1_9:0.7028, F1_avg:0.7943
Epoch 00060: reducing learning rate of group 0 to 1.9531e-06.
Epoch 60 | Loss:0.0935 | Acc18:0.6591, F1_18:0.6866 | F1_bin:0.8827, F1_9:0.7056, F1_avg:0.7941
Early stopping.

=== Fold 3 ===
Epoch 1 | Loss:1.7392 | Acc18:0.5114, F1_18:0.5268 | F1_bin:0.8356, F1_9:0.5472, F1_avg:0.6914
Epoch 2 | Loss:1.2153 | Acc18:0.6171, F1_18:0.6193 | F1_bin:0.8946, F1_9:0.6554, F1_avg:0.7750
Epoch 3 | Loss:1.0470 | Acc18:0.6238, F1_18:0.6568 | F1_bin:0.8874, F1_9:0.6768, F1_avg:0.7821
Epoch 4 | Loss:0.9343 | Acc18:0.6159, F1_18:0.6523 | F1_bin:0.8822, F1_9:0.6796, F1_avg:0.7809
Epoch 5 | Loss:0.8736 | Acc18:0.6454, F1_18:0.6653 | F1_bin:0.8942, F1_9:0.6886, F1_avg:0.7914
Epoch 6 | Loss:0.8017 | Acc18:0.6134, F1_18:0.6485 | F1_bin:0.8932, F1_9:0.6677, F1_avg:0.7804
Epoch 7 | Loss:0.7688 | Acc18:0.6435, F1_18:0.6683 | F1_bin:0.8929, F1_9:0.6929, F1_avg:0.7929
Epoch 8 | Loss:0.7185 | Acc18:0.6220, F1_18:0.6583 | F1_bin:0.9020, F1_9:0.6850, F1_avg:0.7935
Epoch 9 | Loss:0.6826 | Acc18:0.6540, F1_18:0.6734 | F1_bin:0.8982, F1_9:0.6957, F1_avg:0.7970
Epoch 10 | Loss:0.6516 | Acc18:0.6804, F1_18:0.6994 | F1_bin:0.8960, F1_9:0.7208, F1_avg:0.8084
Epoch 11 | Loss:0.6187 | Acc18:0.6773, F1_18:0.6958 | F1_bin:0.8965, F1_9:0.7144, F1_avg:0.8054
Epoch 12 | Loss:0.6072 | Acc18:0.6847, F1_18:0.7020 | F1_bin:0.9044, F1_9:0.7213, F1_avg:0.8129
Epoch 13 | Loss:0.5870 | Acc18:0.7031, F1_18:0.7164 | F1_bin:0.9136, F1_9:0.7357, F1_avg:0.8246
Epoch 14 | Loss:0.5543 | Acc18:0.6724, F1_18:0.6956 | F1_bin:0.9097, F1_9:0.7145, F1_avg:0.8121
Epoch 15 | Loss:0.5207 | Acc18:0.6994, F1_18:0.7136 | F1_bin:0.9051, F1_9:0.7418, F1_avg:0.8235
Epoch 16 | Loss:0.5051 | Acc18:0.6859, F1_18:0.7089 | F1_bin:0.8979, F1_9:0.7286, F1_avg:0.8133
Epoch 00017: reducing learning rate of group 0 to 5.0000e-04.
Epoch 17 | Loss:0.4916 | Acc18:0.6601, F1_18:0.6859 | F1_bin:0.8886, F1_9:0.7060, F1_avg:0.7973
Epoch 18 | Loss:0.4082 | Acc18:0.7234, F1_18:0.7399 | F1_bin:0.9125, F1_9:0.7548, F1_avg:0.8337
Epoch 19 | Loss:0.3939 | Acc18:0.7203, F1_18:0.7331 | F1_bin:0.9172, F1_9:0.7492, F1_avg:0.8332
Epoch 20 | Loss:0.3576 | Acc18:0.6976, F1_18:0.7236 | F1_bin:0.9103, F1_9:0.7328, F1_avg:0.8215
Epoch 21 | Loss:0.3428 | Acc18:0.7185, F1_18:0.7330 | F1_bin:0.9177, F1_9:0.7494, F1_avg:0.8336
Epoch 00022: reducing learning rate of group 0 to 2.5000e-04.
Epoch 22 | Loss:0.3301 | Acc18:0.7130, F1_18:0.7306 | F1_bin:0.9061, F1_9:0.7436, F1_avg:0.8248
Epoch 23 | Loss:0.2895 | Acc18:0.7259, F1_18:0.7440 | F1_bin:0.9124, F1_9:0.7576, F1_avg:0.8350
Epoch 24 | Loss:0.2720 | Acc18:0.7093, F1_18:0.7263 | F1_bin:0.9099, F1_9:0.7450, F1_avg:0.8275
Epoch 25 | Loss:0.2590 | Acc18:0.7228, F1_18:0.7388 | F1_bin:0.9104, F1_9:0.7567, F1_avg:0.8335
Epoch 26 | Loss:0.2556 | Acc18:0.7234, F1_18:0.7384 | F1_bin:0.9118, F1_9:0.7587, F1_avg:0.8352
Epoch 00027: reducing learning rate of group 0 to 1.2500e-04.
Epoch 27 | Loss:0.2379 | Acc18:0.7203, F1_18:0.7381 | F1_bin:0.9144, F1_9:0.7558, F1_avg:0.8351
Epoch 28 | Loss:0.2132 | Acc18:0.7197, F1_18:0.7384 | F1_bin:0.9093, F1_9:0.7576, F1_avg:0.8335
Epoch 29 | Loss:0.2060 | Acc18:0.7216, F1_18:0.7404 | F1_bin:0.9124, F1_9:0.7569, F1_avg:0.8347
Epoch 30 | Loss:0.2008 | Acc18:0.7259, F1_18:0.7469 | F1_bin:0.9136, F1_9:0.7629, F1_avg:0.8383
Epoch 31 | Loss:0.1958 | Acc18:0.7234, F1_18:0.7416 | F1_bin:0.9051, F1_9:0.7593, F1_avg:0.8322
Epoch 32 | Loss:0.1849 | Acc18:0.7271, F1_18:0.7457 | F1_bin:0.9118, F1_9:0.7644, F1_avg:0.8381
Epoch 33 | Loss:0.1792 | Acc18:0.7339, F1_18:0.7509 | F1_bin:0.9130, F1_9:0.7693, F1_avg:0.8411
Epoch 34 | Loss:0.1809 | Acc18:0.7277, F1_18:0.7372 | F1_bin:0.9198, F1_9:0.7593, F1_avg:0.8395
Epoch 35 | Loss:0.1659 | Acc18:0.7277, F1_18:0.7446 | F1_bin:0.9143, F1_9:0.7618, F1_avg:0.8381
Epoch 36 | Loss:0.1635 | Acc18:0.7210, F1_18:0.7380 | F1_bin:0.9136, F1_9:0.7562, F1_avg:0.8349
Epoch 00037: reducing learning rate of group 0 to 6.2500e-05.
Epoch 37 | Loss:0.1660 | Acc18:0.7216, F1_18:0.7402 | F1_bin:0.9130, F1_9:0.7567, F1_avg:0.8349
Epoch 38 | Loss:0.1509 | Acc18:0.7283, F1_18:0.7447 | F1_bin:0.9160, F1_9:0.7621, F1_avg:0.8390
Epoch 39 | Loss:0.1461 | Acc18:0.7277, F1_18:0.7437 | F1_bin:0.9118, F1_9:0.7641, F1_avg:0.8380
Epoch 40 | Loss:0.1451 | Acc18:0.7326, F1_18:0.7478 | F1_bin:0.9154, F1_9:0.7667, F1_avg:0.8411
Epoch 00041: reducing learning rate of group 0 to 3.1250e-05.
Epoch 41 | Loss:0.1479 | Acc18:0.7302, F1_18:0.7437 | F1_bin:0.9135, F1_9:0.7632, F1_avg:0.8384
Epoch 42 | Loss:0.1367 | Acc18:0.7320, F1_18:0.7463 | F1_bin:0.9124, F1_9:0.7680, F1_avg:0.8402
Epoch 43 | Loss:0.1416 | Acc18:0.7277, F1_18:0.7449 | F1_bin:0.9130, F1_9:0.7635, F1_avg:0.8383
Epoch 44 | Loss:0.1293 | Acc18:0.7289, F1_18:0.7418 | F1_bin:0.9172, F1_9:0.7615, F1_avg:0.8393
Epoch 00045: reducing learning rate of group 0 to 1.5625e-05.
Epoch 45 | Loss:0.1315 | Acc18:0.7259, F1_18:0.7390 | F1_bin:0.9130, F1_9:0.7602, F1_avg:0.8366
Epoch 46 | Loss:0.1286 | Acc18:0.7283, F1_18:0.7416 | F1_bin:0.9136, F1_9:0.7631, F1_avg:0.8383
Epoch 47 | Loss:0.1314 | Acc18:0.7296, F1_18:0.7407 | F1_bin:0.9118, F1_9:0.7634, F1_avg:0.8376
Epoch 48 | Loss:0.1269 | Acc18:0.7283, F1_18:0.7396 | F1_bin:0.9130, F1_9:0.7613, F1_avg:0.8371
Epoch 00049: reducing learning rate of group 0 to 7.8125e-06.
Epoch 49 | Loss:0.1281 | Acc18:0.7271, F1_18:0.7433 | F1_bin:0.9130, F1_9:0.7623, F1_avg:0.8377
Epoch 50 | Loss:0.1264 | Acc18:0.7271, F1_18:0.7427 | F1_bin:0.9167, F1_9:0.7622, F1_avg:0.8394
Epoch 51 | Loss:0.1205 | Acc18:0.7228, F1_18:0.7382 | F1_bin:0.9111, F1_9:0.7589, F1_avg:0.8350
Epoch 52 | Loss:0.1260 | Acc18:0.7289, F1_18:0.7442 | F1_bin:0.9111, F1_9:0.7654, F1_avg:0.8382
Epoch 00053: reducing learning rate of group 0 to 3.9063e-06.
Epoch 53 | Loss:0.1282 | Acc18:0.7277, F1_18:0.7421 | F1_bin:0.9123, F1_9:0.7627, F1_avg:0.8375
Early stopping.

=== Fold 4 ===
Epoch 1 | Loss:1.7722 | Acc18:0.4963, F1_18:0.5140 | F1_bin:0.8270, F1_9:0.5252, F1_avg:0.6761
Epoch 2 | Loss:1.2419 | Acc18:0.5486, F1_18:0.5581 | F1_bin:0.8264, F1_9:0.5655, F1_avg:0.6959
Epoch 3 | Loss:1.0585 | Acc18:0.5941, F1_18:0.6214 | F1_bin:0.8729, F1_9:0.6363, F1_avg:0.7546
Epoch 4 | Loss:0.9582 | Acc18:0.5898, F1_18:0.6080 | F1_bin:0.8682, F1_9:0.6308, F1_avg:0.7495
Epoch 5 | Loss:0.8762 | Acc18:0.6039, F1_18:0.6095 | F1_bin:0.8615, F1_9:0.6447, F1_avg:0.7531
Epoch 6 | Loss:0.8176 | Acc18:0.6304, F1_18:0.6485 | F1_bin:0.8867, F1_9:0.6710, F1_avg:0.7789
Epoch 7 | Loss:0.7680 | Acc18:0.6396, F1_18:0.6526 | F1_bin:0.8911, F1_9:0.6768, F1_avg:0.7840
Epoch 8 | Loss:0.7233 | Acc18:0.6365, F1_18:0.6632 | F1_bin:0.8831, F1_9:0.6857, F1_avg:0.7844
Epoch 9 | Loss:0.6710 | Acc18:0.6476, F1_18:0.6666 | F1_bin:0.8876, F1_9:0.6905, F1_avg:0.7890
Epoch 10 | Loss:0.6609 | Acc18:0.6759, F1_18:0.6911 | F1_bin:0.8941, F1_9:0.7090, F1_avg:0.8016
Epoch 11 | Loss:0.6312 | Acc18:0.6371, F1_18:0.6603 | F1_bin:0.8930, F1_9:0.6852, F1_avg:0.7891
Epoch 12 | Loss:0.5911 | Acc18:0.6845, F1_18:0.7089 | F1_bin:0.9112, F1_9:0.7161, F1_avg:0.8137
Epoch 13 | Loss:0.5677 | Acc18:0.6654, F1_18:0.6840 | F1_bin:0.9003, F1_9:0.7033, F1_avg:0.8018
Epoch 14 | Loss:0.5456 | Acc18:0.6685, F1_18:0.6984 | F1_bin:0.8831, F1_9:0.7163, F1_avg:0.7997
Epoch 15 | Loss:0.5172 | Acc18:0.6790, F1_18:0.7088 | F1_bin:0.9013, F1_9:0.7174, F1_avg:0.8094
Epoch 00016: reducing learning rate of group 0 to 5.0000e-04.
Epoch 16 | Loss:0.5027 | Acc18:0.6630, F1_18:0.6801 | F1_bin:0.8964, F1_9:0.6963, F1_avg:0.7964
Epoch 17 | Loss:0.4257 | Acc18:0.6919, F1_18:0.7150 | F1_bin:0.8973, F1_9:0.7264, F1_avg:0.8118
Epoch 18 | Loss:0.3812 | Acc18:0.7134, F1_18:0.7406 | F1_bin:0.9033, F1_9:0.7541, F1_avg:0.8287
Epoch 19 | Loss:0.3639 | Acc18:0.7017, F1_18:0.7304 | F1_bin:0.9039, F1_9:0.7425, F1_avg:0.8232
Epoch 20 | Loss:0.3350 | Acc18:0.6808, F1_18:0.7085 | F1_bin:0.8905, F1_9:0.7233, F1_avg:0.8069
Epoch 21 | Loss:0.3293 | Acc18:0.7134, F1_18:0.7342 | F1_bin:0.9040, F1_9:0.7457, F1_avg:0.8249
Epoch 00022: reducing learning rate of group 0 to 2.5000e-04.
Epoch 22 | Loss:0.3146 | Acc18:0.6913, F1_18:0.7241 | F1_bin:0.8997, F1_9:0.7333, F1_avg:0.8165
Epoch 23 | Loss:0.2743 | Acc18:0.7134, F1_18:0.7346 | F1_bin:0.9038, F1_9:0.7498, F1_avg:0.8268
Epoch 24 | Loss:0.2450 | Acc18:0.7109, F1_18:0.7414 | F1_bin:0.9046, F1_9:0.7479, F1_avg:0.8263
Epoch 25 | Loss:0.2318 | Acc18:0.7060, F1_18:0.7438 | F1_bin:0.9026, F1_9:0.7526, F1_avg:0.8276
Epoch 26 | Loss:0.2280 | Acc18:0.7202, F1_18:0.7440 | F1_bin:0.9077, F1_9:0.7531, F1_avg:0.8304
Epoch 27 | Loss:0.2299 | Acc18:0.7109, F1_18:0.7362 | F1_bin:0.9058, F1_9:0.7484, F1_avg:0.8271
Epoch 28 | Loss:0.2085 | Acc18:0.7017, F1_18:0.7341 | F1_bin:0.8991, F1_9:0.7413, F1_avg:0.8202
Epoch 29 | Loss:0.2070 | Acc18:0.6999, F1_18:0.7260 | F1_bin:0.9033, F1_9:0.7393, F1_avg:0.8213
Epoch 00030: reducing learning rate of group 0 to 1.2500e-04.
Epoch 30 | Loss:0.1900 | Acc18:0.6900, F1_18:0.7265 | F1_bin:0.8948, F1_9:0.7398, F1_avg:0.8173
Epoch 31 | Loss:0.1794 | Acc18:0.7023, F1_18:0.7320 | F1_bin:0.9015, F1_9:0.7411, F1_avg:0.8213
Epoch 32 | Loss:0.1644 | Acc18:0.6986, F1_18:0.7343 | F1_bin:0.9028, F1_9:0.7442, F1_avg:0.8235
Epoch 33 | Loss:0.1613 | Acc18:0.7146, F1_18:0.7470 | F1_bin:0.9089, F1_9:0.7509, F1_avg:0.8299
Epoch 34 | Loss:0.1512 | Acc18:0.7226, F1_18:0.7519 | F1_bin:0.9150, F1_9:0.7579, F1_avg:0.8364
Epoch 35 | Loss:0.1466 | Acc18:0.7116, F1_18:0.7423 | F1_bin:0.9046, F1_9:0.7541, F1_avg:0.8294
Epoch 36 | Loss:0.1465 | Acc18:0.7208, F1_18:0.7483 | F1_bin:0.9125, F1_9:0.7520, F1_avg:0.8323
Epoch 37 | Loss:0.1347 | Acc18:0.7128, F1_18:0.7402 | F1_bin:0.9120, F1_9:0.7479, F1_avg:0.8300
Epoch 00038: reducing learning rate of group 0 to 6.2500e-05.
Epoch 38 | Loss:0.1331 | Acc18:0.7048, F1_18:0.7351 | F1_bin:0.9101, F1_9:0.7428, F1_avg:0.8265
Epoch 39 | Loss:0.1237 | Acc18:0.7091, F1_18:0.7364 | F1_bin:0.9077, F1_9:0.7486, F1_avg:0.8281
Epoch 40 | Loss:0.1231 | Acc18:0.7134, F1_18:0.7412 | F1_bin:0.9089, F1_9:0.7502, F1_avg:0.8296
Epoch 41 | Loss:0.1145 | Acc18:0.7171, F1_18:0.7456 | F1_bin:0.9064, F1_9:0.7546, F1_avg:0.8305
Epoch 00042: reducing learning rate of group 0 to 3.1250e-05.
Epoch 42 | Loss:0.1138 | Acc18:0.7153, F1_18:0.7427 | F1_bin:0.9107, F1_9:0.7494, F1_avg:0.8301
Epoch 43 | Loss:0.1071 | Acc18:0.7116, F1_18:0.7424 | F1_bin:0.9083, F1_9:0.7490, F1_avg:0.8286
Epoch 44 | Loss:0.1060 | Acc18:0.7165, F1_18:0.7458 | F1_bin:0.9107, F1_9:0.7520, F1_avg:0.8314
Epoch 45 | Loss:0.1112 | Acc18:0.7079, F1_18:0.7380 | F1_bin:0.9083, F1_9:0.7480, F1_avg:0.8281
Epoch 00046: reducing learning rate of group 0 to 1.5625e-05.
Epoch 46 | Loss:0.1079 | Acc18:0.7109, F1_18:0.7395 | F1_bin:0.9077, F1_9:0.7485, F1_avg:0.8281
Epoch 47 | Loss:0.1066 | Acc18:0.7116, F1_18:0.7399 | F1_bin:0.9076, F1_9:0.7493, F1_avg:0.8285
Epoch 48 | Loss:0.0986 | Acc18:0.7103, F1_18:0.7413 | F1_bin:0.9058, F1_9:0.7510, F1_avg:0.8284
Epoch 49 | Loss:0.1000 | Acc18:0.7109, F1_18:0.7384 | F1_bin:0.9083, F1_9:0.7466, F1_avg:0.8274
Epoch 00050: reducing learning rate of group 0 to 7.8125e-06.
Epoch 50 | Loss:0.0992 | Acc18:0.7116, F1_18:0.7395 | F1_bin:0.9095, F1_9:0.7471, F1_avg:0.8283
Epoch 51 | Loss:0.1024 | Acc18:0.7091, F1_18:0.7363 | F1_bin:0.9052, F1_9:0.7456, F1_avg:0.8254
Epoch 52 | Loss:0.0970 | Acc18:0.7116, F1_18:0.7412 | F1_bin:0.9089, F1_9:0.7502, F1_avg:0.8295
Epoch 53 | Loss:0.1002 | Acc18:0.7128, F1_18:0.7415 | F1_bin:0.9070, F1_9:0.7497, F1_avg:0.8283
Epoch 00054: reducing learning rate of group 0 to 3.9063e-06.
Epoch 54 | Loss:0.1015 | Acc18:0.7091, F1_18:0.7343 | F1_bin:0.9052, F1_9:0.7468, F1_avg:0.8260
Early stopping.

=== Fold 5 ===
Epoch 1 | Loss:1.7345 | Acc18:0.4869, F1_18:0.4875 | F1_bin:0.8317, F1_9:0.5194, F1_avg:0.6755
Epoch 2 | Loss:1.1943 | Acc18:0.5321, F1_18:0.5496 | F1_bin:0.8565, F1_9:0.5693, F1_avg:0.7129
Epoch 3 | Loss:1.0260 | Acc18:0.5994, F1_18:0.6063 | F1_bin:0.8770, F1_9:0.6310, F1_avg:0.7540
Epoch 4 | Loss:0.9329 | Acc18:0.5939, F1_18:0.6118 | F1_bin:0.8809, F1_9:0.6315, F1_avg:0.7562
Epoch 5 | Loss:0.8686 | Acc18:0.6251, F1_18:0.6554 | F1_bin:0.8898, F1_9:0.6695, F1_avg:0.7796
Epoch 6 | Loss:0.8113 | Acc18:0.6391, F1_18:0.6691 | F1_bin:0.8912, F1_9:0.6876, F1_avg:0.7894
Epoch 7 | Loss:0.7518 | Acc18:0.6232, F1_18:0.6329 | F1_bin:0.8955, F1_9:0.6521, F1_avg:0.7738
Epoch 8 | Loss:0.7192 | Acc18:0.6153, F1_18:0.6465 | F1_bin:0.8850, F1_9:0.6699, F1_avg:0.7775
Epoch 9 | Loss:0.6870 | Acc18:0.6410, F1_18:0.6746 | F1_bin:0.8995, F1_9:0.6819, F1_avg:0.7907
Epoch 10 | Loss:0.6497 | Acc18:0.6520, F1_18:0.6624 | F1_bin:0.9104, F1_9:0.6829, F1_avg:0.7966
Epoch 11 | Loss:0.6255 | Acc18:0.6312, F1_18:0.6591 | F1_bin:0.9053, F1_9:0.6720, F1_avg:0.7887
Epoch 12 | Loss:0.5931 | Acc18:0.6367, F1_18:0.6693 | F1_bin:0.8966, F1_9:0.6836, F1_avg:0.7901
Epoch 13 | Loss:0.5632 | Acc18:0.6508, F1_18:0.6756 | F1_bin:0.9122, F1_9:0.6865, F1_avg:0.7994
Epoch 14 | Loss:0.5316 | Acc18:0.6722, F1_18:0.6963 | F1_bin:0.9208, F1_9:0.7119, F1_avg:0.8164
Epoch 15 | Loss:0.5192 | Acc18:0.6673, F1_18:0.6875 | F1_bin:0.9172, F1_9:0.7074, F1_avg:0.8123
Epoch 16 | Loss:0.5062 | Acc18:0.6520, F1_18:0.6697 | F1_bin:0.8980, F1_9:0.6888, F1_avg:0.7934
Epoch 17 | Loss:0.4777 | Acc18:0.6508, F1_18:0.6720 | F1_bin:0.9125, F1_9:0.6870, F1_avg:0.7997
Epoch 00018: reducing learning rate of group 0 to 5.0000e-04.
Epoch 18 | Loss:0.4604 | Acc18:0.6709, F1_18:0.6878 | F1_bin:0.9057, F1_9:0.7070, F1_avg:0.8063
Epoch 19 | Loss:0.3763 | Acc18:0.6801, F1_18:0.7118 | F1_bin:0.9161, F1_9:0.7240, F1_avg:0.8201
Epoch 20 | Loss:0.3372 | Acc18:0.6924, F1_18:0.7230 | F1_bin:0.9270, F1_9:0.7321, F1_avg:0.8296
Epoch 21 | Loss:0.3201 | Acc18:0.6930, F1_18:0.7240 | F1_bin:0.9148, F1_9:0.7409, F1_avg:0.8279
Epoch 22 | Loss:0.3156 | Acc18:0.6856, F1_18:0.7090 | F1_bin:0.9176, F1_9:0.7246, F1_avg:0.8211
Epoch 23 | Loss:0.2891 | Acc18:0.6862, F1_18:0.7170 | F1_bin:0.9194, F1_9:0.7302, F1_avg:0.8248
Epoch 24 | Loss:0.2967 | Acc18:0.6838, F1_18:0.7134 | F1_bin:0.9154, F1_9:0.7249, F1_avg:0.8202
Epoch 00025: reducing learning rate of group 0 to 2.5000e-04.
Epoch 25 | Loss:0.2780 | Acc18:0.6887, F1_18:0.7196 | F1_bin:0.9155, F1_9:0.7312, F1_avg:0.8234
Epoch 26 | Loss:0.2248 | Acc18:0.6948, F1_18:0.7276 | F1_bin:0.9197, F1_9:0.7365, F1_avg:0.8281
Epoch 27 | Loss:0.2144 | Acc18:0.6954, F1_18:0.7262 | F1_bin:0.9197, F1_9:0.7339, F1_avg:0.8268
Epoch 28 | Loss:0.2061 | Acc18:0.6875, F1_18:0.7236 | F1_bin:0.9147, F1_9:0.7360, F1_avg:0.8253
Epoch 29 | Loss:0.1982 | Acc18:0.6966, F1_18:0.7285 | F1_bin:0.9233, F1_9:0.7402, F1_avg:0.8318
Epoch 30 | Loss:0.1950 | Acc18:0.6795, F1_18:0.7126 | F1_bin:0.9173, F1_9:0.7281, F1_avg:0.8227
Epoch 31 | Loss:0.1828 | Acc18:0.6924, F1_18:0.7212 | F1_bin:0.9227, F1_9:0.7337, F1_avg:0.8282
Epoch 32 | Loss:0.1662 | Acc18:0.6930, F1_18:0.7227 | F1_bin:0.9231, F1_9:0.7348, F1_avg:0.8290
Epoch 00033: reducing learning rate of group 0 to 1.2500e-04.
Epoch 33 | Loss:0.1683 | Acc18:0.6905, F1_18:0.7180 | F1_bin:0.9258, F1_9:0.7306, F1_avg:0.8282
Epoch 34 | Loss:0.1552 | Acc18:0.6869, F1_18:0.7199 | F1_bin:0.9239, F1_9:0.7334, F1_avg:0.8287
Epoch 35 | Loss:0.1496 | Acc18:0.6924, F1_18:0.7280 | F1_bin:0.9232, F1_9:0.7408, F1_avg:0.8320
Epoch 36 | Loss:0.1398 | Acc18:0.6905, F1_18:0.7225 | F1_bin:0.9240, F1_9:0.7332, F1_avg:0.8286
Epoch 00037: reducing learning rate of group 0 to 6.2500e-05.
Epoch 37 | Loss:0.1349 | Acc18:0.6905, F1_18:0.7248 | F1_bin:0.9220, F1_9:0.7335, F1_avg:0.8277
Epoch 38 | Loss:0.1278 | Acc18:0.6917, F1_18:0.7235 | F1_bin:0.9246, F1_9:0.7337, F1_avg:0.8291
Epoch 39 | Loss:0.1241 | Acc18:0.6924, F1_18:0.7185 | F1_bin:0.9196, F1_9:0.7328, F1_avg:0.8262
Epoch 40 | Loss:0.1188 | Acc18:0.6966, F1_18:0.7335 | F1_bin:0.9307, F1_9:0.7423, F1_avg:0.8365
Epoch 41 | Loss:0.1190 | Acc18:0.7009, F1_18:0.7354 | F1_bin:0.9264, F1_9:0.7492, F1_avg:0.8378
Epoch 42 | Loss:0.1125 | Acc18:0.6966, F1_18:0.7264 | F1_bin:0.9252, F1_9:0.7390, F1_avg:0.8321
Epoch 43 | Loss:0.1041 | Acc18:0.6948, F1_18:0.7227 | F1_bin:0.9257, F1_9:0.7353, F1_avg:0.8305
Epoch 44 | Loss:0.1144 | Acc18:0.6936, F1_18:0.7209 | F1_bin:0.9209, F1_9:0.7343, F1_avg:0.8276
Epoch 00045: reducing learning rate of group 0 to 3.1250e-05.
Epoch 45 | Loss:0.1080 | Acc18:0.6893, F1_18:0.7205 | F1_bin:0.9215, F1_9:0.7328, F1_avg:0.8272
Epoch 46 | Loss:0.1060 | Acc18:0.6960, F1_18:0.7285 | F1_bin:0.9239, F1_9:0.7376, F1_avg:0.8308
Epoch 47 | Loss:0.1009 | Acc18:0.7003, F1_18:0.7290 | F1_bin:0.9251, F1_9:0.7389, F1_avg:0.8320
Epoch 48 | Loss:0.1029 | Acc18:0.6966, F1_18:0.7283 | F1_bin:0.9214, F1_9:0.7406, F1_avg:0.8310
Epoch 00049: reducing learning rate of group 0 to 1.5625e-05.
Epoch 49 | Loss:0.1015 | Acc18:0.6924, F1_18:0.7211 | F1_bin:0.9221, F1_9:0.7339, F1_avg:0.8280
Epoch 50 | Loss:0.0989 | Acc18:0.6972, F1_18:0.7283 | F1_bin:0.9214, F1_9:0.7409, F1_avg:0.8312
Epoch 51 | Loss:0.0919 | Acc18:0.6942, F1_18:0.7252 | F1_bin:0.9203, F1_9:0.7381, F1_avg:0.8292
Epoch 52 | Loss:0.0966 | Acc18:0.6966, F1_18:0.7265 | F1_bin:0.9208, F1_9:0.7408, F1_avg:0.8308
Epoch 00053: reducing learning rate of group 0 to 7.8125e-06.
Epoch 53 | Loss:0.0950 | Acc18:0.6942, F1_18:0.7229 | F1_bin:0.9245, F1_9:0.7356, F1_avg:0.8300
Epoch 54 | Loss:0.0964 | Acc18:0.6936, F1_18:0.7226 | F1_bin:0.9227, F1_9:0.7365, F1_avg:0.8296
Epoch 55 | Loss:0.0978 | Acc18:0.6917, F1_18:0.7205 | F1_bin:0.9221, F1_9:0.7350, F1_avg:0.8285
Epoch 56 | Loss:0.0955 | Acc18:0.6930, F1_18:0.7262 | F1_bin:0.9196, F1_9:0.7417, F1_avg:0.8307
Epoch 00057: reducing learning rate of group 0 to 3.9063e-06.
Epoch 57 | Loss:0.0939 | Acc18:0.6905, F1_18:0.7212 | F1_bin:0.9202, F1_9:0.7351, F1_avg:0.8276
Epoch 58 | Loss:0.0942 | Acc18:0.6960, F1_18:0.7253 | F1_bin:0.9227, F1_9:0.7371, F1_avg:0.8299
Epoch 59 | Loss:0.0946 | Acc18:0.6942, F1_18:0.7253 | F1_bin:0.9226, F1_9:0.7374, F1_avg:0.8300
Epoch 60 | Loss:0.0947 | Acc18:0.6875, F1_18:0.7193 | F1_bin:0.9190, F1_9:0.7341, F1_avg:0.8266
Epoch 00061: reducing learning rate of group 0 to 1.9531e-06.
Epoch 61 | Loss:0.0907 | Acc18:0.6899, F1_18:0.7206 | F1_bin:0.9184, F1_9:0.7350, F1_avg:0.8267
Early stopping.
"""
