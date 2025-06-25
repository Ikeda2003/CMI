import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# -------------------- モデル定義 --------------------
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        out += self.shortcut(x)
        return self.relu(out)

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)  # (B, T, 1)
        return torch.sum(weights * x, dim=1)  # (B, D)

class MultimodalModel(nn.Module):
    def __init__(self, imu_dim, num_classes=18):
        super().__init__()
        self.imu_cnn = nn.Sequential(
            nn.Conv1d(imu_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            ResidualSEBlock(128, 256),
            ResidualSEBlock(256, 256)
        )
        self.tof_cnn = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.tof_gru = nn.GRU(64, 128, batch_first=True, bidirectional=True)
        self.tof_att = Attention(256)
        self.fusion_gru = nn.GRU(512, 256, batch_first=True, bidirectional=True)
        self.fusion_att = Attention(512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, imu_x, tof_x):
        x_imu = imu_x.permute(0, 2, 1)
        imu_feat = self.imu_cnn(x_imu).permute(0, 2, 1)  # (B, T, 256)
        B, T, C, H, W = tof_x.shape
        x_tof = self.tof_cnn(tof_x.view(B * T, C, H, W)).view(B, T, -1)
        x_tof, _ = self.tof_gru(x_tof)
        fused = torch.cat([imu_feat, x_tof], dim=2)
        x, _ = self.fusion_gru(fused)
        x = self.fusion_att(x)
        return self.classifier(x)

# -------------------- Dataset定義 --------------------
class GestureDataset(Dataset):
    def __init__(self, df, seq_ids, main_cols, tof_channels=5):
        self.df = df
        self.seq_ids = seq_ids
        self.main_cols = main_cols
        self.tof_channels = tof_channels
        self.tof_cols = [c for c in df.columns if c.startswith("tof_")]
        self.tof_len = len(self.tof_cols) // (tof_channels * 8 * 8)

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx):
        seq_id = self.seq_ids[idx]
        seq_df = self.df[self.df["sequence_id"] == seq_id]
        main = seq_df[self.main_cols].values.astype(np.float32)
        pad_len = max(0, MAX_LEN - len(main))
        if pad_len > 0:
            main = np.vstack([main, np.zeros((pad_len, main.shape[1]))])
        else:
            main = main[:MAX_LEN]
        tof_seq = seq_df[self.tof_cols].values.astype(np.float32)
        if pad_len > 0:
            tof_seq = np.vstack([tof_seq, np.zeros((pad_len, tof_seq.shape[1]))])
        else:
            tof_seq = tof_seq[:MAX_LEN]
        tof = tof_seq.reshape(MAX_LEN, self.tof_channels, 8, 8)
        label = gesture_map[seq_id]
        return (
            torch.tensor(main, dtype=torch.float32),
            torch.tensor(tof, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )

# -------------------- ハイパーパラメータとデータ読み込み --------------------
df = pd.read_csv("train.csv")
main_cols = [c for c in df.columns if c.startswith(("acc_", "rot_", "thm_"))]
feature_cols = main_cols + [c for c in df.columns if c.startswith("tof_")]
MAX_LEN = 100
BATCH_SIZE = 128
EPOCHS = 50
SAVE_DIR = "train_test_18class_14"
os.makedirs(SAVE_DIR, exist_ok=True)

le = LabelEncoder()
df["gesture"] = df["gesture"].fillna("unknown")
df["gesture_class"] = le.fit_transform(df["gesture"])
gesture_map = df[["sequence_id", "gesture_class"]].drop_duplicates().set_index("sequence_id")["gesture_class"].to_dict()
num_classes = len(le.classes_)

# -------------------- 学習ループ --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_seq = df["sequence_id"].unique()
subject_map = df.drop_duplicates("sequence_id").set_index("sequence_id")["subject"].to_dict()
subjects = np.array([subject_map[seq] for seq in all_seq])
kf = GroupKFold(n_splits=5)

for fold, (tr_idx, va_idx) in enumerate(kf.split(all_seq, groups=subjects)):
    print(f"\n=== Fold {fold+1} ===")
    fold_dir = os.path.join(SAVE_DIR, f"fold{fold+1}")
    os.makedirs(fold_dir, exist_ok=True)

    train_df = df[df["sequence_id"].isin(all_seq[tr_idx])].copy()
    val_df = df[df["sequence_id"].isin(all_seq[va_idx])].copy()
    for cols in [feature_cols]:
        train_df[cols] = train_df[cols].replace(-1, 0).fillna(0)
        val_df[cols] = val_df[cols].replace(-1, 0).fillna(0)

    scaler = StandardScaler()
    scaler.fit(train_df[main_cols])
    train_df[main_cols] = scaler.transform(train_df[main_cols])
    val_df[main_cols] = scaler.transform(val_df[main_cols])
    joblib.dump(scaler, os.path.join(fold_dir, "scaler.pkl"))

    train_set = GestureDataset(train_df, all_seq[tr_idx], main_cols)
    val_set = GestureDataset(val_df, all_seq[va_idx], main_cols)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = MultimodalModel(imu_dim=len(main_cols), num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for main_x, tof_x, y in train_loader:
            main_x, tof_x, y = main_x.to(device), tof_x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(main_x, tof_x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 検証
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for main_x, tof_x, y in val_loader:
                main_x, tof_x = main_x.to(device), tof_x.to(device)
                out = model(main_x, tof_x)
                pred = out.argmax(1).cpu().numpy()
                preds.extend(pred)
                labels.extend(y.numpy())

        labels = np.array(labels)
        preds = np.array(preds)
        bin_labels = (labels != 0).astype(int)
        bin_preds = (preds != 0).astype(int)
        mask = labels != 0

        acc_18 = accuracy_score(labels, preds)
        f1_18 = f1_score(labels, preds, average="macro")
        acc_bin = accuracy_score(bin_labels, bin_preds)
        f1_bin = f1_score(bin_labels, bin_preds)
        acc_9 = accuracy_score(labels[mask], preds[mask])
        f1_9 = f1_score(labels[mask], preds[mask], average="macro")
        f1_avg = (f1_bin + f1_9) / 2

        print(f"Fold {fold+1} | Epoch {epoch+1:3d} | TrainLoss: {total_loss:.4f} | "
              f"Acc18: {acc_18:.4f}, F1_18: {f1_18:.4f} | Acc_bin: {acc_bin:.4f}, "
              f"F1_bin: {f1_bin:.4f} | Acc_9: {acc_9:.4f}, F1_9: {f1_9:.4f} | F1_avg_2+9: {f1_avg:.4f}")

        if (epoch+1) % 10 == 0:
            model_path = os.path.join(fold_dir, f"model_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), model_path)
