import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------------------- データ前処理 --------------------
df = pd.read_csv("train.csv")
tof_cols = [c for c in df.columns if c.startswith("tof_")]
main_cols = [c for c in df.columns if c.startswith(("acc_", "rot_", "thm_"))]
feature_cols = main_cols + tof_cols

TOF_CHANNELS = 5
TOF_IMAGE_SIZE = 8
MAX_LEN = 100
BATCH_SIZE = 128
EPOCHS = 160

# 欠損処理とマスク
for cols in [main_cols, tof_cols]:
    df[cols] = df[cols].replace(-1, 0).fillna(0)
    df[cols] = df[cols].mask(df[cols] == 0, 1e-3)

# ✅ 標準化（全体でfit→transform）
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# ラベルエンコード
le = LabelEncoder()
df["gesture"] = df["gesture"].fillna("unknown")
df["gesture_class"] = le.fit_transform(df["gesture"])
gesture_map = df[["sequence_id", "gesture_class"]].drop_duplicates().set_index("sequence_id")["gesture_class"].to_dict()
num_classes = len(le.classes_)

# -------------------- Dataset 定義 --------------------
def pad_sequence(df_group):
    main = df_group[main_cols].values
    tof = df_group[tof_cols].values

    pad_len = max(0, MAX_LEN - len(main))
    if pad_len > 0:
        main = np.vstack([main, np.zeros((pad_len, len(main_cols)))]).astype(np.float32)
        tof = np.vstack([tof, np.zeros((pad_len, len(tof_cols)))]).astype(np.float32)
    else:
        main = main[:MAX_LEN]
        tof = tof[:MAX_LEN]

    tof = tof.reshape(MAX_LEN, TOF_CHANNELS, TOF_IMAGE_SIZE, TOF_IMAGE_SIZE)
    return main, tof

class GestureDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.seq_ids = df["sequence_id"].unique()

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx):
        seq_id = self.seq_ids[idx]
        seq_df = self.df[self.df["sequence_id"] == seq_id]
        main, tof = pad_sequence(seq_df)
        label = gesture_map[seq_id]
        return (
            torch.tensor(main, dtype=torch.float32),
            torch.tensor(tof, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )

# -------------------- モデル構造（省略せずそのまま） --------------------
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
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
            nn.Conv1d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        self.relu = nn.ReLU()

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
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(weights * x, dim=1)

class FiveMillionModel(nn.Module):
    def __init__(self, main_dim, num_classes=18):
        super().__init__()
        self.imu_branch = nn.Sequential(
            nn.Conv1d(main_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            ResidualSEBlock(128, 256),
            ResidualSEBlock(256, 256)
        )

        self.tof_cnn = nn.Sequential(
            nn.Conv2d(TOF_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
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

    def forward(self, main_x, tof_x):
        x_imu = main_x.permute(0, 2, 1)
        imu_feat = self.imu_branch(x_imu).permute(0, 2, 1)

        B, T, C, H, W = tof_x.shape
        x_tof = self.tof_cnn(tof_x.view(B * T, C, H, W)).view(B, T, -1)
        x_tof, _ = self.tof_gru(x_tof)

        fused = torch.cat([imu_feat, x_tof], dim=2)
        x, _ = self.fusion_gru(fused)
        x = self.fusion_att(x)
        return self.classifier(x)

# -------------------- 学習ループ --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = GestureDataset(df)
train_size = int(0.8 * len(train_ds))
train_set, val_set = random_split(train_ds, [train_size, len(train_ds) - train_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

model = FiveMillionModel(main_dim=len(main_cols), num_classes=num_classes).to(device)
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

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for main_x, tof_x, y in val_loader:
            out = model(main_x.to(device), tof_x.to(device))
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

    print(f"Epoch {epoch+1:3d} | Loss: {total_loss:.4f} | Acc18: {acc_18:.4f}, F1_18: {f1_18:.4f} | "
          f"Acc_bin: {acc_bin:.4f}, F1_bin: {f1_bin:.4f} | Acc_9: {acc_9:.4f}, F1_9: {f1_9:.4f} | F1_avg_2+9: {f1_avg:.4f}")

    if (epoch + 1) % 20 == 0:
        torch.save(model.state_dict(), f"gesture_model_9_epoch{epoch+1}.pt")

torch.save(model.state_dict(), "gesture_model_9_final.pt")
print("✅ モデル学習・保存完了")
