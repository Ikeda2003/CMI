import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, fbeta_score
from sklearn.preprocessing import LabelEncoder

# ---------- ハイパーパラメータ ----------
TOF_CHANNELS = 5
TOF_IMAGE_SIZE = 8
MAX_LEN = 100
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3

# ---------- データ読み込み ----------
df = pd.read_csv("train.csv")
tof_cols = [c for c in df.columns if c.startswith("tof_")]
main_cols = [c for c in df.columns if c.startswith(("acc_", "rot_", "thm_"))]
TOF_TOTAL_DIM = TOF_CHANNELS * TOF_IMAGE_SIZE * TOF_IMAGE_SIZE
assert len(tof_cols) == TOF_TOTAL_DIM

# 欠損処理
for colgroup in [main_cols, tof_cols]:
    df[colgroup] = df[colgroup].replace(-1, 0).fillna(0)
    df[colgroup] = df[colgroup].mask(df[colgroup] == 0, 1e-3)

# ラベルエンコード
le = LabelEncoder()
df["gesture"] = df["gesture"].fillna("unknown")
df["gesture_class"] = le.fit_transform(df["gesture"])
gesture_map = df[["sequence_id", "gesture_class"]].drop_duplicates().set_index("sequence_id")["gesture_class"].to_dict()
num_classes = len(le.classes_)

# ---------- データセット定義 ----------
def pad_sequence(df_group):
    main = df_group[main_cols].values
    tof = df_group[tof_cols].values

    pad_len = max(0, MAX_LEN - len(main))
    if pad_len > 0:
        main = np.vstack([main, np.zeros((pad_len, len(main_cols)))])
        tof = np.vstack([tof, np.zeros((pad_len, len(tof_cols)))])
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
        return torch.tensor(main, dtype=torch.float32), torch.tensor(tof, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ---------- モデル定義 ----------
class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)
    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(weights * x, dim=1)

class MultiModalCNNGRUWithFusion(nn.Module):
    def __init__(self, main_dim, num_classes=18):
        super().__init__()
        self.gru_main = nn.GRU(main_dim, 64, batch_first=True)
        self.att_main = Attention(64)

        self.cnn_tof = nn.Sequential(
            nn.Conv2d(TOF_CHANNELS, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.tof_feature_dim = 32
        self.gru_tof = nn.GRU(self.tof_feature_dim, 64, batch_first=True)
        self.att_tof = Attention(64)

        self.fusion = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, main_x, tof_x):
        main_out, _ = self.gru_main(main_x)
        main_feat = self.att_main(main_out)
        B, T, C, H, W = tof_x.shape
        x = self.cnn_tof(tof_x.view(B * T, C, H, W))
        x = x.view(B, T, -1)
        tof_out, _ = self.gru_tof(x)
        tof_feat = self.att_tof(tof_out)
        combined = torch.cat([main_feat, tof_feat], dim=1)
        fused = self.fusion(combined)
        return self.classifier(fused)

# ---------- 評価関数 ----------
def compute_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f2_macro = fbeta_score(labels, preds, beta=2, average="macro")

    # binary: 0 vs other
    binary_preds = [0 if p == 0 else 1 for p in preds]
    binary_labels = [0 if l == 0 else 1 for l in labels]
    f2_binary = fbeta_score(binary_labels, binary_preds, beta=2)

    # 9-class: gesture 0 vs other 8 classes
    def to_9class(x): return 0 if x == 0 else (x - 1) % 8 + 1
    nine_preds = [to_9class(p) for p in preds]
    nine_labels = [to_9class(l) for l in labels]
    f2_nine = fbeta_score(nine_labels, nine_preds, beta=2, average="macro")

    return acc, f1_macro, f2_macro, f2_binary, f2_nine, (f2_binary + f2_nine) / 2

# ---------- 学習 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = GestureDataset(df)
train_size = int(0.8 * len(train_ds))
train_set, val_set = random_split(train_ds, [train_size, len(train_ds) - train_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

model = MultiModalCNNGRUWithFusion(main_dim=len(main_cols), num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

    # 評価
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for main_x, tof_x, y in val_loader:
            out = model(main_x.to(device), tof_x.to(device))
            pred = out.argmax(1).cpu().numpy()
            preds.extend(pred)
            labels.extend(y.cpu().numpy())

    acc, f1, f2, f2_bin, f2_9, f2_avg = compute_metrics(preds, labels)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}, "
          f"F2: {f2:.4f}, F2_bin: {f2_bin:.4f}, F2_9: {f2_9:.4f}, F2_avg_9+2: {f2_avg:.4f}")

# モデル保存
torch.save(model.state_dict(), "multimodal_fusion_model.pt")
print("✅ モデル保存完了")
