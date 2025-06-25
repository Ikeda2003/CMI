import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# -------------------- Data Loading --------------------
df = pd.read_csv("train.csv")

tof_cols = [col for col in df.columns if col.startswith("tof_")]
main_cols = [col for col in df.columns if col.startswith(("acc_", "rot_", "thm_"))]

# TOF構造
TOF_CHANNELS = 5
TOF_IMAGE_SIZE = 8
TOF_TOTAL_DIM = TOF_CHANNELS * TOF_IMAGE_SIZE * TOF_IMAGE_SIZE
assert len(tof_cols) == TOF_TOTAL_DIM

# ハイパーパラメータ
MAX_LEN = 100
BATCH_SIZE = 64
EPOCHS = 100

# 欠損処理
for colgroup in [main_cols, tof_cols]:
    df[colgroup] = df[colgroup].replace(-1, 0).fillna(0)
    df[colgroup] = df[colgroup].mask(df[colgroup] == 0, 1e-3)

# ラベル
le = LabelEncoder()
df["gesture"] = df["gesture"].fillna("unknown")
df["gesture_class"] = le.fit_transform(df["gesture"])
gesture_map = df[["sequence_id", "gesture_class"]].drop_duplicates().set_index("sequence_id")["gesture_class"].to_dict()
num_classes = len(le.classes_)

# -------------------- Dataset --------------------
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
        return (
            torch.tensor(main, dtype=torch.float32),
            torch.tensor(tof, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )

# -------------------- Model --------------------
class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)  # (B, T, 1)
        return torch.sum(weights * x, dim=1)           # (B, D)

class MultiModalCNNGRUWithFusion(nn.Module):
    def __init__(self, main_dim, num_classes=18):
        super().__init__()
        # main branch
        self.gru_main = nn.GRU(main_dim, 64, batch_first=True)
        self.att_main = Attention(64)

        # tof branch
        self.cnn_tof = nn.Sequential(
            nn.Conv2d(TOF_CHANNELS, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # (B*T, 32, 1, 1)
        )
        self.tof_feature_dim = 32
        self.gru_tof = nn.GRU(self.tof_feature_dim, 64, batch_first=True)
        self.att_tof = Attention(64)

        # fusion + classifier
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
        # main branch
        main_out, _ = self.gru_main(main_x)
        main_feat = self.att_main(main_out)  # (B, 64)

        # tof branch
        B, T, C, H, W = tof_x.shape
        x = tof_x.view(B * T, C, H, W)
        x = self.cnn_tof(x)
        x = x.view(B, T, self.tof_feature_dim)
        tof_out, _ = self.gru_tof(x)
        tof_feat = self.att_tof(tof_out)     # (B, 64)

        # fusion
        combined = torch.cat([main_feat, tof_feat], dim=1)  # (B, 128)
        fused = self.fusion(combined)                       # (B, 64)
        return self.classifier(fused)                       # (B, 18)

# -------------------- Training --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = GestureDataset(df)
train_size = int(0.8 * len(train_ds))
train_set, val_set = random_split(train_ds, [train_size, len(train_ds) - train_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

model = MultiModalCNNGRUWithFusion(main_dim=len(main_cols), num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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

    # Eval
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for main_x, tof_x, y in val_loader:
            out = model(main_x.to(device), tof_x.to(device))
            pred = out.argmax(1).cpu().numpy()
            preds.extend(pred)
            labels.extend(y.cpu().numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")

# モデル保存
torch.save(model.state_dict(), "multimodal_fusion_model.pt")
print("✅ モデル保存完了")

'''
Epoch 1, Loss: 239.2161, Acc: 0.2636, F1: 0.1588
Epoch 2, Loss: 189.0629, Acc: 0.3243, F1: 0.2565
Epoch 3, Loss: 164.4864, Acc: 0.3740, F1: 0.3482
Epoch 4, Loss: 151.2709, Acc: 0.4476, F1: 0.4390
Epoch 5, Loss: 141.7192, Acc: 0.4666, F1: 0.4621
Epoch 6, Loss: 133.4001, Acc: 0.4647, F1: 0.4650
Epoch 7, Loss: 129.7035, Acc: 0.5156, F1: 0.5191
Epoch 8, Loss: 124.9179, Acc: 0.4813, F1: 0.4666
Epoch 9, Loss: 122.8864, Acc: 0.4923, F1: 0.5011
Epoch 10, Loss: 116.8755, Acc: 0.5058, F1: 0.5037
Epoch 11, Loss: 113.9372, Acc: 0.5659, F1: 0.5768
Epoch 12, Loss: 110.5285, Acc: 0.5494, F1: 0.5462
Epoch 13, Loss: 106.7734, Acc: 0.5549, F1: 0.5635
Epoch 14, Loss: 105.2904, Acc: 0.5549, F1: 0.5422
Epoch 15, Loss: 102.4338, Acc: 0.5512, F1: 0.5482
Epoch 16, Loss: 99.0896, Acc: 0.5230, F1: 0.5340
Epoch 17, Loss: 97.4853, Acc: 0.5445, F1: 0.5538
Epoch 18, Loss: 93.4324, Acc: 0.5500, F1: 0.5434
Epoch 19, Loss: 92.7922, Acc: 0.5530, F1: 0.5482
Epoch 20, Loss: 90.6763, Acc: 0.6058, F1: 0.5966
Epoch 21, Loss: 87.1543, Acc: 0.5733, F1: 0.5540
Epoch 22, Loss: 87.4123, Acc: 0.5628, F1: 0.5625
Epoch 23, Loss: 86.6720, Acc: 0.6131, F1: 0.6146
Epoch 24, Loss: 82.9733, Acc: 0.5953, F1: 0.5863
Epoch 25, Loss: 82.6091, Acc: 0.5831, F1: 0.5735
Epoch 26, Loss: 78.8086, Acc: 0.6119, F1: 0.6162
Epoch 27, Loss: 79.8484, Acc: 0.5757, F1: 0.5900
Epoch 28, Loss: 76.8200, Acc: 0.5978, F1: 0.5946
Epoch 29, Loss: 75.1562, Acc: 0.5978, F1: 0.6000
Epoch 30, Loss: 73.8285, Acc: 0.5849, F1: 0.5938
Epoch 31, Loss: 73.4919, Acc: 0.6193, F1: 0.6196
Epoch 32, Loss: 71.3455, Acc: 0.6107, F1: 0.6087
Epoch 33, Loss: 68.4249, Acc: 0.6143, F1: 0.6088
Epoch 34, Loss: 68.1725, Acc: 0.6064, F1: 0.6128
Epoch 35, Loss: 64.5385, Acc: 0.6070, F1: 0.5992
Epoch 36, Loss: 63.2417, Acc: 0.5218, F1: 0.5332
Epoch 37, Loss: 65.0425, Acc: 0.5616, F1: 0.5648
Epoch 38, Loss: 64.3766, Acc: 0.5782, F1: 0.5847
'''
