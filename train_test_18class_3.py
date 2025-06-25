import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# データ読み込み
df = pd.read_csv("train.csv")

# 特徴量抽出（TOFを除く）
feature_cols = [col for col in df.columns if col.startswith(('acc_', 'rot_', 'thm_'))]
df[feature_cols] = df[feature_cols].replace(-1, 0).fillna(0)
df[feature_cols] = df[feature_cols].mask(df[feature_cols] == 0, 1e-3)

# gesture を18クラスとしてエンコード
df['gesture'] = df['gesture'].fillna("unknown")
le = LabelEncoder()
df['gesture_class'] = le.fit_transform(df['gesture'])
gesture_map = df[['sequence_id', 'gesture_class']].drop_duplicates().set_index('sequence_id')['gesture_class'].to_dict()
num_classes = len(le.classes_)
print("Number of gesture classes:", num_classes)
print("Classes:", list(le.classes_))

# パディング設定
MAX_LEN = 100
FEATURE_DIM = len(feature_cols)

def pad_sequence(seq_df):
    seq = seq_df[feature_cols].values
    if len(seq) >= MAX_LEN:
        return seq[:MAX_LEN]
    else:
        pad = np.zeros((MAX_LEN - len(seq), FEATURE_DIM))
        return np.vstack([seq, pad])

# Dataset定義
class GestureSequenceDataset(Dataset):
    def __init__(self, df):
        self.sequence_ids = df['sequence_id'].unique()
        self.df = df

    def __len__(self):
        return len(self.sequence_ids)

    def __getitem__(self, idx):
        seq_id = self.sequence_ids[idx]
        seq_df = self.df[self.df['sequence_id'] == seq_id]
        x = pad_sequence(seq_df)
        y = gesture_map[seq_id]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# GRUモデル
class GRUGestureClassifier(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=64, output_dim=num_classes):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# データローダー構築
dataset = GestureSequenceDataset(df)
train_size = int(0.8 * len(dataset))
train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# モデル・損失関数・最適化
model = GRUGestureClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 学習ループ
for epoch in range(100):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    # 評価
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    # 2値分類スコア（0〜7:ターゲット, 8〜17:非ターゲット）
    target_classes = set(range(8))
    binary_labels = [1 if y in target_classes else 0 for y in all_labels]
    binary_preds  = [1 if y in target_classes else 0 for y in all_preds]
    binary_acc = accuracy_score(binary_labels, binary_preds)
    binary_f1  = f1_score(binary_labels, binary_preds)

    # 9クラススコア（非ターゲットはすべてクラス8へ）
    def to_9class(y): return y if y in target_classes else 8
    y9_labels = list(map(to_9class, all_labels))
    y9_preds  = list(map(to_9class, all_preds))
    acc_9 = accuracy_score(y9_labels, y9_preds)
    f1_9 = f1_score(y9_labels, y9_preds, average='macro')

    print(f"Epoch {epoch+1}, "
          f"Loss: {total_loss / len(train_loader):.4f}, "
          f"Accuracy: {acc:.4f}, F1: {f1:.4f}, "
          f"Binary Accuracy: {binary_acc:.4f}, Binary F1: {binary_f1:.4f}, "
          f"9-Class Accuracy: {acc_9:.4f}, 9-Class F1: {f1_9:.4f}")

# モデル保存
torch.save(model.state_dict(), "gesture_gru_classifier_noTOF.pt")
print("✅ モデルを gesture_gru_classifier_noTOF.pt に保存しました")
