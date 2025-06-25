import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# -------------------- データ読み込み --------------------
df = pd.read_csv("train.csv")

main_cols = [c for c in df.columns if c.startswith(("acc_", "rot_", "thm_"))]
feature_cols = main_cols

MAX_LEN = 100
BATCH_SIZE = 128
EPOCHS = 50
SAVE_DIR = "train_test_18class_12"
os.makedirs(SAVE_DIR, exist_ok=True)

le = LabelEncoder()
df["gesture"] = df["gesture"].fillna("unknown")
df["gesture_class"] = le.fit_transform(df["gesture"])
gesture_map = df[["sequence_id", "gesture_class"]].drop_duplicates().set_index("sequence_id")["gesture_class"].to_dict()
num_classes = len(le.classes_)

# -------------------- Dataset 定義 --------------------
def pad_sequence(df_group):
    main = df_group[main_cols].values.astype(np.float32)
    pad_len = max(0, MAX_LEN - len(main))
    if pad_len > 0:
        main = np.vstack([main, np.zeros((pad_len, len(main_cols)), dtype=np.float32)])
    else:
        main = main[:MAX_LEN]
    return main

class GestureDataset(Dataset):
    def __init__(self, df, seq_ids):
        self.df = df
        self.seq_ids = seq_ids

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx):
        seq_id = self.seq_ids[idx]
        seq_df = self.df[self.df["sequence_id"] == seq_id]
        main = pad_sequence(seq_df)
        label = gesture_map[seq_id]
        return (
            torch.tensor(main, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )

# -------------------- モデル定義 --------------------
class CNNTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.transformer_encoder(x)
        w = torch.softmax(self.attn(x), dim=1)
        x = torch.sum(w * x, dim=1)
        return self.classifier(x)

# -------------------- 学習 --------------------
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

    # -------- 前処理（旧バージョン） --------
    for cols in [main_cols]:
        train_df[cols] = train_df[cols].replace(-1, 0).fillna(0)
        val_df[cols] = val_df[cols].replace(-1, 0).fillna(0)

    # -------- スケーリング（trainのみでfit） --------
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    joblib.dump(scaler, os.path.join(fold_dir, "scaler.pkl"))

    # -------- データローダー --------
    train_set = GestureDataset(train_df, all_seq[tr_idx])
    val_set = GestureDataset(val_df, all_seq[va_idx])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = CNNTransformer(input_dim=len(main_cols), num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for main_x, y in train_loader:
            main_x, y = main_x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(main_x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # -------- 検証 --------
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for main_x, y in val_loader:
                out = model(main_x.to(device))
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

        print(f"Epoch {epoch+1:3d} | TrainLoss: {total_loss:.4f} | Acc18: {acc_18:.4f}, F1_18: {f1_18:.4f} | "
              f"Acc_bin: {acc_bin:.4f}, F1_bin: {f1_bin:.4f} | Acc_9: {acc_9:.4f}, F1_9: {f1_9:.4f} | F1_avg_2+9: {f1_avg:.4f}")

        if (epoch+1) % 10 == 0:
            model_path = os.path.join(fold_dir, f"model_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), model_path)
