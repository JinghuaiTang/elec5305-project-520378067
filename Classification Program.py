import os
import math
import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# -----------------------
# Configurations
# -----------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "UrbanSound8K"                             # Change if needed
CSV_PATH  = os.path.join(DATA_ROOT, "metadata", "UrbanSound8K.csv")
AUDIO_DIR = os.path.join(DATA_ROOT, "audio")

# Audio/Feature Params
SAMPLE_RATE = 22050
TARGET_DURATION = 4.0                                   # Seconds
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_DURATION)
N_MELS = 64
HOP_LENGTH = 512
N_FFT = 1024

# Training Params
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3

# Folds Split (UrbanSound8K has folds 1-10)
TRAIN_FOLDS = set([1,2,3,4,5,6,7,8])
VAL_FOLDS   = set([9])
TEST_FOLDS  = set([10])

# -----------------------
# Utility Functions
# -----------------------
def load_audio_fixed(path, sr=SAMPLE_RATE, target_len=TARGET_SAMPLES):
    """Load audio, resample, and pad/trim to fixed length."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    if len(y) < target_len:
        pad_len = target_len - len(y)
        y = np.pad(y, (0, pad_len), mode="constant")
    else:
        y = y[:target_len]
    return y

def to_log_mel(y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Convert waveform to Log-Mel spectrogram."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)  # (n_mels, time)

def simple_augment(y, noise_scale=(0.0, 0.02), shift_max=0.2):
    """Apply lightweight augmentation: time shift and additive noise."""
    # Time shift
    shift = int(random.uniform(-shift_max, shift_max) * len(y))
    y = np.roll(y, shift)
    # Add noise
    noise = np.random.randn(len(y)) * random.uniform(*noise_scale)
    y = y + noise
    return y

# -----------------------
# Dataset
# -----------------------
class UrbanSoundDataset(Dataset):
    """UrbanSound8K Dataset -> Log-Mel Features"""
    def __init__(self, df, audio_dir=AUDIO_DIR, sr=SAMPLE_RATE, train=False):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.sr = sr
        self.train = train
        self.class_names = sorted(self.df["class"].unique().tolist())
        self.class_to_idx = {c:i for i,c in enumerate(self.class_names)}
        self.idx_to_class = {i:c for c,i in self.class_to_idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fold = row["fold"]
        fname = row["slice_file_name"]
        label_name = row["class"]
        label = self.class_to_idx[label_name]

        wav_path = os.path.join(self.audio_dir, f"fold{fold}", fname)
        y = load_audio_fixed(wav_path, sr=self.sr, target_len=TARGET_SAMPLES)

        if self.train:
            y = simple_augment(y)

        feat = to_log_mel(y, sr=self.sr)  # (n_mels, time)
        # Normalize per-sample
        mu, sigma = feat.mean(), feat.std() + 1e-9
        feat = (feat - mu) / sigma

        # Add channel dimension: (1, n_mels, time)
        feat = np.expand_dims(feat, axis=0)
        feat = torch.tensor(feat, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return feat, label

# -----------------------
# Model (Simple CNN)
# -----------------------
class SimpleAudioCNN(nn.Module):
    """A Small CNN For Log-Mel Based Audio Classification"""
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2,2)
        self.drop  = nn.Dropout(0.3)
        self.gap   = nn.AdaptiveAvgPool2d((1,1))
        self.fc    = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (B,16,*,*)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (B,32,*,*)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # (B,64,*,*)
        x = self.drop(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)         # (B,64)
        x = self.fc(x)                                   # (B,C)
        return x

# -----------------------
# Data Loading
# -----------------------
meta = pd.read_csv(CSV_PATH)
train_df = meta[meta["fold"].isin(TRAIN_FOLDS)].copy()
val_df   = meta[meta["fold"].isin(VAL_FOLDS)].copy()
test_df  = meta[meta["fold"].isin(TEST_FOLDS)].copy()

train_ds = UrbanSoundDataset(train_df, train=True)
val_ds   = UrbanSoundDataset(val_df, train=False)
test_ds  = UrbanSoundDataset(test_df, train=False)

n_classes = len(train_ds.class_names)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# -----------------------
# Training Utilities
# -----------------------
def train_one_epoch(model, loader, optimizer, criterion):
    """Train For One Epoch."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    """Evaluate On Validation/Test Loader."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())
    acc = correct / total if total > 0 else 0.0
    return total_loss / total, acc, np.array(all_preds), np.array(all_labels)

# -----------------------
# Model Init
# -----------------------
model = SimpleAudioCNN(n_classes=n_classes).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# -----------------------
# Training Loop
# -----------------------
best_val_acc = 0.0
for epoch in range(1, EPOCHS+1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
    print(f"Epoch {epoch:02d} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved Best Model.")

# -----------------------
# Final Test Evaluation
# -----------------------
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion)
print(f"Test Loss {test_loss:.4f} | Test Acc {test_acc:.4f}")

# -----------------------
# Reports
# -----------------------
idx_to_class = train_ds.idx_to_class
target_names = [idx_to_class[i] for i in range(n_classes)]
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("results_confusion_matrix.png", dpi=200)
print("Saved confusion matrix to results_confusion_matrix.png")