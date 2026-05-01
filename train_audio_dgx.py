import os
import subprocess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
from tqdm import tqdm
import json
import random
from sklearn.metrics import accuracy_score

CONFIG = {
    'data_root'  : '/workspace/deepguard/datasets/audio_dataset'
                   '/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2',
    'save_dir'   : '/workspace/deepguard/models/weights',
    'sample_rate': 16000,
    'duration'   : 5,
    'n_mels'     : 128,
    'batch_size' : 32,
    'epochs'     : 15,
    'lr'         : 0.001,
    'num_workers': 0,
    'max_files'  : 5000,
}

REAL_FOLDERS = ['RealVideo-RealAudio', 'FakeVideo-RealAudio']
FAKE_FOLDERS = ['FakeVideo-FakeAudio', 'RealVideo-FakeAudio']

def extract_audio(mp4_path, wav_path):
    cmd = ['ffmpeg', '-i', mp4_path, '-vn',
           '-acodec', 'pcm_s16le',
           '-ar', str(CONFIG['sample_rate']),
           '-ac', '1', wav_path, '-y', '-loglevel', 'error']
    subprocess.run(cmd, check=True)

def collect_files(folders, label):
    files = []
    for folder in folders:
        folder_path = os.path.join(CONFIG['data_root'], folder)
        for root, dirs, filenames in os.walk(folder_path):
            for f in filenames:
                if f.endswith('.mp4'):
                    files.append(
                        (os.path.join(root, f), label))
    random.shuffle(files)
    return files[:CONFIG['max_files']]

class AudioDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.sr = CONFIG['sample_rate']
        self.duration = CONFIG['duration']
        self.n_mels = CONFIG['n_mels']
        self.samples = self.sr * self.duration

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        mp4_path, label = self.file_list[idx]
        wav_path = f'/tmp/audio_{idx}.wav'
        try:
            extract_audio(mp4_path, wav_path)
            y, sr = librosa.load(
                wav_path, sr=self.sr, duration=self.duration)
            if len(y) < self.samples:
                y = np.pad(y, (0, self.samples - len(y)))
            else:
                y = y[:self.samples]
            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=self.n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - mel_db.mean()) / (
                mel_db.std() + 1e-8)
            mel_tensor = torch.FloatTensor(mel_db).unsqueeze(0)
            if os.path.exists(wav_path):
                os.remove(wav_path)
            return mel_tensor, torch.FloatTensor([label])
        except Exception as e:
            if os.path.exists(wav_path):
                os.remove(wav_path)
            return torch.zeros(1, self.n_mels, 157), \
                   torch.FloatTensor([label])

class AudioDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(self.cnn(x)).squeeze(1)

def train():
    print("="*60)
    print("  DEEPGUARD AUDIO TRAINING ON NVIDIA DGX B200")
    print("="*60)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    real_files = collect_files(REAL_FOLDERS, 0)
    fake_files = collect_files(FAKE_FOLDERS, 1)
    print(f"Real: {len(real_files)} | Fake: {len(fake_files)}")
    all_files = real_files + fake_files
    random.shuffle(all_files)
    split = int(0.85 * len(all_files))
    train_files = all_files[:split]
    val_files = all_files[split:]
    train_loader = DataLoader(
        AudioDataset(train_files),
        batch_size=CONFIG['batch_size'],
        shuffle=True, num_workers=0)
    val_loader = DataLoader(
        AudioDataset(val_files),
        batch_size=CONFIG['batch_size'],
        shuffle=False, num_workers=0)
    model = AudioDeepfakeDetector().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CONFIG['lr'])
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5)
    best_val_acc = 0
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_preds, train_labels = [], []
        pbar = tqdm(train_loader,
            desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]")
        for mels, labels in pbar:
            mels = mels.to(device)
            labels = labels.squeeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(mels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_preds.extend(
                (outputs > 0.5).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for mels, labels in tqdm(val_loader,
                desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]"):
                mels = mels.to(device)
                labels = labels.squeeze(1).to(device)
                outputs = model(mels)
                val_preds.extend(
                    (outputs > 0.5).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_acc = accuracy_score(val_labels, val_preds) * 100
        train_acc = accuracy_score(
            train_labels, train_preds) * 100
        print(f"\nEpoch [{epoch+1}/{CONFIG['epochs']}]")
        print(f"  Train Acc: {train_acc:.2f}%")
        print(f"  Val   Acc: {val_acc:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                os.path.join(CONFIG['save_dir'],
                             'best_audio_model.pth'))
            print(f"  Best model saved! Val Acc: {val_acc:.2f}%")
        scheduler.step()
    print(f"\nTraining complete! Best Val Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train()