import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 1. 数据增强与预处理 ----------
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


class CIFAR10Dataset(Dataset):
    def __init__(self, data_path, label_path):
        df = pd.read_csv(label_path)
        # 预加载原始 PIL 图像到内存（不带任何变换，仅加载一次，节省内存和加载时间）
        self.images = [
            Image.open(os.path.join(data_path, f"{i}.png")).convert('RGB') 
            for i in df["id"]
        ]
        self.labels = df["label"].astype("category").cat.codes.tolist()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 仅返回原始图像和标签，具体的 Transforms 留到切分后应用
        return self.images[idx], self.labels[idx]


class DatasetWithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------- 3. 网络结构 ----------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 确保 padding=1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 确保 padding=1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 此时两边尺寸都是 32x32（或相同缩放尺寸），完美相加
        out = self.relu(out)
        return out


class CIFARResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # ⚠️ 检查这里：确保第一个卷积层也带 padding=1 保持 32x32 尺寸
        self.prep = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 后续的残差层
        self.layer1 = ResidualBlock(32, 32, stride=1)   # 32x32 -> 32x32
        self.layer2 = ResidualBlock(32, 64, stride=2)   # 32x32 -> 16x16
        self.layer3 = ResidualBlock(64, 128, stride=2)  # 16x16 -> 8x8
        self.layer4 = ResidualBlock(128, 256, stride=2) # 8x8 -> 4x4
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def main():
    print("Using Device:", device)
    

    data_path = r"D:\vscode\kaggle\CIFAR-10\train"
    label_path = r"D:\vscode\kaggle\CIFAR-10\trainLabels.csv"

    print("Loading dataset into memory...")
    full_dataset = CIFAR10Dataset(data_path, label_path)
    print(f"Dataset loaded. Total images: {len(full_dataset)}")

    # 严格切分训练集与验证集索引
    g = torch.Generator().manual_seed(42)
    train_split, val_split = random_split(full_dataset, [45000, 5000], generator=g)

    # 3. 分别绑定训练集增强和验证集转换
    train_ds = DatasetWithTransform(train_split, train_transform)
    val_ds = DatasetWithTransform(val_split, val_transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = CIFARResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    EPOCH = 20
    print("Start training...")
    for epoch in range(EPOCH):
        start_time = time.time()

        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)

        train_loss = total_loss / len(train_ds)

        # ---------- 验证阶段 ----------
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images).argmax(dim=1)
                correct += (pred == labels).sum().item()

        val_acc = 100 * correct / len(val_ds)
        

        scheduler.step()

        # 打印日志
        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1:02d}/{EPOCH:02d}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.2f}s")


if __name__ == "__main__":
    main()