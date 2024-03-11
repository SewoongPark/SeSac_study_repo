import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import pickle
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# 시드 고정
torch.manual_seed(42)
np.random.seed(42)

pathFolder = "./data/"
os.makedirs(pathFolder, exist_ok=True)

xTrainName = "x_train.pkl"
yTrainName = "y_train.pkl"

with open(pathFolder + xTrainName, "rb") as f1:
    X = pickle.load(f1)

with open(pathFolder + yTrainName, "rb") as f2:
    y = pickle.load(f2)

image_size = 32
batch_size = 64

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)


def tile_transform(x):
    x = np.tile(x, (32, 6))[:, :-4]
    return x


# 정형 데이터 -> 이미지 차원과 동일하게 변환
X_trainList = np.array([tile_transform(xrow) for xrow in np.array(X_train)])
X_testList = np.array([tile_transform(xrow) for xrow in np.array(X_test)])

y_train = np.array(y_train)
y_test = np.array(y_test)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


train_dataset = TensorDataset(
    torch.Tensor(X_trainList[:, None, :, :].repeat(3, 1)), y_train
)
test_dataset = TensorDataset(
    torch.Tensor(X_testList[:, None, :, :].repeat(3, 1)), y_test
)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

log_dir = "./logs"  # Specify the directory for TensorBoard logs
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        num_classes = len(np.unique(y_train))
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc1 = nn.Linear(num_ftrs * 2, num_classes)
        # self.resnet.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = F.relu(x)

        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(np.unique(y_train))  # 데이터셋의 클래스 수
resnet_model = ResNetClassifier(num_classes)
resnet_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)
num_epochs = 10
best_accuracy = 0.0
best_model_path = "../models"

### 학습
for epoch in range(num_epochs):
    resnet_model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = resnet_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

        # Logging training loss to TensorBoard
        writer.add_scalar("Training Loss", loss.item(), epoch * len(train_loader) + i)

    epoch_loss = running_loss / len(train_loader.dataset)

    # 검증 데이터셋을 사용하여 정확도 계산
    resnet_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = resnet_model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    # Logging accuracy to TensorBoard
    writer.add_scalar("Accuracy", accuracy, epoch)

    print(
        f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}"
    )

    # 현재 모델의 정확도가 최고 정확도보다 높으면 모델을 저장
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(resnet_model.state_dict(), best_model_path)

print(f"Best accuracy: {best_accuracy:.4f}, Best model saved to {best_model_path}")

# Close the writer after training
writer.close()

"""
num_ftr(feature)수를 2배로 (512 -> 1024)로 늘렸더니 sota 찍음

Epoch 5/10,
Loss: 0.6187,
Accuracy: 0.7922
"""
