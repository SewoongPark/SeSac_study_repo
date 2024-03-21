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
import multiprocessing
from torch.optim.lr_scheduler import CosineAnnealingLR


class ConvNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ConvNetClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(
            64 * 8 * 8, 512
        )  # 이미지 크기가 32x32이므로 최종 feature map 크기는 8x8입니다.
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvNetTrainer:
    def __init__(
        self,
        num_epochs=10,
        log_dir="./logs_03",
        best_model_path="./models/ConvNet/stretch/best_model_cnn.pth",
    ):
        self.num_epochs = num_epochs
        self.log_dir = log_dir
        self.best_model_path = best_model_path

    def train(
        self, model, train_loader, test_loader, optimizer, criterion, scheduler, device
    ):
        best_accuracy = 0.0

        for epoch in range(self.num_epochs):
            model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            scheduler.step()
            epoch_loss = running_loss / len(train_loader.dataset)

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.argmax(dim=1)).sum().item()

            accuracy = correct / total

            print(
                f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}"
            )

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), self.best_model_path)

        print(
            f"Best accuracy: {best_accuracy:.4f}, Best model saved to {self.best_model_path}"
        )

    def run(
        self, model, train_loader, test_loader, optimizer, criterion, scheduler, device
    ):
        os.makedirs(self.log_dir, exist_ok=True)
        writer = SummaryWriter(self.log_dir)

        self.train(
            model, train_loader, test_loader, optimizer, criterion, scheduler, device
        )

        writer.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()

    torch.manual_seed(42)
    np.random.seed(42)

    pathFolder = "./data/stretched_data_diabetes/"  # train, test pickle 파일의 경로
    os.makedirs(pathFolder, exist_ok=True)

    xTrainName = (
        "xTrain_noStretch.pkl"  # normalize 코드 추가 위해 non-stretched 파일 load
    )
    yTrainName = "yTrain_onehot.pkl"

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

    X_trainList = np.array([tile_transform(xrow) for xrow in np.array(X_train)])
    X_testList = np.array([tile_transform(xrow) for xrow in np.array(X_test)])

    y_train = torch.tensor(y_train, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)

    train_dataset = TensorDataset(
        torch.Tensor(X_trainList[:, None, :, :].repeat(3, 1)), y_train
    )
    test_dataset = TensorDataset(
        torch.Tensor(X_testList[:, None, :, :].repeat(3, 1)), y_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(np.unique(y_train))
    model = ConvNetClassifier(num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)

    trainer = ConvNetTrainer()
    trainer.run(
        model, train_loader, test_loader, optimizer, criterion, scheduler, device
    )

"""
최초 시도: Epoch 10/10, Loss: 0.4412, Accuracy: 0.7969

--- superior params: ----

    Optimizer: Adam, Batch Size: 64, Learning Rate: 0.001, Momentum: 0.9, Dropout Rate: 0.5, Epoch 7/10, Loss: 0.4500, Accuracy: 0.8125
    Optimizer: Adam, Batch Size: 64, Learning Rate: 0.01, Momentum: 0.95, Dropout Rate: 0.3, Epoch 5/10, Loss: 0.5145, Accuracy: 0.8125
    Optimizer: Adam, Batch Size: 128, Learning Rate: 0.001, Momentum: 0.95, Dropout Rate: 0.1, Epoch 10/10, Loss: 0.4435, Accuracy: 0.8125
    Optimizer: Adam, Batch Size: 64, Learning Rate: 0.01, Momentum: 0.95, Dropout Rate: 0.1, Epoch 5/10, Loss: 0.5242, Accuracy: 0.8281
    Optimizer: Adam, Batch Size: 64, Learning Rate: 0.01, Momentum: 0.9,  Epoch 8/10, Loss: 0.4571, Accuracy: 0.8438
------------------------

SOTA)
Best accuracy: 0.8438,
Best hyperparameters:
    {'optimizer': 'Adam',
    'batch_size': 64,
    'learning_rate': 0.01,
    'momentum': 0.9,
    }

"""
