import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
import pickle

# 데이터 불러오기
with open('x_train.pkl', 'rb') as f:
    x_train = pickle.load(f)
with open('x_test.pkl', 'rb') as f:
    x_test = pickle.load(f)
with open('y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)


# PyTorch용 데이터로 변환 및 reshape
X_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)  # 1차원 데이터를 2차원으로 변환
X_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1)  # 1차원 데이터를 2차원으로 변환

# 데이터 tile 함수를 사용하여 확장
X_train_tensor = torch.tile(X_train_tensor, (1, 1, 6))  # 1차원 데이터를 2차원 네트워크에 맞게 확장
X_test_tensor = torch.tile(X_test_tensor, (1, 1, 6))  # 1차원 데이터를 2차원 네트워크에 맞게 확장

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 데이터 로더 생성
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ConvNet 클래스 정의
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 1D Convolution layer
        self.conv1d = nn.Conv1d(in_channels=6, out_channels=12, kernel_size=2)
        # 2D Convolution layer
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=2)
        # Fully connected layers
        self.fc1 = nn.Linear(264, 120)  # Change input size to 6 * 5 * 6
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # assuming 2 classes for binary classification
        # Batch normalization layer
        self.batch_norm = nn.BatchNorm1d(12)  # BatchNorm1d for 1D Conv
        # Dropout
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # ResNet-18의 특징 추출기에 데이터를 전달하여 특징 추출
     
        # 1D Convolution layer
        x = F.relu(self.conv1d(x.permute(0, 2, 1)))  # permute to make channels last for 1D Conv
        # Batch normalization
        x = self.batch_norm(x)
        # 2D Convolution layer
        x = x.unsqueeze(1)  # add channel dimension for 2D Conv
        x = F.relu(self.conv2d(x))
        # Flatten before fully connected layers
        x = x.view(-1, self.flatten_features(x))
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def flatten_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        features = 1
        for s in size:
            features *= s
        return features

# 모델 인스턴스 생성
model = ConvNet()

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습
num_epochs = 100
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 예측
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)

# 정확도 계산
accuracy = (predicted == y_test_tensor).float().mean()
print(f"Epoch [{epoch+1}/{num_epochs}], Accuracy: %.2f%%" % (accuracy * 100.0))

### save best_model 

best_accuracy = 0.0
best_model_path = 'best_model.pth'

for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    # Validation
    with torch.no_grad():
        model.eval()
        val_outputs = model(X_test_tensor)
        _, val_predicted = torch.max(val_outputs, 1)
        val_accuracy = (val_predicted == y_test_tensor).float().mean()
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print("Best model saved!")
            print("xt:", X_train_tensor.shape,"yt:", y_train_tensor.shape)
