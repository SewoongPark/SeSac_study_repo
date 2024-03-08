import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
import pickle

# 데이터 로드
with open('x_train.pkl', 'rb') as f:
    x_train = pickle.load(f)
with open('x_test.pkl', 'rb') as f:
    x_test = pickle.load(f)
with open('y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

# 데이터를 Tensor로 변환
X_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(0) # 3차원으로 변경
X_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(0)  # 3차원으로 변경

# 데이터를 3채널로 확장
X_train_tensor = X_train_tensor.repeat(3, 3, 1, 1)  # [3, 3, 576, 6]로 확장
X_test_tensor = X_test_tensor.repeat(3, 3, 1, 1) 

# y_train, y_test를 3채널로 확장
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(0).unsqueeze(2)

y_train_tensor = y_train_tensor.repeat(3, 3, 1, 1)  # [3, 3, 576, 1]로 확장
y_test_tensor = y_test_tensor.repeat(3, 3, 1, 1) 

# 데이터 로더 생성
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ConvNet 클래스 정의
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # ResNet-18의 특징 추출기 부분만 사용
        self.resnet = models.resnet18(pretrained=True)
        # ResNet-18의 fully connected layer를 수정
        self.resnet.fc = nn.Linear(512, 2)  # 출력 클래스가 2인 경우

    def forward(self, x):
        # ResNet-18의 특징 추출기에 데이터를 전달하여 특징 추출
        features = self.resnet(x)
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
        loss = criterion(outputs, labels.squeeze(dim=-1).long())  # 오류 수정
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 예측
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)

# 정확도 계산
accuracy = (predicted == y_test_tensor.squeeze().long()).float().mean()
print(f"Accuracy: {accuracy.item() * 100:.2f}%")
