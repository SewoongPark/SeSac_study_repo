"""

--- 목적 ---------------------

딥러닝 모델로 머신러닝의 정형 데이터 classification 성능 앞서기
--- 머신러닝과 딥러닝 모델의 차이점은 정형 데이터 분석 가능 유무에 있습니다.
--- 대부분의 연구에서 정형 데이터에 대한 prediction과 classification은 머신러닝에서 수행됩니다.  
--- 저희 팀의 프로젝트는 딥러닝 학습에서 가능한 한 많은 경우의 수를 적용해보면서 정형 데이터를 분석하고 기존 SOTA인 XGBoost 성능을 
앞지르는 Deep Neural network를 구성해보는 것이 목표입니다.

------------------------------


--- 설계 ---

데이터 셋: diabetes(당뇨) classifiaction을 위한 정형 데이터
numeric feature는 7개이고, target data는 categorical set입니다.

 1차 목표(Resnet 전이 학습):
    - 비정형 데이터 shape의 구조와 동일하게 맞추기
        - pytorch network는 input을 [batch_size, channel, width, height]로 받기 때문에 
        - (576, 6)의 shape을 갖는 정형 데이터 셋(dataframe으로부터 추출)을 차원 변환합니다. 
        - 이를 처리하기 위해 dataframe을 arrayr로 변환한 데이터에 대해 for문을 돌면서 한 행마다의 데이터 shape을 변환하는 dataloader를 구축했습니다.
        - np.tile함수를 사용하여 (1, 6)의 데이터 차원을 (1, 36)으로 복제(열 차원에 원하는 값을 곱해서 복제하는 방식)
        - 다시 한번 (1, 36)을 (32, 36)로 복제한 후(axis=0) (32, 32)로 slicing
        - 이후 np.repeat함수 사용하여 (64, 3, 32, 32)의 차원으로 만들어줍니다
            - 이때 batch_size와 channel은 임의로 정했습니다. 

        --- Channel variation ---

        -- 최초 시도 시에는 channel의 수를 1로 지정하여 gray scale의 이미지 형식으로 처리했습니다. 
        -- 예측 결과:
            experiment 
            1) 1 channel에 대해 진행: [64, 1, 32, 32]
                - Epoch 10/10, Loss: 0.3638, Accuracy: 0.6883

            2) 3 channel에 대해 진행: [64, 3, 32, 32]
                - Epoch 10/10, Loss: 0.3455, Accuracy: 0.7273

            --> 3 channel일 때 성능이 향상 됐다는 결과를 얻음.
        
    2차 목표: 성능 향상
        - 전이 학습을 위한 resnet network를 구성했습니다. 
        - Resnet 입력을 위해 Conv1d layer를 input으로 받고 처리했습니다. 
        - layer는 train=True로 설정하여 기존의 imagenet weights를 사용하도록 했습니다.
        - pretrained model의 weight를 사용하지 않는 방식을 사용해보아야 할 것 같습니다.  
        - Fully connected layer에서 input으로 들어가는 feature의 수를 512에서 1024로 늘렸습니다.
            --> 성능이 향상되었습니다. 


    3차: stretch and grid search
-       1) grid search:

        - 변경하고자 하는 parameter들은 다음과 같습니다
        
            batch_size_list = [32, 64, 128]
            learning_rate_list = [0.001, 0.01, 0.1]
            momentum_list = [0.9, 0.95, 0.99]
            dropout_rate_list = [0.1, 0.3, 0.5]
            optimizer = [Adam, SGD]
        
        - 특히 momentum은 optimizer들마다 고유한 값이 있지만, 
            이 값들에 대한 추가 변경을 통해 기존의 학습 속도 반영률을 조정하여 다양한 결과를 얻고자 했습니다. 
        
        2) Stretch:
         - 기존의 0 ~ 1 사이의 값으로 정규화한 기존 데이터를 -1 ~ 1 사이의 범위로 조정하는 stretch 기법을 사용했습니다.
         - MinMaxScaler를 취한 각각의 값에 2를 곱하고 1를 빼는 연산을 진행했습니다. 
           
    종합: 성능 향상의 요인

    - 변경 사항: grid search 수행, CosineAnnealingLearningRate scheduler 사용
    - stretch 기법 사용
    - XGboost 뛰어 넘은 parameter tuning (Grid search):
        Best accuracy: 0.8052, Best hyperparameters: 
        {'optimizer': 'Adam',
        'batch_size': 128,
        'learning_rate': 0.1,
        'momentum': 0.95,
        'dropout_rate': 0.1}

------------------------------------------------------------

    --- ConvolutionNetwork(FC layer) experiment ---

        - 기존의 Resnet보다 더 뛰어난 성능을 보였습니다.
        - grid search와 scheduler 사용은 유지한 채로 network만 변경해서 성능을 비교하고자 했습니다. 
        - train과 test loader의 shape도 동일합니다.
        - stretch된 데이터에 대해 학습을 진행했습니다.
        
        - 설계:
            - Conv1D, Conv2D, fc_layer1 + fc_layer2
            - Grid search 수행 후
            - params:
            { Optimizer: Adam,
            Batch Size: 128,
            Learning Rate: 0.001,
            Momentum: 0.95,
            Dropout Rate: 0.1,
            Epoch 10/10, 
            Loss: 0.4435,
            Accuracy: 0.8125 }

        - 특히 활성화 함수 RMSprop은 loss가 대량으로 잡혔기 때문에 early stopping 했습니다.

    --- FC layer 2차 실험 ---
     
     - 변경 사항: label벡터를 OnehotEncoding하고 array로 변환했습니다.
        - pd.get_dummies()했을 때 각 클래스마다 True, False로 할당된 값을 .astype(int)로 변경해서 0과 1의 int 값을 갖도록 변경
        - dropout layer를 제거했습니다.
     
     - SOTA 달성
        { Optimizer: Adam, Batch Size: 64, Learning Rate: 0.01, Momentum: 0.9,  Epoch 8/10, Loss: 0.4571, Accuracy: 0.8438 }

    ------------------------
"""

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
    
stretced = ["stretch", "centre", "stretch_centre" ]
    centre = [[False, ]]
    # normalize : stretch
    # stretch: [], centre: []
    
    

"""
