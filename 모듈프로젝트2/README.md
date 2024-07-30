## **Pima Indians Diabetes Classification**

### 목표: 딥러닝 모델로 머신러닝의 정형 데이터 classification 성능 앞서기
##### 가제: 다양한 특성에 따라 환자가 당뇨병에 취약한지를 분류/예측
* **보완해야할 내용: 실험에서 hyper parameter 변경 사항(활성화 함수, 정규화, optimization 변경 등)을 선택한 근거가 없음**
- 머신러닝과 딥러닝 모델의 차이점은 정형 데이터 분석 가능 유무에 있습니다.
- 대부분의 연구에서 정형 데이터에 대한 prediction과 classification은 머신러닝에서 수행됩니다.  
- 저희 팀의 프로젝트는 딥러닝 학습에서 가능한 한 많은 경우의 수를 적용해보면서 정형 데이터를 분석하고 **기존 SOTA인 XGBoost 성능을 앞지르는 Deep Neural network를 구성해보는 것이 목표입니다.**

**데이터 셋 기술**
* (Pregnancies): 임신한 횟수
* (Glucose): 경구 글루코스 내성 실험에서 2시간 후의 혈장 글루코스 농도
* (BloodPressure): 이완기 혈압 (mm Hg)
* (SkinThickness): 삼두근 피부 주름 두께 (mm)
* (Insulin): 2시간 혈청 인슐린 (mu U/ml)
* BMI: 체질량 지수 (체중(kg) / (신장(m))^2)
* (DiabetesPedigreeFunction): 당뇨병 혈통 기능
* (Age): 나이 (세)
* (Outcome): 클래스 변수 (0 또는 1), 768 중 268은 1이고 나머지는 0입니다.

---


### 설계 

* 데이터 셋: diabetes(당뇨) classifiaction을 위한 정형 데이터
  * numeric feature는 7개이고, target data는 categorical set입니다.

* **1차 목표(Resnet 전이 학습):**
    - 비정형 데이터 shape의 구조와 동일하게 맞추기
        - pytorch network는 input을 [batch_size, channel, width, height]로 받기 때문에 
        - (576, 6)의 shape을 갖는 정형 데이터 셋(dataframe으로부터 추출)을 차원 변환합니다. 
        - 이를 처리하기 위해 dataframe을 arrayr로 변환한 데이터에 대해 for문을 돌면서 한 행마다의 데이터 shape을 변환하는 dataloader를 구축했습니다.
        - np.tile함수를 사용하여 (1, 6)의 데이터 차원을 (1, 36)으로 복제(열 차원에 원하는 값을 곱해서 복제하는 방식)
        - 다시 한번 (1, 36)을 (32, 36)로 복제한 후(axis=0) (32, 32)로 slicing
        - 이후 np.repeat함수 사용하여 (64, 3, 32, 32)의 차원으로 만들어줍니다
            - 이때 batch_size와 channel은 임의로 정했습니다.
              
    - Channel variation 

        -- 최초 시도 시에는 channel의 수를 1로 지정하여 gray scale의 이미지 형식으로 처리했습니다.<br> 
        
        - **예측 결과**<br>
           - experiment 1)<br>
            - 1 channel에 대해 진행: **[64, 1, 32, 32]**
             - `Epoch 10/10, Loss: 0.3638, Accuracy: 0.6883`
           - experiment 2) <br>
            -  3 channel에 대해 진행: **[64, 3, 32, 32]**
                - `Epoch 10/10, Loss: 0.3455, Accuracy: 0.7273`

            ### 3 channel일 때 성능이 향상 됐다는 결과를 얻음.
        
* **2차 목표: 성능 향상**
    - 전이 학습을 위한 resnet network를 구성했습니다. 
    - Resnet 입력을 위해 Conv1d layer를 input으로 받고 처리했습니다. 
    - layer는 `train=True`로 설정하여 기존의 imagenet weights를 사용하도록 했습니다.
    - pretrained model의 weight를 사용하지 않는 방식을 사용해보아야 할 것 같습니다.  
    - Fully connected layer에서 input으로 들어가는 feature의 수를 512에서 1024로 늘렸습니다.
      <br>  --> 성능이 향상되었습니다. 


* **3차 목표: stretch and grid search**
-       1) grid search:

        - 변경하고자 하는 parameter들은 다음과 같습니다<br>
          ```
          batch_size_list = [32, 64, 128]
          learning_rate_list = [0.001, 0.01, 0.1]
          momentum_list = [0.9, 0.95, 0.99]
          dropout_rate_list = [0.1, 0.3, 0.5]
          optimizer = [Adam, SGD]
          ```

        - 특히 momentum은 optimizer들마다 고유한 값이 있지만, <br>
            이 값들에 대한 추가 변경을 통해 기존의 학습 속도 반영률을 조정하여 다양한 결과를 얻고자 했습니다. 
        
        2) Stretch:
         - 기존의 0 ~ 1 사이의 값으로 정규화한 기존 데이터를 -1 ~ 1 사이의 범위로 조정하는 stretch 기법을 사용했습니다.
         - MinMaxScaler를 취한 각각의 값에 2를 곱하고 1를 빼는 연산을 진행했습니다. 
           
    종합: 성능 향상의 요인

    - 변경 사항: grid search 수행, CosineAnnealingLearningRate scheduler 사용
    - stretch 기법 사용
    - XGboost 뛰어 넘은 parameter tuning (Grid search):
        ```
        Best accuracy: 0.8052, Best hyperparameters: 
        {'optimizer': 'Adam',
        'batch_size': 128,
        'learning_rate': 0.1,
        'momentum': 0.95,
        'dropout_rate': 0.1}

------------------------------------------------------------

### Convolution Network(FC layer) experiment

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

--- 

* **FC layer 2차 실험**
     
     - 변경 사항: label벡터를 OnehotEncoding하고 array로 변환했습니다.
        - pd.get_dummies()했을 때 각 클래스마다 True, False로 할당된 값을 .astype(int)로 변경해서 0과 1의 int 값을 갖도록 변경
        - dropout layer를 제거했습니다.
     
- **SOTA 달성**
        ```
        { Optimizer: Adam,
          Batch Size: 64,
          Learning Rate: 0.01,
          Momentum: 0.9,
          Epoch 8/10,
          Loss: 0.4571,
          Accuracy: 0.8438 }

---


### 설계 

* 데이터 셋: diabetes(당뇨) classifiaction을 위한 정형 데이터
  * numeric feature는 7개이고, target data는 categorical set입니다.

* **1차 목표(Resnet 전이 학습):**
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
        ``` 
        Best accuracy: 0.8052, Best hyperparameters: 
        {'optimizer': 'Adam',
        'batch_size': 128,
        'learning_rate': 0.1,
        'momentum': 0.95,
        'dropout_rate': 0.1}

------------------------------------------------------------

* **ConvolutionNetwork(FC layer) experiment**
 - 기존의 Resnet보다 더 뛰어난 성능을 보였습니다.
 - grid search와 scheduler 사용은 유지한 채로 network만 변경해서 성능을 비교하고자 했습니다. 
 - train과 test loader의 shape도 동일합니다.
 -  stretch된 데이터에 대해 학습을 진행했습니다.

- 설계:
  - Conv1D, Conv2D, fc_layer1 + fc_layer2
  - Grid search 수행 후
```
params:
            { Optimizer: Adam,
            Batch Size: 128,
            Learning Rate: 0.001,
            Momentum: 0.95,
            Dropout Rate: 0.1,
            Epoch 10/10, 
            Loss: 0.4435,
            Accuracy: 0.8125 }
```
    - 특히 활성화 함수 RMSprop은 loss가 대량으로 잡혔기 때문에 early stopping 했습니다.
    
- **FC layer 2차 실험**
     
     - 변경 사항: label벡터를 OnehotEncoding하고 array로 변환했습니다.
        - `pd.get_dummies()`했을 때 각 클래스마다 True, False로 할당된 값을 `.astype(int)`로 변경해서 0과 1의 int 값을 갖도록 변경
        - dropout layer를 제거했습니다.


- **SOTA 달성**
```
        { Optimizer: Adam,
 Batch Size: 64,
 Learning Rate: 0.01,
 Momentum: 0.9,
  Epoch 8/10,
 Loss: 0.4571,
 Accuracy: 0.8438 }

    ------------------------
"""
