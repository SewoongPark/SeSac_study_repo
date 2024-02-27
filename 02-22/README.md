## **TIL**
* Pytorch로 딥러닝 기본 복습하기
* 딥러닝에 사용되는 Dense 구성해보기,
* epoch, optimizer, gradient desecent 만들어보기

### **Optimizer**
> 딥러닝(Deep Learning)은 입력층(Input Layer)과 출력층(Output Layer) 사이에 여러 은닉층(Hidden Layer)으로 이루어진 인공 신경망입니다.<br>
층에 깊이가 깊고 복잡해질수록 Hyperparameter 또한 많아지게 됩니다. 이는 모델의 학습 속도나 성능에 직접적인 영향을 주게 되므로, 이를 잘 결정하여 모델이 원하는 결과를 낼 수 있도록 하는 것이 딥러닝 학습의 핵심입니다.

 > <br> 학습하며 얼마나 틀렸지를 Loss 라고 하며, 이에 대한 스코어를 반환하는 함수를 Loss Function이라고 합니다.<br>
딥러닝에서 학습의 목표는 이 Loss Function의 최솟값을 찾는 것입니다. 이 과정을 Optimization이라고 하며, 최적화라고 부르기도 합니다. 이는 학습을 빠르고 안정적으로 하는 것이 목표입니다.
그리고 이를 수행하는 알고리즘이 Optimizer입니다. Optimizer는 여러 종류가 있으며, 그 종류에 따라 Loss의 최저점을 찾아가는 방식이 다릅니다.

<img src = "https://miro.medium.com/v2/resize:fit:640/format:webp/1*Y2KPVGrVX9MQkeI8Yjy59Q.gif">

* 출처: https://medium.com/cdri/optimizer%EC%97%90-%EB%8C%80%ED%95%9C-%EC%A0%84%EB%B0%98%EC%A0%81%EC%9D%B8-%EC%9D%B4%ED%95%B4-633d8ec9ac1b

---

> * Optimizer는 Learning rate나 Gradient를 어떻게 할 지에 따라 종류가 다양합니다.
GD는 Gradient Descent이고, SGD는 Stochastic Gradient Descent입니다.

<img src = "https://miro.medium.com/v2/resize:fit:720/format:webp/1*YcOyk3IkjRW1LZLxiVWUnQ.png">

> ## $$w:=w−α∇J(w)$$


*   $w$는 업데이트할 매개 변수를 나타냅니다.
*   $α$는 학습률(learning rate)로, 각 단계에서 얼마나 매개 변수를 조정할지 결정합니다.
*   $α∇J(w)$는 손실 함수 $J(w)$의 gradient입니다.

> 이 갱신 규칙은 현재 위치에서의 gradient 방향으로 학습률에 따라 적절한 거리만큼 이동하여 매개 변수를 갱신합니다. 이 과정을 여러 번 반복하여 손실 함수를 최소화하는 최적의 매개 변수를 찾습니다.

> **Learning rate**는 한 번에 얼마나 학습할지, Gradient는 어떤 방향으로 학습할지를 나타냅니다. Optimizer의 차이점은 이에서 비롯되며, 이를 수정하며 발전합니다.
RMSProp, Adagrad, AdaDelta는 Learning rate를 수정한 Optimizer이고, Nag, Momentum은 Gradient를 수정하였습니다.
두 분류의 장점을 모두 가진 Optimizer는 **Adam, Nadam** 입니다.

* 읽어보기: https://wikidocs.net/152765


### **training data의 back propagation**
  * data size / batch size만큼의 epoch를 도는 동안, 한 epoch가 끝나면 오차를 계산하고 역전파 작업을 수행합니다.

### **ANN 및 퍼셉트론의 기본 이론**
>  * $AND, OR, NOT, XOR$등의 논리 게이트의 y 출력 값을 계산하기 위해
$X_n$의 값을 조절한다.
  <img src = "https://blog.kakaocdn.net/dn/c5gUSA/btqVddxpJTc/ENmE5C7wMicrOuB6BqMVlK/img.png">
  
  > * $y$값이 맞지 않는 오차가 생김 -> $w$값 재계산
  > * learning rate 값이 클 수록 이전 값과의 갱신폭 차이가 크다
  
  * 오차 계산과 가중치 업데이트: 출력 값이 실제 값과 일치하지 않을 때 오차가 발생하고, 이 오차를 최소화하기 위해 가중치를 조정합니다. 이 과정은 보통 경사 하강법을 사용하여 수행되며, 학습률(learning rate)은 이전 값과의 갱신 폭의 차이를 결정합니다.
  
  * 학습률(learning rate): 학습률은 매개 변수를 업데이트할 때의 보폭을 결정하는 하이퍼파라미터입니다. 학습률이 클수록 각 업데이트 단계에서 가중치가 크게 변화하며, 작을수록 변화가 더 부드럽게 이루어집니다. 너무 작은 학습률은 학습 속도를 늦출 수 있고, 너무 큰 학습률은 수렴을 방해할 수 있으므로 적절한 값을 선택해야 합니다.
  
  * 퍼셉트론은 하나의 선형 결정 경계를 학습하고 XOR과 같은 비선형 문제를 해결할 수 없는 반면, 다층 퍼셉트론(MLP)은 여러 개의 은닉층을 사용하여 복잡한 패턴을 학습할 수 있습니다. XOR과 같은 비선형 문제를 해결하기 위해 MLP가 사용됩니다.

**편미분 방정식**
  * $y = wx + b$의 선형 함수에서 변수($x$)가 여러개인 다변수 함수에서 각 변수와의 연산에 따른 $y$ 출력을 내적 연산($y$는 벡터 형태임)하여 최적의 해를 찾아가는 과정


---



### **과적합 해결법**

  * Consine learning rate decay
    >  learning rate scheduler로 cosine annealing을 추천합니다. cosine annealing은 초기에 학습률을 느리게 감소하고, 중간에는 선형으로 학습률을 감소하고, 마지막에는 다시 천천히 학습률을 감소합니다.
      <img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FYxRW0%2Fbtq1Xg7QvVs%2Ftfw8GcTCcgH7hUMkP1PLMK%2Fimg.png">
  
  * **Dropout (드롭아웃)**
  > 입력값의 일부를 0으로 두기 때문에 역전파시 파라미터 업데이트가 되지 않고, 이는 모형의 불확실성을 증가시켜 과적합 해결에 기여합니다.

  * **L1(Lasso)/L2(Ridge) Regularization**
  > 손실함수에 람다항을 추가해서 일종의 페널티를 주는 방법으로, 학습에 기여하지 못하는 모수를 0으로 만드는 기법입니다. <br>regularization은 가급적 출력층에 사용하는 것이 좋습니다.

  * **Batch Normalization(배치정규화)**
  > 파라미터 업데이트과정에서 0에 가까운 값이 지속적으로 곱해지면 vanishing gradient(기울기 소실) 문제가 발생합니. 이렇게 되면 파라미터 업데이트가 거의 일어나지 않고 수렴 속도도 아주 느리게 되어 최적화에 실패하게 되는데, 이 문제를 해결하는 방법으로는 배치정규화 외에도 relu 등의 활성함수를 사용하거나 가중치 초기화(weight initialization)을 적용하는 방법이 있습니다.<br>배치정규화는 mini batch 별로 분산과 표준편차를 구해 분포를 조정합니다. 역전파시 파라미터 크기에 영향을 받지 않기 때문에 좋다고 합니다.<br> 배치정규화는 각 은닉층에서 활성함수 적용 직전에 사용되어야합니다. 일반적으로 선형결합-배치정규화-활성함수-드롭아웃 순으로 은닉층 연산이 진행됩니다.

### **관심있는 모델 정리**
* Hugging-face: VisionTransformer(multi-modal)
  * https://huggingface.co/openai/clip-vit-large-patch14

* Tensorflow Image segmentation:
 * https://www.tensorflow.org/tutorials/images/segmentation?hl=ko
 * https://www.kaggle.com/models/keras/mit

* Tensorflow brain tumor classification
  * https://huggingface.co/Devarshi/Brain_Tumor_Classification
  * https://www.kaggle.com/code/paultimothymooney/mobilenetv2-with-tensorflow

### 의료 데이터 이미지 증강 문제 자료 읽어보기
> https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9431842/
  <br>의료 데이터에서는 증강에서의 한계가 있기 때문에 Autoencoder나 GAN을 사용한 생성 이미지를 활용하는 것이 가능하고,<br> K-FOLD 교차검증 기법등을 사용하여 경우의 수를 늘리는 방법들이 있음
  * GAN은 확률 분포에 의한 생성 기법이므로 원본과 정확하게 일치하지 않기 때문에 의료 데이터 셋에서는 의사나 의학 관계자의 검증을 받은뒤 사용해야 한다는 주의점이 있다.
  
### **U-NET 사용 권장**

### **Tensorflow Hub**
* TensorFlow Hub는 어디서나 미세 조정 및 배포 가능한 학습된 머신러닝 모델의 저장소입니다.
* 몇 줄의 코드만으로 BERT 및 Faster R-CNN과 같은 학습된 모델을 재사용할 수 있습니다.
* Tensorflow의 Huggingface 버전
