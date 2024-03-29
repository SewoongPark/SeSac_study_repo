## **TIL**

### **전이 학습(Transfer Learning)**
* 전이 학습은 다양한 사전 훈련된 모델이 있는 작업에 유용합니다. 예를 들어, 널리 사용되는 많은 CNN(컨벌루션 신경망)은 1,400만 개 이상의 영상과 수천 개의 영상 클래스를 포함하는 ImageNet 데이터셋에 대해 사전 훈련되었습니다. <br>만약 여러분이 정원의 꽃 영상(또는 ImageNet 데이터셋에 포함되지 않은 영상)을 분류해야 하는데 꽃 영상 수가 제한적인 경우, SqueezeNet 신경망에서 계층과 가중치를 전이하고 최종 계층을 교체한 후 보유한 영상으로 모델을 재훈련시킬 수 있습니다.<br>
이 접근법으로 여러분은 전이 학습을 통해 더 짧은 시간에 더 높은 모델 정확도를 달성할 수 있습니다.
<img src = "https://kr.mathworks.com/discovery/transfer-learning/_jcr_content/mainParsys/band/mainParsys/lockedsubnav/mainParsys/columns_1336656927/a8810b41-e0c1-4c16-91f9-10b6a66488fe/image_copy_1488534355.adapt.full.medium.gif/1705253487834.gif" style="width: 50%;">

* **모델 종류: Resnet 50**
> * ResNet-50은 딥러닝에서 널리 사용되는 컨볼루션 신경망 중 하나로, Microsoft Research에서 개발되었습니다. "ResNet"은 "Residual Network"의 줄임말로, 네트워크의 깊이를 키우는 동안 그라디언트 소실 문제를 해결하기 위한 특별한 구조를 가지고 있습니다
<img src = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/resnet_architecture.png">

* 우수 활용 사례: tensorflow Airbnb
  * 방 유형을 분류해서 사진을 업로드해야하는 상황(주방이냐, 거실이냐 등)
  * 데이터 라벨링을 위해 하이브리드(사람 검수 + 자동화)방법 도입
  * 사람들이 인터넷에 업로드한 사진과 설명을 사용하여 라벨링에 이용
  > * 주방 사진만 가져오기 위해서 SQL로 필터링
  
    ``` SQL
    WHERE LOWER(caption) NOT LIKE "%livingroom%"
    WHERE LOWER(caption) NOT LIKE "%bed%"
    WHERE LOWER(caption) NOT LIKE "%bath%"
    ```

* #### ResNet-50 모델의 특징
* **깊이:**<br>
 > ResNet-50은 50개의 레이어로 구성됩니다. 이는 이전의 네트워크 아키텍처인 VGG나 AlexNet보다 더 깊은 네트워크입니다.
* **Residual connections:** <br>
> ResNet의 핵심 아이디어는 잔차 학습(Residual Learning)입니다. 각 블록에서 입력이 출력에 직접 추가되는 잔차 연결(residual connection)이 있습니다. 이를 통해 네트워크가 더 쉽게 학습하고 최적화할 수 있습니다.
* **Bottleneck 구조:** <br>
  > ResNet-50은 "bottleneck" 구조를 사용합니다. 이는 1x1, 3x3, 1x1 컨볼루션 레이어로 구성된 블록으로, 연산량을 줄이고 효율적으로 학습을 진행할 수 있도록 도와줍니다.

* **Pre-trained 모델:** <br>
> ResNet-50은 대규모 이미지 데이터셋(ImageNet)에 대해 사전 훈련된 가중치를 가진 모델로 제공됩니다. 이를 통해 전이 학습(transfer learning)을 사용하여 다양한 컴퓨터 비전 작업에 쉽게 적용할 수 있습니다.

* **AlexNet**

<img src = "https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbxzopl%2FbtqO7hUvidV%2FSTksKg4JDu6O34PpbVFxl1%2Fimg.png" style = "width:70%">

* ConV층에서 Pooling은 지난 강의에서 보통 max Pooling으로 receptive field에서 지정한 구간내의 max값을 취한다고 했었죠? 그러므로 맥스풀링에서는 파라미터가 없습니다.
* 그렇게 ConV layer를 거치면 이 feature map들을 flatten해주므로 4096개 뉴런이 있는 FC layer로 진입합니다.
* FC층에서 FC6,7 layer에서는 가장 흔히 쓰는 Relu함수를 비선형 함수로 이용합니다
* 그리고 출력층 FC8에서는 1000개의 class score를 뱉기 위한 softmax함수를 이용합니다.
* 2개의 Normalization층은 실험 결과, 크게 효과가 없다고 밝혀져서 현재는 잘 사용하지 않지만 AlexNet에서는 쓰였습니다.

## **CNN**

* Convolution Neural Network란?
> * Convolutional neural network(CNN 또는 ConvNet)란 데이터로부터 직접 학습하는 딥러닝의 신경망 아키텍처입니다.
> * CNN은 영상에서 객체, 클래스, 범주 인식을 위한 패턴을 찾을 때 특히 유용합니다. 또한, 오디오, 시계열 및 신호 데이터를 분류하는 데도 매우 효과적입니다.

* #### 특징 학습, 계층 및 분류
* CNN은 입력 계층, 출력 계층, 그리고 그 사이의 은닉 계층으로 구성됩니다.
<img src = "https://kr.mathworks.com/discovery/convolutional-neural-network/_jcr_content/mainParsys/band_copy_copy/mainParsys/lockedsubnav/mainParsys/columns/a32c7d5d-8012-4de1-bc76-8bd092f97db8/image_792810770_copy.adapt.full.medium.jpg/1704399443460.jpg" style = "width: 70%">

> * **컨볼루션 계층**은 입력 영상을 일련의 컨볼루션 필터에 통과시킵니다. <br>각 **필터는 영상에서 feature를** 활성화합니다.<br>행과 열에 대해서 Inner product연산을 수행하기 때문에 Convolution 2D라고 칭합니다. -> feature map을 추출합니다.
  * 필터(kernel) 곧 weight이고, 필터는 많으면 많을수록 좋다.
  * 연산 중 channel의 수는 bias의 수와 동일하다. 

> * **ReLU(Rectified Linear Unit) 계층**은 음수 값은 0에 매핑하고 양수 값은 그대로 두어서 더 빠르고 효과적인 훈련이 이루어지도록 합니다. <br>이때 활성화된 특징만 다음 계층으로 전달되므로 이를 활성화라고도 합니다.

> * **풀링 계층**은 비선형 다운샘플링(down-sampling)을 수행하여 신경망이 학습해야 하는 파라미터의 개수를 줄임으로써 출력을 단순화합니다.<br>
  * Max Pooling은 각 영역에서 가장 큰 값을 선택하여 해당 영역을 대표하는 값을 출력합니다. 일반적으로 2x2 윈도우 크기와 2x2 스트라이드(stride)를 사용합니다. 이는 특징 맵의 크기를 절반으로 줄입니다.
  * 예를 들어, 2x2 Max Pooling은 2x2 영역에서 가장 큰 값을 선택하여 2x2 픽셀 영역을 1개의 픽셀로 대체합니다.
  이러한 연산이 수십 또는 수백 개의 계층에 대해 반복되며, 각 계층은 서로 다른 특징을 식별하도록 학습합니다.
  > * **Max Pooling에서 데이터 손실은 없는가?**
    * Max Pooling에서는 각 영역에서 가장 큰 값을 선택하고, 선택된 값만 사용하여 특징 맵을 다운샘플링합니다. 이러한 과정에서 선택되지 않은 값들은 버려지게 됩니다. 따라서 Max Pooling은 데이터의 일부 손실을 유발합니다.
    * 그러나 이 손실이 항상 부정적인 영향을 미치는 것은 아닙니다. Max Pooling은 불필요한 세부 정보를 제거하고 중요한 특징을 강조하는 역할을 하며, 과적합을 방지하고 모델의 일반화 성능을 향상시킬 수 있습니다. 또한, 최종적으로는 손실된 세부 정보가 네트워크의 다른 레이어에서 보정될 수 있습니다.<br>
  따라서 Max Pooling은 데이터 손실을 유발하지만, 이는 모델의 학습과 일반화에 긍정적인 영향을 미칠 수 있습니다.

    * 컴퓨터의 성능이 좋아지고 있으므로, 이미지의 크기를 줄이는 과정은 사라지는 추세임.
  
  

> * **Fully Connected 레이어:**
  * Fully Connected 레이어는 모든 입력 뉴런과 출력 뉴런이 서로 연결된 레이어입니다. 이는 각 입력과 출력 사이의 모든 가능한 연결을 가집니다.
  * Fully Connected 레이어는 주로 분류(classification)나 회귀(regression)와 같은 작업을 위해 사용됩니다. 주로 출력 레이어로 사용되며, 입력 데이터를 평탄화(flatten)한 후에 사용됩니다.
  * 예를 들어, 입력 이미지의 특징을 추출한 후에 Fully Connected 레이어를 사용하여 이미지에 대한 분류를 수행할 수 있습니다. 이때, Fully Connected 레이어의 출력은 각 클래스에 대한 확률 값을 나타냅니다.
  * Fully Connected 레이어의 출력은 보통 클래스 수와 같은 크기를 가지며, 각 클래스에 대한 점수를 나타냅니다. 이 점수는 일반적으로 활성화 함수를 통해 클래스에 속할 확률로 변환됩니다. Softmax 함수는 이 점수를 확률로 변환하여 각 클래스에 속할 확률을 계산합니다.

<img src = "https://kr.mathworks.com/discovery/convolutional-neural-network/_jcr_content/mainParsys/band_copy_copy/mainParsys/lockedsubnav/mainParsys/columns/a32c7d5d-8012-4de1-bc76-8bd092f97db8/image_2109075398_cop.adapt.full.medium.jpg/1704399443473.jpg" style = "width: 70%">

> 여러 컨볼루션 계층이 있는 신경망의 예. 각 훈련 영상에 서로 다른 해상도의 필터가 적용되고, 컨볼루션된 각 영상은 다음 계층의 입력으로 사용됩니다.

## **CNN 작동원리**
<img src = "https://velog.velcdn.com/images/nayeon_p00/post/e6e205e9-4bd7-4789-bf58-8c1b42fde7bd/image.png">

> 위 input image에서 순서대로 필터와 같은 크기 부분과 필터의 inner product 연산을 해주면 아래 빨간색 테두리 처럼 해당위치의 결과값은 4가 나옵니다. 이과정을 순서대로 위치를 옮겨가며 진행하게 됩니다.

<img src = "https://velog.velcdn.com/images/nayeon_p00/post/218f9c49-9030-4e02-9cb3-160766b82ef3/image.png">

> 3 * 3 결과 행렬에는 필터의 값과 이미지 행렬의 값을 곱하고 총합해준 값들이 저장됩니다.  
