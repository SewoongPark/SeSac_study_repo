
# 화산 폭발 분류

### **주제 설명**

**화산 폭발**은 지구 상에서 자연적으로 발생하는 현상 중 하나로, 그 위험성은 다양한 요인에 의해 결정됩니다. 몇 가지 주요 위험 요인은 다음과 같습니다:

1. **화산성 재해**: 화산 폭발로 인해 생성되는 화산재, 화산재류, 화산재류 등의 재해는 인명 및 재산에 위협을 가합니다. 이러한 재해는 주변 지역의 주거지나 농경지에 피해를 입힐 수 있습니다.
2. **대기 오염**: 화산 폭발로 인해 대기 중으로 방출되는 화학물질과 먼지는 대기 오염을 유발할 수 있습니다. 이러한 오염은 공기 중의 입자물질과 유독 가스 농도를 증가시켜 호흡기 질환을 유발하거나 악화시킬 수 있습니다.
3. **지질적 영향**: 화산 폭발은 지형을 변경하고 지질학적 특성을 변형시킬 수 있습니다. 이는 지역 생태계에 영향을 미치고, 수자원 및 토양 품질에 영향을 줄 수 있습니다.
4. **항공 안전**: 화산 폭발로 발생하는 화산재 및 재해류는 항공 안전에도 위협을 줄 수 있습니다. 화산재는 비행기 엔진을 손상시키거나 비행 경로를 방해할 수 있습니다.
5. **사회 경제적 영향**: 큰 화산 폭발은 지역 사회와 경제에 큰 영향을 미칠 수 있습니다. 주거지와 인프라의 파괴, 농작물 손실, 관광 산업에 대한 영향 등이 있습니다.

> *최근 기상청 및 소방방재청의 연구결과 백두산 화산은 가까운 장래에 폭발할 것이라고 전망하고 있다. 화산 폭발의 영향은 화산마다 다르지만 폭발 기간이 적게는 수시간, 많으면 10년 이상 지속된다.*
> 

***발췌: “백두산 화산폭발에 대비한 도로에서의 화산재 청소 방안”*** 

### **피해 예시**

[아이슬란드 화산: 한 달 만에 또 폭발...용암이 주택가 집어삼켜 - BBC News 코리아](https://www.bbc.com/korean/articles/cv2d7390rv4o)

"https://prod-files-secure.s3.us-west-2.amazonaws.com/15c3acb8-87af-40e5-a830-d7a234031660/f380504f-8c11-4c3b-b922-c29e946e1632/Untitled.png"

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/15c3acb8-87af-40e5-a830-d7a234031660/e72cb05a-a690-4d78-a533-4c2f30f96674/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/15c3acb8-87af-40e5-a830-d7a234031660/7ecbfab7-7f70-4a88-bb73-84bd3178a9ed/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/15c3acb8-87af-40e5-a830-d7a234031660/78f983a0-0f71-4c44-afdd-106f404b23ee/Untitled.png)

---

### 데이터 셋 소개: **On Board Volcanic Eruption Detection**

> 원격 감지(Remote Sensing, RS): RS의 모든 가능한 분야 중 이 논문은 위성 이미지에서 객체 검출에 중점을 두며 최종 목표는 관련된 인공 지능 기반 알고리즘을 구현하는 것입니다.
> 

**링크**

https://github.com/alessandrosebastianelli/OnBoardVolcanicEruptionDetection

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/15c3acb8-87af-40e5-a830-d7a234031660/e7bf6a82-59fa-47cb-bbc8-d353e0346026/Untitled.png)

*The final dataset contains 260 images for the class eruption and 1500 for the class non-eruption.*

폭발 클래스: 260장

정상 클래스: 1500장 

(불균형 분포 : Few shot-learning이나 longtail-learning 기법 필요할 것으로 보임)

### 논문 소개: ***On-Board Volcanic Eruption Detection through CNNs and Satellite Multispectral Imagery***

### **Proposed Model: CNN Network**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/15c3acb8-87af-40e5-a830-d7a234031660/70bc6d9a-4227-4016-b081-b90ef0283a01/Untitled.png)

제안된 CNN은 두 개의 하위 네트워크로 나눌 수 있습니다: 

**첫 번째 합성곱 네트워크는 특징 추출을 담당**하고 **두 번째 완전히 연결된 네트워크는 분류 작업을 담당**합니다. 제안된 모델을 처음부터 구축하기 위해 사용된 아키텍처는 컴퓨터 비전 (CV) 커뮤니티에서 자주 사용되는 전통적인 모델에서 유도될 수 있습니다. 예를 들어, AlexNet/VggF 또는 LeNet-5 등이 있습니다. 이러한 아키텍처와의 **주요 차이점은 합성곱 및 밀집 레이어의 수 및 전통적인 Flatten 레이어 대신 Global Average pooling의 사용**입니다.

첫 번째 네트워크는 일곱 개의 합성곱 레이어로 구성되어 있으며, 각각의 레이어 뒤에는 배치 정규화 레이어, ReLU 활성화 함수 및 Max pooling layer가 따릅니다. 각 합성곱 레이어는 (1,1)의 Stride 값과 16에서 512까지 증가하는 필터(unit) 수를 갖습니다. 각 Max pooling layer(커널과 스트라이드 모두 크기가 (2,2)인 경우)는 feature map 차원을 반으로 줄입니다. 

두 번째  네트워크는 다섯 개의 Fully-connected layer로 구성되어 있으며, 각 레이어 뒤에는 ReLU 활성화 함수와 Dropout 레이어가 따릅니다. 이 경우 각 레이어의 element(unit을 말하는 듯) 수가 감소합니다. 제안된 아키텍처에서 두 하위 네트워크는 전통적인 flatten 레이어 대신 Global Average pooling layer로 연결되어 있습니다. 이는 훈련 과정을 가속화하기 위해 학습 가능한 매개변수의 수를 크게 줄입니다.

### ***3.1. Image Loader and Data Balancing***

 데이터셋은 불균형적인 결과를 보입니다. 한 클래스의 예제 수가 다른 클래스보다 훨씬 많은 불균형 데이터셋은 모델이 지배적인 클래스만 인식하도록 할 수 있습니다. 이 문제를 해결하기 위해 Phi-Lab의 `ai4eo.preprocessing` 라이브러리에서 `Image Loader`라는 외부 함수가 사용되었습니다.

이 라이브러리를 사용하면 기존의 `Keras` 버전보다 훨씬 효율적인 이미지 로더를 정의할 수 있습니다. 더 나아가 데이터 증강기를 구현하여 추가적인 변환을 정의할 수 있습니다. 이 라이브러리의 가장 강력한 기능 중 하나는 오버샘플링 기술을 사용하여 데이터셋을 균형 잡는 것과 관련된 것입니다. 특히, 각 클래스는 해당 클래스 샘플의 수에 기초한 값으로 독립적으로 가중치가 부여됩니다. 이러한 오버샘플링 메커니즘은 소수 클래스인 이 경우 분출 클래스에 데이터 증강을 적용하여 작동합니다.

후자는 소수 클래스가 주수 클래스와 동일한 샘플 수가 될 때까지 시작 이미지의 변환(예: 회전, 잘라내기 등)을 적용하여 새로운 이미지를 생성합니다. 이러한 널리 알려진 전략은 훈련의 계산 비용을 증가시키지만, 학습 단계에서 분류기가 특정 클래스에 강화되지 않도록 도와줍니다. 이 프로시저 이후 각 클래스에 대해 사용 가능한 데이터가 동등하게 되므로 이러한 전략은 분류기의 효율성을 향상시킵니다.

### *5.1. Training on the PC*

대형 확장 모델에서는 accuracy가 85%에 이르렀고, 가볍고 경량화된 모델은 83%에 이르렀습니다. 

다음의 표는 테스트 데이터에서 실행된 예측에 대한 GT값과 예측 값의 비교를 기록한 것입니다. 

| Column | Row 1 |  | Row 2 |  | Row 3 |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | Ground Truth | Predicted | Ground Truth | Predicted | Ground Truth | Predicted |
| 1 | 1.00 | 0.13 | 0.00 | 0.03 | 0.00 | 0.00 |
| 2 | 1.00 | 0.99 | 1.00 | 0.99 | 1.00 | 0.99 |
| 3 | 1.00 | 0.99 | 1.00 | 0.99 | 0.00 | 0.00 |
| 4 | 1.00 | 0.95 | 0.00 | 0.00 | 1.00 | 0.99 |
| 5 | 1.00 | 0.98 | 1.00 | 0.99 | 0.00 | 0.00 |
| 6 | 1.00 | 0.99 | 1.00 | 0.99 | 0.00 | 0.02 |
| 7 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.99 |
| 8 | 0.00 | 0.00 | 0.00 | 0.26 | 0.00 | 0.88 |
| 9 | 0.00 | 0.00 | 1.00 | 0.99 | 0.00 | 0.01 |

낮은 예측 값은 낮은 폭발 가능성을 나타냅니다. 그 반대의 경우는 높은 폭발 가능성을 나타냅니다. 

0.5 옆의 값은 결정이 어려운 예측을 나타냅니다. 이 문제가 이진 분류임을 감안하여, 예측에 기반하여 클래스를 식별하기 위해 임계값으로 0.5의 값을 선택했습니다. 0.5보다 낮은 값은 0으로 반올림되어 특정 클래스를 나타내며(분출이 아닌 클래스), 0.5보다 높은 값은 1로 반올림되어 다른 클래스를 나타냅니다(분출 클래스).

---

**추가 데이터 셋 확보를 위한 사이트**

*GOOGLE EARTH ENGINE*

[](https://www.sciencedirect.com/science/article/pii/S0098300422001650)

[Sentinel-2 Datasets in Earth Engine  |  Earth Engine Data Catalog  |  Google for Developers](https://developers.google.com/earth-engine/datasets/catalog/sentinel-2)

크롤링 해볼만한 사이트

[Yar_Jabal](https://gbank.gsj.jp/vsidb/image/Yar_Jabal/aster_p1.html)

**참조 논문**
