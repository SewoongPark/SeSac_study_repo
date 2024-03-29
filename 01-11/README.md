## **TIL**
**Pandas 추가 이해: groupby**
1. 그룹화의 이유:

###### * 데이터를 특정 기준에 따라 세분화하고 각 그룹에 대한 특정 연산을 수행할 수 있습니다.  * 데이터의 패턴이나 특징을 파악하기 쉽고 간편하게 할 수 있습니다.
그룹화를 통해 데이터를 더 쉽게 이해하고 해석할 수 있습니다.

집계 함수 적용: groupby를 사용하면 각 그룹에 대해 다양한 집계 함수를 적용할 수 있습니다. 예를 들어, 평균, 합계, 표준편차 등을 계산할 수 있습니다.

데이터 탐색과 이해: 그룹화를 통해 데이터를 쉽게 탐색하고 이해할 수 있습니다. 특히 범주형 데이터에 유용하며, 각 범주에 대한 통계를 쉽게 확인할 수 있습니다.

시각화와 함께 사용: groupby 결과를 시각화하여 데이터의 패턴을 빠르게 파악할 수 있습니다. 그룹별로 시각화를 하면 데이터의 특성이나 추세를 시각적으로 이해하기 용이합니다.

데이터 전처리와 결합: 그룹화를 통해 데이터를 전처리하거나 다른 데이터프레임과 결합할 때 유용합니다. 그룹화된 데이터를 기준으로 다른 데이터프레임을 병합할 수 있습니다.

### **데이터 병합 실습**
**특정 column의 월별 평균 값 구하기**
**모든 column의 월별 평균 값 구하기**
**결측치 확인 및 시각화**
  `missingno.matrix()`

**`value_counts()`와 `groupby` 활용하기**
**`지급일자`컬럼을 날짜형식으로 변경하기**

### **정규분포와 왜도, 첨도**
* 표준정규분포: 평균이 0, 표준편차가 1인 정규 분포
    * **표준화** 과정을 거치면 모든 정규분포를 표준정규분포의 형태로 나타낼 수 있다.  
* 관측값과 평균값의 오차를 계산하여 weight값을 정할 때 표준정규분포를 따르는 값으로 지정함

#### **왜도와 첨도**
* **왜도:** 데이터 분포의 좌우 비대칭 정도를 표현하는 척도, 분포가 좌우대칭을 이룰수록 왜도값은 작아지고, 한 쪽으로 심하게 몰려 있으면 왜도값이 증가함

* **첨도:** 분포가 정규분포보다 얼마나 뾰족하거나 완만한지의 정도를 나타내는 척도.

  ##### **확률밀도함수(Probability Density Function, PDF)**
* 확률밀도함수는 연속형 확률 변수에 대해 사용되며, 이를 이용하여 특정 구간에 대한 확률을 계산할 수 있습니다.

* 일반적으로 확률밀도함수는 다음과 같은 특성을 가집니다:
    * 확률밀도함수는 항상 0 이상의 값을 갖습니다.
    * 전체 실수 축에 대한 적분 값은 1이 됩니다.

 
