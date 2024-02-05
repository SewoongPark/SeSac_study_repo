
## **TIL**

### **텍스트 마이닝**
* 문자를 숫자로 바꾸는 작업은 전통적으로 있어 왔음
* 회사의 설정에 맞게 문자를 변경하는 작업 필요

* 문장을 정확하게 분석해야 될 때는 문장을 쪼개야 함
    > * **쪼개는 단위: Tokenize**
    > * 회사별로 필요한 단어셋을 만드는데 많은 작업이 필요함
    > * 특정 도메인에 특화된 어휘를 구성하는 것이 필요

    * 텍스트 분석에서 문서 중 단어의 중요도를 측정하는 방법은<br>
        특정한 단어가 한 문서내에서 얼마나 자주 반복되는지,<br>
        그리고 문서 그룹내에서 동일한 단어가 얼마나 많이 출현하는지를 측정한다.

* **주의할 점: 문자(글자) 그대로 해석하면 안 된다**
    * EX) "보조 도구가 불편하다": 보조 도구가 불편한 게 아니라 **사회 생활을 잘 하기 위한 국가의 보조가 부족**하다는 의미
---
#### **문자 데이터는 벡터로 변환해야 한다**
* 벡터 단위는 크기와 방향을 갖고 있는 스칼라(Scalar)의 집합
* Tensor 연산에서는 숫자로 취급
> * 텍스트들의 분포를 나타내는 **중심 경향성**
    > * **평균, 중앙값, 표준 편차** 등 기술 통계 값 = <span style = "color:skyblue">**단어의 대표값**<span>
* 벡터는 토큰화를 의미하고, 내가 사용하고자 하는 단어의 의미 단위이다. 
* 토큰화된 단어들이 중심 경향성에 따라 모인 벡터 공간이 형성된다. 

**Sparse Matrix(희소 행렬)**
* 문자 데이터를 Embedding 할 때 0으로 채워지는 행렬의 수가 비효율적으로 많음
* 희소 행렬은 0이 아닌 원소들만을 저장하므로, 연산 시에 0인 부분들을 계산하는 비용이 절약

**공기어**
* 하나의 단어와 연관된 다른 단어들의 상관 관계 분석

![image.png](attachment:image.png)
* 출처: *신문 텍스트로 살펴본 문화 소비 현상의 트렌드*<br>
*(A Trend Analysis of Cultural Consumption Basedon Newspaper Texts)*<br>
김혜영,김흥규,강범모
---
* 텍스트를 **벡터화 하고 대표값을 설정**하는 예시

![image-2.png](attachment:image-2.png)
* 출처: *CNN 딥러닝을 활용한 경관 이미지 분석 방법 평가-<br>힐링장소를 대상으로* <br>
*Assessment of Visual Landscape Image Analysis Method Using CNN Deep Learning - Focused on Healing Place*<br>
성정한, 이경진
공주대학교 일반대학원 조경․환경계획학과 박사수료, <br>공주대학교 조경학과 교수

#### **KoNLPy**
* 품사 태깅
  *  형태소 분석기를 사용하여 문장 내의 각 단어에 해당하는 품사를 부착하는 작업

**한국어 형태소 분석기**
* 품사태그셋  https://happygrammer.github.io/nlp/postag-set/
* Hannanum: 한나눔. KAIST Semantic Web Research Center 개발.
* Kkma: 꼬꼬마. 서울대학교 IDS(Intelligent Data Systems) 연구실 개발. http://kkma.snu.ac.kr/
* Komoran: 코모란. Shineware에서 개발. https://github.com/shin285/KOMORAN
* Mecab: 메카브. 일본어용 형태소 분석기를 한국어를 사용할 수 있도록 수정. https://bitbucket.org/eunjeon/mecab-ko
* Open Korean Text: 오픈 소스 한국어 분석기. (구.트위터) 과거 트위터 형태소 분석기. 
  https://github.com/open-korean-text/open-korean-text

<img src = "https://konlpy.org/ko/latest/_images/time.png" style = "width: 50%; height: auto;"><br>
출처: https://konlpy.org/ko/latest/morph/#pos-tagging-with-konlpy
