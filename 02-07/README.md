
## **TIL**
### **텍스트 전처리와 그래프 이론**
* **kaggle Netflix 데이터 전처리 및 영화 추천 예제**
* `vectorizer` -> 띄어쓰기 단위로 `token`화 
* `tfidf_vectorizer = TfidfVectorizer()`
* `tfidf_matrix = tfidf_vectorizer.fit_transform(df['all_tokens'])`
     
    * TF-IDF 벡터화의 fit_transform() 함수 역할
    fit_transform() 함수는 TF-IDF 벡터화 과정을 수행하는 주요 함수입니다. 이 함수는 다음과 같은 역할을 수행합니다.

        > 1. 학습 데이터로부터 단어 집합(vocabulary) 생성:
        학습 데이터로부터 모든 단어를 추출하여 단어 집합을 생성합니다.
        단어 집합은 TF-IDF 벡터화 과정에서 사용되는 기본 단위입니다.

        > 2. 각 단어의 TF-IDF 점수 계산:
        학습 데이터 각 문서에서 각 단어의 빈도(TF)를 계산합니다.
        전체 문서 집합에서 각 단어의 문서 빈도(DF)를 계산합니다.
        TF-IDF 점수를 계산합니다.

        > 3. 학습 데이터 벡터화:
        각 문서를 TF-IDF 점수 벡터로 변환합니다.
        벡터화된 문서는 정보 검색, 문서 분류, 문서 요약 등 다양한 NLP 작업에 활용될 수 있습니다.

        > 4. 변환된 벡터 반환:
        학습 데이터의 벡터화된 결과를 반환합니다.
* **그래프 알고리즘은 너무 어려우니까 깊이 공부하지 말기**
* **탐색 알고리즘 공부하는 것이 적절함**

* **그래프 알고리즘**
    * 파이썬으로 그려보기
    <a>https://chaelist.github.io/docs/network_analysis/network_basics/</a>
* 예) 구글의 검색엔진 알고리즘

* **PageRank 소개**
* > 1996년 구글 창업자인 Larry Page와 Sergey Brin이 개발한 PageRank는 웹페이지의 중요도를 결정하는 알고리즘으로 1998년 구글의 검색엔진에 도입됨
* > 구글의 검색 메커니즘에는 PageRank가 존재하나 구글 검색에서 사용하는 약 200여 가지의 검색 알고리즘 중 하나임
* > PageRank는 고유벡터(eigenvector)값으로 구할 수 있음
* > 웹페이지의 관계와 댐핑 팩터를 고려한 정방행렬의 최대 고유치(maximum eigen- value)에 대한 고유벡터로 PageRank값을 구할 수 있음
