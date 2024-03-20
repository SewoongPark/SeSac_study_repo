## **TIL**
### 현업에서 중요한 기술은 MODEL serving과 경량화
* 기술 블로그 읽어보기
*   <a>https://channeltech.naver.com/contentDetail/76</a>

> 모델을 serving할 때 어떤 형태로 배포될지 정해진 것이 없기 때문에 적절한 형태(binary file, .h5등)로 변환해야 함
* 최종 serving시 별도의 변환 없이 바로 적용할 수 있는 모델을 만드는 것이 이상적이지만<br>이러한 경우는 거의 없으므로 모델의 적절한 변환에 대해 고민해봐야 함
* 최종적인 output이 어떤 product로 쓰일 것인가에 대한 명확한 요구 사항 필요
  * ex) 모바일, 데스크톱등의 size에 따라 달라지는 UI, UX
* **💠주목할 하드웨어 연산기: TensorRT**
  * nvidia 사이트 자주 확인 <a>https://www.nvidia.com/ko-kr/</a> 
