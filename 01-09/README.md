### Mediapipe와 Face mesh detection을 활용한 영상 원격 제어 프로그램

![image](https://github.com/SewoongPark/SeSac_study_repo/assets/98893325/1ee38d00-94a9-4d5d-a97f-96552c08ddb1)

* mediapipe의 FaceDetection 함수를 활용하여 얼굴을 탐지
* 왼쪽, 오른쪽 동공의 key_point를 측정하여 좌표를 얻고 distance 변수로 할당
* distance 값의 임계점을 지정
* 고개를 돌려 distance 값이 임계점보다 작아지면 time모듈의 sleep 함수를 사용하여 1초 동안 대기
* 원래대로 돌리면 영상을 다시 재생

**Diagram**
![image](https://github.com/SewoongPark/SeSac_study_repo/assets/98893325/21441d4c-25b4-442c-8023-fdab990c64d5)
