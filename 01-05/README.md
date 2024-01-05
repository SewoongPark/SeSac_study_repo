### **OpenCV를 활용한 Face Detection 및 여러가지 기능 활용**

#### **OpenCV에서의 얼굴 검출**
* Harr cascading
* 정면, 측면, 반측면등의 얼굴 위치를 찾는 좌표를 기록해놓은 xml 자료
* github에서 clone해와서 활용해보자

#### Harr cascading 이용한 눈, 얼굴 검출
* FaceDetection.py 참조

### **모자이크 효과의 원리**
* 원본에서 모자이크 할 영역을 추출한다
* 추출한 영역을 resize를 통해서 화질을 저하시킨다
* 효과를 적용한 영역을 원본 이미지의 원래 영역에 위치시킨다
