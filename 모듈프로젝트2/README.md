## **Pima Indians Diabetes Classification**

### 목표: 다양한 특성에 따라 환자가 당뇨병에 취약한지를 분류/예측
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

tensor말고 array로 바꾸기<br>
reshape (행 4000) 4000 * 32 * 32 4000 *1 *7 -> 4000 1 7 1 -> tile 4000 1 32 32<br>
32는 7의 배수가 아님 35까지 만들고 slicing<br>
4000 1 32 1 tile 4000 35 35 1  tile or zero padding 4000 35 35 3 index로 slicing<br>

![image](https://github.com/SewoongPark/SeSac_study_repo/assets/98893325/13c9a788-5fcf-40e6-8b73-2c706b82c077)
