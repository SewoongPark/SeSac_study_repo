import cv2
import mediapipe as mp
import numpy as np
import time

LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


video_path = r"C:\Users\bluecom001\Desktop\01-08\VideoControl\video_pokemon.mp4"

video_cap = (cv2.VideoCapture(video_path)) # video_path에 지정된 동영상 파일을 읽기 위한 VideoCapture 객체를 생성
cap = cv2.VideoCapture(0) # 카메라를 통한 실시간 비디오 스트림을 읽기 위한 VideoCapture 객체를 생성
                            # 0: 기본 카메라(내장된 웹캠) 

mp_face_detection = mp.solutions.face_detection
# 이미지나 비디오에서 얼굴을 감지하고 해당 얼굴의 위치를 탐지하는 모듈

mp_drawing = mp.solutions.drawing_utils
''' 이미지나 비디오에서 검출된 얼굴이나 메시의 결과를 시각화하기 위한 유틸리티 함수들이 들어있는 모듈
        얼굴 주위에 사각형을 그리거나 얼굴 메시의 점을 연결하는 등의 시각적인 효과를 추가'''

mp_face_mesh = mp.solutions.face_mesh
# 얼굴 메시(landmarks)를 추출하기 위한 모듈, 얼굴의 다양한 부분에 대한 랜드마크(점)를 감지

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection, \
    mp_face_mesh.FaceMesh(
        max_num_faces=1, # 감지할 얼굴 수 
        refine_landmarks=True,
        min_detection_confidence=0.5, # 얼굴 검출 신뢰도를 위한 임계값(confidence)설정
        min_tracking_confidence=0.5 # 얼굴 추적을 위한 신뢰도 임계값
    ) as face_mesh:
  # with (
      # ) as face_mesh: FaceDetection과 FaceMesh 모듈을 함께 사용하기 위한 구성  
     
    while cap.isOpened() or video_cap.isOpened():
        success, camera_frame = cap.read()
        sucess2, video_frame = video_cap.read()
        '''        
        success: 프레임을 성공적으로 읽었으면 True, 그렇지 않으면 False를 반환.
        camera_frame: 현재 프레임의 이미지를 반환.
        '''

        if not success:
            print("웹캠을 찾을 수 없습니다.")
            continue
                # 웹캠이 없거나 초기화에 실패한 경우에 대비하여 예외 처리를 수행
                
        image = cv2.cvtColor(cv2.flip(camera_frame, 1), cv2.COLOR_BGR2RGB)
            # 카메라 캡처에서 얻은 이미지와 화면에 보여지는 이미지 간의 좌우 반전을 조정
            # 이는 일반적으로 카메라가 촬영한 영상의 원점(원점은 좌상단)이 이미지 표시 시스템의 원점과 일치하지 않아서 발생
            # opencv에서 이미지를 구성하는 BGR배열에서 표준인 RGB로 변환  
             
        image.flags.writeable = False # 이미지를 읽기 전용으로 설정, 메모리 관리 및 성능 최적화
        results = face_detection.process(image) # 이미지를 face_detection에 전달하여 얼굴 감지를 수행하고, 결과를 results 변수에 저장
        
        image.flags.writeable = True #이미지를 다시 수정 가능하도록 설정
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #OpenCV에서 이미지를 처리할 때 사용
        
        # Face Mesh
        mesh_results = face_mesh.process(image)
        img_h, img_w = camera_frame.shape[:2] # 카메라 프레임의 높이와 너비를 img_h와 img_w 변수에 저장
        if mesh_results.multi_face_landmarks: #Face Mesh 모델이 얼굴 메시를 성공적으로 감지했다면, 다음 코드 블록을 실행
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in mesh_results.multi_face_landmarks[0].landmark])
                # 감지한 얼굴 Mesh의 랜드마크들을 이미지 크기에 맞게 변환
            center_left = np.mean(mesh_points[LEFT_EYE], axis=0, dtype=np.int32)
                # 왼쪽 눈 부분에 해당하는 랜드마크들의 중심 좌표 계산
            center_right = np.mean(mesh_points[RIGHT_EYE], axis=0, dtype=np.int32)
                # 오른쪽 눈 부분에 해당하는 랜드마크들의 중심 좌표 계산
            
            # Iris_EyeTracking과 segmentation
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                # Mesh 모델로부터 얻은 왼쪽 눈과 오른쪽 눈의 동공 주위를 감싸는 최소한의 원을 생성
                
            center_left_iris = np.array([int(l_cx), int(l_cy)], dtype=np.int32)
            center_right_iris = np.array([int(r_cx), int(r_cy)], dtype=np.int32)
                # 동공의 중심 좌표를 연산을 위해 array로 변환
            
            cv2.circle(image, center_left_iris, int(l_radius), (255, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(image, center_right_iris, int(r_radius), (255, 0, 255), 1, cv2.LINE_AA)
                # 동공의 위치를 시각적으로 표현, cv2.circle(객체, 중심 좌표, 반지름, 색상, 두께, 안티에일리어싱을 적용한 smoothing)
        
        if results.detections: # 감지된 얼굴의 존재 여부를 확인
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
                    # 이미지에 얼굴 감지 결과를 시각적으로 나타냄
                    # detection: 감지된 얼굴에 대한 정보를 담고 있는 객체
                    
            # 왼쪽 눈과 오른쪽 눈의 거리 계산
            if len(results.detections) > 0: # 얼굴 감지 결과가 하나 이상 있는 경우에만 실행
                detection = results.detections[0] # 첫 번째로 감지된 얼굴에 대한 정보를 detection 변수에 저장
                left_eye_pos = mp_face_detection.get_key_point(
                    detection, mp_face_detection.FaceKeyPoint.LEFT_EYE)
                # 얼굴 감지 결과와 얻고자 하는 키포인트 유형(mp_face_detection.FaceKeyPoint.LEFT_EYE)을 인자로 받아 
                # 해당 키포인트의 위치를 반환
                right_eye_pos = mp_face_detection.get_key_point(
                    detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)
                
                distance = ((left_eye_pos.x - right_eye_pos.x) ** 2 +
                            (left_eye_pos.y - right_eye_pos.y) ** 2) ** 0.5
                 # 두 눈의 x, y 좌표를 이용하여 유클리드 거리를 계산
                
                eye_pos_y = np.round(np.mean([center_left[1], center_right[1]]) ** 0.5, 3)

                # 이미지를 복제하여 새로운 이미지를 생성
                modified_image = video_frame.copy()

                bright_factor = -50
                total_control_factor = np.round((eye_pos_y * bright_factor * distance), 2)
                # 밝기 조절 요소: 1) 카메라와 감지된 얼굴 객체간의 거리 
                                # 2) 동공의 좌표
                
                modified_image = cv2.addWeighted(modified_image, 1, modified_image, 0, total_control_factor )                
                # 동공의 위치와 거리에 따라 밝기를 조절하여 추가
                 # 이미지가 하나이므로 첫번째와 두번째 이미지에 가중치를 각각 1과 0을 부여

                adjustable_frame = modified_image
                # 결과 이미지를 다시 원래 변수에 할당

        # 양쪽 눈 동공 사이의 거리 출력        
        cv2.putText(image, f"Left Eye to Right Eye Distance: {total_control_factor}", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 화상 캠 실시간 화면 출력
        cv2.imshow('MediaPipe Face Detection', image)
        # 시청할 영상 출력
        cv2.imshow('video', adjustable_frame )
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
        # 양쪽 눈 사이의 거리 값에 따라 영상 정지, 재생 제어
         # 1초 동안의 정지 시간을 설정한 이유는 다시 고개를 원래대로 돌렸을 때 즉각 재생되게 하기 위함
        if distance < 0.06:
            time.sleep(1)

# 영상과 화상 캠 화면의 객체를 각각 해제:
#   1) 프로그램 종료 및 메모리 정리
#   2) 관련 리소스 할당 해제
cap.release()
video_cap.release()

cv2.destroyAllWindows()
