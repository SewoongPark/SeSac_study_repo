import cv2
import mediapipe as mp
import numpy as np
import time


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


video_path = r"C:\Users\bluecom001\Desktop\01-08\VideoControl\video_pokemon.mp4"

video_cap = (cv2.VideoCapture(video_path))
# frame_width = int(video_cap.get(3))
# frame_height = int(video_cap.get(4))
# video_cap = cv2.resize(video_cap, (900,450), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
cap = cv2.VideoCapture(0)


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection, \
    mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

    while cap.isOpened() or video_cap.isOpened():
        success, camera_frame = cap.read()
        sucess2, video_frame = video_cap.read()

        if not success:
            print("웹캠을 찾을 수 없습니다.")
            continue
        
        # 보기 편하기 위해 이미지를 좌우를 반전하고, BGR 이미지를 RGB로 변환합니다.
        image = cv2.cvtColor(cv2.flip(camera_frame, 1), cv2.COLOR_BGR2RGB)
        
        # 성능을 향상시키려면 이미지를 작성 여부를 False으로 설정하세요.
        image.flags.writeable = False
        results = face_detection.process(image)
        
        # 영상에 얼굴 감지 주석 그리기 기본값 : True.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Face Mesh
        mesh_results = face_mesh.process(image)
        img_h, img_w = camera_frame.shape[:2]
        if mesh_results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in mesh_results.multi_face_landmarks[0].landmark])
            center_left = np.mean(mesh_points[LEFT_EYE], axis=0, dtype=np.int32)
            center_right = np.mean(mesh_points[RIGHT_EYE], axis=0, dtype=np.int32)

            # Iris_EyeTracking 그리기
                 # - 동공 추적
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left_iris = np.array([int(l_cx), int(l_cy)], dtype=np.int32)
            center_right_iris = np.array([int(r_cx), int(r_cy)], dtype=np.int32)
            cv2.circle(image, center_left_iris, int(l_radius), (255, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(image, center_right_iris, int(r_radius), (255, 0, 255), 1, cv2.LINE_AA)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

            # 왼쪽 눈과 오른쪽 눈의 거리 계산
            if len(results.detections) > 0:
                detection = results.detections[0]
                left_eye_pos = mp_face_detection.get_key_point(
                    detection, mp_face_detection.FaceKeyPoint.LEFT_EYE)
                
                right_eye_pos = mp_face_detection.get_key_point(
                    detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)
                
                distance = ((left_eye_pos.x - right_eye_pos.x) ** 2 +
                            (left_eye_pos.y - right_eye_pos.y) ** 2) ** 0.5
                
                eye_pos_y = np.round(np.mean([center_left[1], center_right[1]]) ** 0.5, 3)

                eye_pos = (center_left[1]) ** 0.5

                # 이미지를 복제하여 새로운 이미지를 생성
                modified_image = video_frame.copy()

                
                bright_factor = -50
                total_control_factor = np.round((eye_pos_y * bright_factor * distance), 2)
                
                # 동공의 위치와 거리에 따라 밝기를 조절하여 추가
                modified_image = cv2.addWeighted(modified_image, 1, modified_image, 0, total_control_factor )

                # 결과 이미지를 다시 원래 변수에 할당
                adjustable_frame = modified_image

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
        if distance < 0.06:
            time.sleep(1)

# 영상과 화상 캠 화면 객체
cap.release()
video_cap.release()

cv2.destroyAllWindows()

