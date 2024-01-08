import cv2 as cv 
import numpy as np
import mediapipe as mp 

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

cap = cv.VideoCapture(0)

# 초기값 설정
max_left_eye_center = (0, 0)
max_right_eye_center = (0, 0)

# 최대값을 저장할 리스트
max_left_eye_centers = []
max_right_eye_centers = []

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            # 왼쪽 눈의 중심 좌표 계산
            center_left = np.mean(mesh_points[LEFT_EYE], axis=0, dtype=np.int32)

            # 오른쪽 눈의 중심 좌표 계산
            center_right = np.mean(mesh_points[RIGHT_EYE], axis=0, dtype=np.int32)

            # Iris 그리기
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left_iris = np.array([int(l_cx), int(l_cy)], dtype=np.int32)
            center_right_iris = np.array([int(r_cx), int(r_cy)], dtype=np.int32)
            cv.circle(frame, center_left_iris, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, center_right_iris, int(r_radius), (255, 0, 255), 1, cv.LINE_AA)

            # 현재 좌표가 최대값을 초과하면 최대값 갱신
            max_left_eye_center, min_left_eye_center = (
                max(max_left_eye_center[0], center_left[0]),
                max(max_left_eye_center[1], center_left[1])
            ), (min(max_left_eye_center[0], center_left[0]),
                min(max_left_eye_center[1], center_left[1]))
            
            max_right_eye_center, min_right_eye_center = (
                max(max_right_eye_center[0], center_right[0]),
                max(max_left_eye_center[1], center_left[1])
            ), (min(max_left_eye_center[0], center_left[0]),
                min(max_left_eye_center[1], center_left[1]))
            

            # 최대값을 초과하지 않으면 현재 좌표를 리스트에 추가
            if center_left[0] == max_left_eye_center[0] and center_left[1] == max_left_eye_center[1]:
                max_left_eye_centers.append(center_left)

            if center_right[0] == max_right_eye_center[0] and center_right[1] == max_right_eye_center[1]:
                max_right_eye_centers.append(center_right)

            cv.putText(frame, f"Left_eye_coordinate: {str(center_left)}", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv.putText(frame, f"Right_eye_coordinate: {str(center_right)}", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.putText(frame, f"Max_Left_eye_coordinate: {str(max_left_eye_center)}", (50, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv.putText(frame, f"Max_Right_eye_coordinate: {str(max_right_eye_center)}", (50, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
          
            # angle_range = (max_left_eye_center[0] - center_left[0]) - 
            # if center_left[0] - max_left_eye_center[0]:

        cv.imshow('img', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
