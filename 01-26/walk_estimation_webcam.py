import cv2
import mediapipe as mp
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model

mp_pose = mp.solutions.pose

# 함수: 화상 캠으로부터 실시간 랜드마크 좌표 추출
def extract_live_landmarks(num_landmarks, model):
    cap = cv2.VideoCapture(0)
    landmarks_list = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Pose 추정
            image_height, image_width, _ = image.shape
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # 랜드마크 좌표 추출
            landmarks = []
            if results.pose_landmarks:
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append(landmark.x)
                    landmarks.append(landmark.y)

                landmarks_list.append(landmarks)

                # 랜드마크 그리기 (선택적)
                mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 화면에 출력
            font = cv2.FONT_HERSHEY_SIMPLEX
            if len(landmarks_list) >= 10:
                live_landmarks = np.array(landmarks_list[-10:])
                
                # 형태 변환
                live_landmarks = live_landmarks.reshape((1, -1, num_landmarks * 2))

                live_predictions = model.predict(live_landmarks)
                predicted_class = np.argmax(live_predictions[0])

                if predicted_class == 1:
                    status_text = "walking."
                else:
                    status_text = "not walking."
            else:
                status_text = "collecting landmarks data..."

            cv2.putText(image, status_text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Live Pose Detection', image)

            # q 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# 모델 로드
model = load_model('walk_est_01.h5')  # 모델 경로를 적절히 지정해주세요

# 모델의 num_landmarks 확인
num_landmarks = model.input_shape[1]

# 화상 캠으로부터 실시간 랜드마크 추출 및 모델 예측
extract_live_landmarks(num_landmarks, model)
