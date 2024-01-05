import numpy as np
import cv2
import pyttsx3
import time
import csv
import os

faceCascade = cv2.CascadeClassifier(r"C:\Users\bluecom001\Downloads\haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier(r"C:\Users\bluecom001\Downloads\haarcascade_eye.xml")

engine = pyttsx3.init('sapi5', debug=True)

# def speak(text):
#     engine.say(text)
#     engine.runAndWait()

def BackgroundRemover(filterImg):

    filter_img = cv2.imread(filterImg)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(filter_img, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for white color in HSV
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([255, 30, 255])

    # Create a mask for white color
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Invert the white mask
    white_mask_inv = cv2.bitwise_not(white_mask)

    # Apply the mask to the original image
    result = cv2.bitwise_and(filter_img, filter_img, mask=white_mask_inv)
    return result

def save_image(image, save_path, filename):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, filename)
    cv2.imwrite(save_file, image)
    print(f"Image saved as {save_file}")

# 설정된 저장 경로와 파일 이름
save_path = './'
filename = 'output_image.png'

filterImg = './ironman.png'
result_image = BackgroundRemover(filterImg)
save_image(result_image, save_path, filename)

# 얼굴 정면 촬영 시간을 기록하는 함수
def record_frontal_time(detected, last_frontal_time):
    current_time = time.time()

    if detected:
        elapsed_time = current_time - last_frontal_time
        if elapsed_time > 2:  # 2초 이상 경과 시에만 출력
            print(f"Frontal face detected for {elapsed_time:.2f} seconds")
            last_frontal_time = current_time
    
    return last_frontal_time



def detect(gray, frame, last_frontal_time):
    img = cv2.imread("./output_image.png")

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

    frontal_face_detected = False

    for (x, y, w, h) in faces:
        hx1 = x
        hx2 = x + w
        hy1 = y
        hy2 = y + h

        # 필터 위치 조정
        w2 = w
        h2 = h 
        hx1 = x
        hx2 = x + w2
        hy1 = y
        hy2 = y + h2

        # 리사이즈
        img = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_AREA)

        # 마스크 생성
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img_g, 10, 255, cv2.THRESH_BINARY)
        mask_i = cv2.bitwise_not(mask)

        # 마스크 적용
        roi = frame[hy1:hy2, hx1:hx2]
        frame_bg = cv2.bitwise_and(roi, roi, mask=mask_i)
        frame_fg = cv2.bitwise_and(img, img, mask=mask)
        dst = cv2.add(frame_fg, frame_bg)

        # 결과 적용
        frame[hy1:hy2, hx1:hx2] = dst

        # 얼굴이 정면이 아닐 때 음성 출력
         # 얼굴이 정면으로 보이는 경우
        if w > 150 and h > 150:
           pass
        else:
            # 정면이 아닌 경우 시간 초기화
            start_time = None
            # 얼굴이 정면이 아닌 경우 음성 출력
            engine.say("hahahahhaha")
            engine.runAndWait()



         # 머리 위에 텍스트 추가
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 4

        text = "I'm IronMan"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

        text_x = int((hx1 + hx2 - text_size[0]) / 2)
        text_y = max(hy1 - 10, text_size[1])

        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        for (ex, ey, ew, eh) in eyeCascade.detectMultiScale(gray[y:y + h, x:x + w], 1.1, 3):
            cv2.rectangle(frame[y:y + h, x:x + w], (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

    return frame, frontal_face_detected

# 웹캠에서 실시간 이미지 받아오기
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

last_frontal_time = time.time()

while webcam.isOpened():
    status, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame, frontal_face_detected = detect(gray, frame, last_frontal_time)

    last_frontal_time = record_frontal_time(frontal_face_detected, last_frontal_time)

    cv2.imshow("testing", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    

webcam.release()
cv2.destroyAllWindows()
