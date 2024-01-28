import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical


# Function to extract landmarks from a frame
def extract_landmarks(frame, pose):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    landmarks = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
    return landmarks

# Function to load and preprocess data
def load_data(video_path):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = extract_landmarks(frame, mp_pose)
        if landmarks:
            data.append(landmarks)

        # cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # cap.release()
    # cv2.destroyAllWindows()

    return np.array(data)

# Function to create and train a simple model
def create_and_train_model(X_train, y_train):
    num_landmarks = len(X_train[0])
    num_classes = 2  # Adjust based on your task (binary classification)

    model = models.Sequential([
        layers.Flatten(input_shape=(num_landmarks, 3)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30, validation_split=0.2)

    return model

# Example usage
video_path = 'videos/walking_all_02.mp4'
data = load_data(video_path)

# Assuming you have binary classification labels
y_train = np.random.randint(2, size=len(data))

# Convert labels to one-hot encoding
y_train_onehot = to_categorical(y_train, num_classes=2)  # Change num_classes accordingly

# Split the data into training and validation sets
X_train, X_val, y_train_onehot, y_val_onehot = train_test_split(data, y_train_onehot, test_size=0.2, random_state=42)

# Train the model
model = create_and_train_model(X_train, y_train_onehot)

# Evaluate the model if needed
model.evaluate(X_val, y_val_onehot)
model.save('walk_est_01.h5')