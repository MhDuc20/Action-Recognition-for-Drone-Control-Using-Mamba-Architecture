import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import keyboard  # Thư viện mô phỏng nhấn phím

# Load the trained model
model = tf.keras.models.load_model('C:/Users/Admin/Desktop/project/hand_action_model.keras')

# Label mapping dictionary
label_map = {
    0: "Forward",
    1: "Stop",
    2: "Up",
    3: "Land",
    4: "Down",
    5: "Back",
    6: "Left",
    7: "Right"
}

# Key mapping dictionary for DJI Flight Simulator
key_map = {
    "Forward": "w",
    "Stop": None,  # Không nhấn nút
    "Up": "up",    # Mũi tên lên
    "Land": "o",   # Phím "O" để hạ cánh
    "Down": "down",  # Mũi tên xuống
    "Back": "s",
    "Left": "left",    # Mũi tên trái
    "Right": "right"   # Mũi tên phải
}

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Track the previous action to prevent repeated key presses
previous_action = None
current_action_text = "No Action"  # Biến để lưu trạng thái hiển thị phím

# Capture video from the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Flip the image horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmark points as a flat list of x, y, z coordinates
            landmarks = []
            h, w, _ = frame.shape  # Get the frame dimensions

            # Get hand landmarks and calculate bounding box
            x_coords = []
            y_coords = []
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_coords.append(x)
                y_coords.append(y)
                landmarks.extend([lm.x, lm.y, lm.z])

            # Ensure the landmarks are of the expected shape (63,)
            if len(landmarks) == 63:
                # Reshape landmarks for model input
                input_data = np.array(landmarks).reshape(1, -1, 63)

                # Make prediction
                prediction = model.predict(input_data)
                predicted_index = np.argmax(prediction)  # Get the index of the highest prediction

                # Check if the predicted index is in the label_map; if not, set to "None"
                if 0 <= predicted_index < len(label_map):
                    predicted_label = label_map[predicted_index]
                else:
                    predicted_label = "None"

                # Check if the action has changed
                if predicted_label != previous_action:
                    # Release all keys if action changes
                    keyboard.release('w')
                    keyboard.release('s')
                    keyboard.release('left')
                    keyboard.release('right')
                    keyboard.release('up')
                    keyboard.release('down')
                    keyboard.release('o')

                    # Press the new key if applicable
                    if predicted_label in key_map and key_map[predicted_label]:
                        keyboard.press(key_map[predicted_label])

                    # Update the previous action
                    previous_action = predicted_label

                # Update the action text for display
                current_action_text = f"Action: {predicted_label}" if predicted_label != "None" else "No Action"

                # Draw the bounding box around the hand
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Bounding box in blue

                # Display the label on the bounding box
                cv2.putText(frame, f'Action: {predicted_label}', (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw landmarks on the hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame with the bounding box
    cv2.imshow('Hand Action Recognition', frame)

    # Create a blank image to display keypress status
    key_status_frame = np.zeros((200, 400, 3), dtype=np.uint8)
    cv2.putText(key_status_frame, current_action_text, (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Keypress Status', key_status_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
hands.close()
