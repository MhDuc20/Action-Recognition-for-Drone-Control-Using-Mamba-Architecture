import os
import cv2
import mediapipe as mp
import csv

# Đường dẫn tới thư mục chứa video
folder_path = r"C:\Users\Admin\Desktop\project\build data\duc"

# Tạo từ điển ánh xạ tên nhãn thành số
label_map = {
    "Forward": 0,
    "Stop": 1,
    "Up": 2,
    "Land": 3,
    "Down": 4,
    "Back": 5,
    "Left": 6,
    "Right": 7
}

# Khởi tạo mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Đường dẫn file CSV để lưu kết quả
output_csv = os.path.join(folder_path, "hand_landmarks.csv")

# Tạo tiêu đề cho cột trong CSV
header = ["label"]
for i in range(21):
    header.extend([f"x{i+1}", f"y{i+1}", f"z{i+1}"])

# Hàm xử lý từng video
def process_video(video_path, label):
    cap = cv2.VideoCapture(video_path)
    with open(output_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Đổi màu ảnh sang RGB để dùng với mediapipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                # Lấy landmarks của bàn tay đầu tiên (nếu có nhiều tay chỉ lấy một tay)
                hand_landmarks = results.multi_hand_landmarks[0]
                # Tạo hàng dữ liệu gồm nhãn và các điểm landmark
                row = [label]
                for landmark in hand_landmarks.landmark:
                    row.extend([landmark.x, landmark.y, landmark.z])
                
                # Ghi hàng dữ liệu vào CSV
                writer.writerow(row)

    cap.release()

# Xóa file CSV cũ nếu tồn tại
if os.path.exists(output_csv):
    os.remove(output_csv)

# Ghi tiêu đề vào CSV
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

# Lặp qua từng file video trong thư mục và xử lý
for label_name, label_num in label_map.items():
    video_path = os.path.join(folder_path, f"{label_name}.mp4")
    if os.path.exists(video_path):
        process_video(video_path, label_num)
    else:
        print(f"Video for label {label_name} not found.")

# Đóng mediapipe
hands.close()

print("Trích xuất điểm bàn tay và lưu vào CSV thành công.")
