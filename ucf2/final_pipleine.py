import cv2
import numpy as np
import joblib
import onnxruntime as ort
from collections import deque
import time
import requests

# ====== 🔐 Telegram Config ======
TELEGRAM_BOT_TOKEN = "8072760336:AAEY8qdsG25tit16wQkdeBvXh9e9zCXRhAc"
TELEGRAM_CHAT_ID = "7773650672"
alert_sent = False

def send_telegram_alert(message, image_path=None):
    try:
        # Send message
        url_msg = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        msg_response = requests.post(url_msg, data=data)

        # Send photo if available
        if image_path:
            url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(image_path, 'rb') as photo:
                photo_data = {"chat_id": TELEGRAM_CHAT_ID}
                photo_files = {"photo": photo}
                requests.post(url_photo, data=photo_data, files=photo_files)

        print("📨 Telegram alert sent!" if msg_response.status_code == 200 else f"⚠️ Alert failed: {msg_response.text}")
    except Exception as e:
        print(f"❌ Telegram error: {e}")

# ====== 🎥 Load Models ======
mc3_session = ort.InferenceSession("last hope/mc3_features.onnx")
mc3_input_name = mc3_session.get_inputs()[0].name
clf = joblib.load("crime_detector2.joblib")
scaler = joblib.load("scaler2.joblib")
frame_buffer = deque(maxlen=16)

# ====== 📼 Load Video ======
cap = cv2.VideoCapture("last hope/Fighting024_x264.mp4")  # Use 0 for webcam
print("✅ Models loaded successfully!")

# ====== ⏱️ FPS Setup ======
prev_time = time.time()

# ====== ⏳ Persistent detection control ======
crime_detected_start = None
detection_threshold_seconds = 5  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS calculation
    curr_time = time.time()
    time_diff = curr_time - prev_time
    fps = 1 / time_diff if time_diff > 0 else 0
    prev_time = curr_time

    # Preprocess for MC3
    resized = cv2.resize(frame, (112, 112))
    preprocessed = resized.astype(np.float32) / 255.0
    frame_buffer.append(preprocessed)

    if len(frame_buffer) == 16:
        video_clip = np.stack(frame_buffer, axis=0)
        video_clip = np.transpose(video_clip, (3, 0, 1, 2))
        video_clip = np.expand_dims(video_clip, axis=0)

        # Feature extraction
        mc3_features = mc3_session.run(None, {mc3_input_name: video_clip})[0]
        features_flat = mc3_features.reshape(1, -1)
        scaled_feat = scaler.transform(features_flat)
        prob = clf.predict_proba(scaled_feat)[0][1]

        # Display label
        label = f"Crime Prob: {prob:.2f}"
        color = (0, 0, 255) if prob > 0.5 else (0, 255, 0)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Persistent crime detection logic
        if prob > 0.5:
            if crime_detected_start is None:
                crime_detected_start = time.time()
            elif not alert_sent and (time.time() - crime_detected_start) >= detection_threshold_seconds:
                screenshot_path = "alert_frame.jpg"
                cv2.imwrite(screenshot_path, frame)
                send_telegram_alert(f"🚨 Crime Detected for {detection_threshold_seconds}+ seconds!\nConfidence: {prob:.2f}", screenshot_path)
                alert_sent = True
        else:
            crime_detected_start = None  # Reset timer if crime ends

    # Show FPS
    fps_label = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Display frame
    cv2.imshow("Crime Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
