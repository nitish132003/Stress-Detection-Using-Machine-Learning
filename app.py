from flask import Flask, render_template, request, Response, jsonify, send_from_directory
import threading
import cv2
from fer import FER
import numpy as np
import pandas as pd
import os
import sounddevice as sd
from scipy.fft import fft, fftfreq

app = Flask(__name__)

# Global variables to store results
live_stress_levels = []
live_dominant_emotions = []
live_dominant_frequencies = []
video_stress_levels = []
video_dominant_emotions = []
stop_live_detection = threading.Event()

# Function to compute stress level based on emotion intensities
def compute_stress_level(emotions):
    dominant_emotion = max(emotions, key=emotions.get)
    stress_mapping = {
        'angry': 80,
        'disgust': 70,
        'fear': 90,
        'happy': 10,
        'sad': 70,
        'surprise': 50,
        'neutral': 30
    }
    stress_level = stress_mapping.get(dominant_emotion, 0)
    emotion_intensity = emotions[dominant_emotion]
    stress_level = int(stress_level * emotion_intensity)
    return stress_level, dominant_emotion

# Function to detect motion between two frames
def detect_motion(prev_frame, curr_frame, threshold=1000):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.sum(mag) > threshold

# Function to preprocess video frame
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

# Class to process audio
class AudioProcessor:
    def __init__(self, chunk_size=1024, sample_rate=44100):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.last_data = np.zeros(chunk_size)
    
    def get_frequency(self):
        def callback(indata, frames, time, status):
            if status:
                print(status, flush=True)
            self.last_data[:] = indata[:, 0]

        with sd.InputStream(callback=callback, channels=1, samplerate=self.sample_rate):
            sd.sleep(1000)  # Capture for 1 second

        fft_vals = np.abs(fft(self.last_data))
        freqs = fftfreq(len(self.last_data), 1 / self.sample_rate)
        return freqs[np.argmax(fft_vals[:len(freqs) // 2])]

# Function to detect stress from live camera feed
def detect_stress_from_camera():
    global live_stress_levels, live_dominant_emotions, live_dominant_frequencies
    live_stress_levels = []
    live_dominant_emotions = []
    live_dominant_frequencies = []
    stop_live_detection.clear()

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    detector = FER(mtcnn=True)
    audio_processor = AudioProcessor()

    ret, prev_frame = cap.read()
    if not ret or prev_frame is None:
        print("Error: Unable to capture from camera.")
        cap.release()
        return
    prev_frame = preprocess_frame(cv2.flip(prev_frame, 1))

    try:
        while not stop_live_detection.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: Unable to capture from camera.")
                break

            frame = preprocess_frame(cv2.flip(frame, 1))
            motion_detected = detect_motion(prev_frame, frame)
            prev_frame = frame.copy()

            result = detector.detect_emotions(frame)
            dominant_freq = audio_processor.get_frequency()

            if result:
                for face in result:
                    x, y, w, h = face["box"]
                    emotions = face["emotions"]
                    stress_level, dominant_emotion = compute_stress_level(emotions)

                    if motion_detected:
                        stress_level += 10

                    live_stress_levels.append(stress_level)
                    live_dominant_emotions.append(dominant_emotion)
                    live_dominant_frequencies.append(dominant_freq)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, f'Stress Level: {stress_level}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(frame, f'Frequency: {dominant_freq:.2f} Hz', (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    finally:
        cap.release()
        cv2.destroyAllWindows()

        if live_stress_levels:
            df = pd.DataFrame({
                'Stress Level': live_stress_levels,
                'Dominant Emotion': live_dominant_emotions,
                'Dominant Audio Frequency': live_dominant_frequencies
            })
            df.to_excel('live_session_results.xlsx', index=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/live-stress-detection')
def live_stress_detection():
    return render_template('live_stress_detection.html')

@app.route('/video-feed')
def video_feed():
    return Response(detect_stress_from_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop-live-detection', methods=['POST'])
def stop_live_detection_route():
    stop_live_detection.set()

    avg_stress_level = np.mean(live_stress_levels) if live_stress_levels else 0
    emotion_counts = {emotion: live_dominant_emotions.count(emotion) for emotion in set(live_dominant_emotions)}
    avg_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "None"
    avg_frequency = np.mean(live_dominant_frequencies) if live_dominant_frequencies else 0

    return jsonify({
        'avg_stress_level': avg_stress_level,
        'avg_emotion': avg_emotion,
        'avg_frequency': avg_frequency
    })

@app.route('/start-live-detection', methods=['POST'])
def start_live_detection_route():
    if not stop_live_detection.is_set():
        threading.Thread(target=detect_stress_from_camera, daemon=True).start()
    return jsonify({'status': 'Live detection started'})

@app.route('/upload-video', methods=['GET', 'POST'])
def upload_video():
    global video_stress_levels, video_dominant_emotions
    if request.method == 'POST':
        video_file = request.files['video']
        if video_file:
            video_path = os.path.join('uploads', video_file.filename)
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            video_file.save(video_path)
            video_stress_levels, video_dominant_emotions = detect_stress_from_video(video_path)
            
            avg_stress_level = np.mean(video_stress_levels) if video_stress_levels else 0
            emotion_counts = {emotion: video_dominant_emotions.count(emotion) for emotion in set(video_dominant_emotions)}
            avg_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "None"
            
            return render_template('upload_video.html', 
                                   avg_stress_level=avg_stress_level, 
                                   avg_emotion=avg_emotion,
                                   download_ready=True)
    return render_template('upload_video.html')

@app.route('/download-live-excel')
def download_live_excel():
    return send_from_directory('.', 'live_session_results.xlsx')

@app.route('/download-video-excel')
def download_video_excel():
    return send_from_directory('.', 'video_session_results.xlsx')

def detect_stress_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    detector = FER(mtcnn=True)
    stress_levels = []
    dominant_emotions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = preprocess_frame(frame)
        result = detector.detect_emotions(frame)

        if result:
            for face in result:
                emotions = face["emotions"]
                stress_level, dominant_emotion = compute_stress_level(emotions)
                stress_levels.append(stress_level)
                dominant_emotions.append(dominant_emotion)

    cap.release()

    if stress_levels:
        df = pd.DataFrame({
            'Stress Level': stress_levels,
            'Dominant Emotion': dominant_emotions,
        })
        df.to_excel('video_session_results.xlsx', index=False)

    return stress_levels, dominant_emotions

if __name__ == "__main__":
    app.run(debug=True)
