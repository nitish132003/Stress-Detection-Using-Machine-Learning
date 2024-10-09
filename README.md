Stress Detection Using Machine Learning
This project focuses on detecting stress levels in individuals using machine learning models by analyzing facial expressions and audio inputs. It integrates computer vision techniques and emotional analysis to compute stress in real-time or from pre-recorded videos.

Features:
Real-time Stress Detection: Utilizes live camera feed to capture facial expressions and compute stress levels based on emotions.
Video Stress Analysis: Upload video files to detect stress levels and analyze audio frequencies.
Emotion Detection: Leverages the Facial Expression Recognition (FER) library to classify emotions like happiness, sadness, anger, and more.
Audio Analysis: Extracts audio from video files to analyze audio frequencies for further stress evaluation.
Data Export: Outputs detailed analysis, including stress levels, dominant emotions, and audio frequency, to an Excel file for both live and video sessions.
Interactive Web Interface: A user-friendly, responsive web interface built with Flask for live and video-based stress detection.
Tech Stack:
Backend: Python, Flask
Frontend: HTML, CSS, JavaScript (with Bootstrap for responsiveness)
Machine Learning: FER library, MTCNN for face detection
Audio Processing: SciPy, MoviePy
Data Storage: Excel file output via Pandas
Libraries/Frameworks: OpenCV, MoviePy, Flask, Pandas, SciPy, MTCNN
How It Works:
Real-time Detection: The camera feed is processed to detect facial expressions and calculate stress levels. Audio analysis is also performed in real-time.
Video Upload: Users can upload a video to detect stress and extract emotion and audio data, with the results exported to an Excel sheet.
Live Visual Feedback: Both the live camera and processed stress level are displayed on the webpage, giving users real-time feedback.
Usage:
Clone the repository.
Install the required dependencies (requirements.txt).
Run the Flask app locally (app.py).
Access the web interface to start live stress detection or upload a video for analysis.
