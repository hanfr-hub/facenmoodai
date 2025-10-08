import gradio as gr
from fer import FER
import cv2

# detector FER
detector = FER(mtcnn=True)

def analyze_emotion(image):
    if image is None:
        return "No image received"
    result = detector.detect_emotions(image)
    if result:
        emotions = result[0]["emotions"]
        top_emotion = max(emotions, key=emotions.get)
        return f"Dominant Emotion: {top_emotion} ({emotions[top_emotion]:.2f})"
    else:
        return "No face detected"

# UI pakai webcam + upload
demo = gr.Interface(
    fn=analyze_emotion,
    inputs=gr.Image(type="numpy", sources=["upload", "webcam"]),
    outputs="text",
    title="Emotion Detection",
    description="Upload a picture or use your webcam to detect emotions."
)

if __name__ == "__main__":
    demo.launch()
