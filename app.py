import gradio as gr
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont

def detect_emotion_from_image(image):
    """
    Detect emotions from an uploaded image
    """
    if image is None:
        return None, "Please upload an image"

    try:
        img_array = np.array(image)

        result = DeepFace.analyze(img_array, actions=['emotion'], enforce_detection=False)

        if isinstance(result, list):
            result = result[0]

        dominant_emotion = result['dominant_emotion']
        emotion_scores = result['emotion']

        img_with_boxes = img_array.copy()
        region = result.get('region', {})

        if region:
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)

            label = f"{dominant_emotion.upper()}"
            cv2.putText(img_with_boxes, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        emotion_text = f"**Dominant Emotion: {dominant_emotion.upper()}**\n\n"
        emotion_text += "**Confidence Scores:**\n"
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        for emotion, score in sorted_emotions:
            emotion_text += f"- {emotion.capitalize()}: {score:.2f}%\n"

        return img_with_boxes, emotion_text

    except Exception as e:
        return image, f"Error detecting emotion: {str(e)}"

def detect_emotion_from_webcam(image):
    """
    Detect emotions from webcam feed
    """
    if image is None:
        return None, "No webcam input detected"

    try:
        img_array = np.array(image)

        result = DeepFace.analyze(img_array, actions=['emotion'], enforce_detection=False)

        if isinstance(result, list):
            result = result[0]

        dominant_emotion = result['dominant_emotion']
        emotion_scores = result['emotion']

        img_with_boxes = img_array.copy()
        region = result.get('region', {})

        if region:
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 3)

            label = f"{dominant_emotion.upper()}"
            font_scale = 1.0
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            cv2.rectangle(img_with_boxes, (x, y-text_height-10), (x+text_width, y), (0, 255, 0), -1)
            cv2.putText(img_with_boxes, label, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        emotion_text = f"**{dominant_emotion.upper()}**\n\n"
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        for emotion, score in sorted_emotions[:3]:
            emotion_text += f"{emotion.capitalize()}: {score:.1f}%\n"

        return img_with_boxes, emotion_text

    except Exception as e:
        return image, f"Error: {str(e)}"

with gr.Blocks(title="Emotion Detection App", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # 🎭 Emotion Detection App

        Detect emotions from facial expressions using AI. Choose between uploading an image or using your webcam for real-time detection.

        **Supported Emotions:** Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
        """
    )

    with gr.Tabs():
        with gr.Tab("📷 Upload Image"):
            gr.Markdown("### Upload a photo to analyze emotions")
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload Image")
                    image_button = gr.Button("Analyze Emotion", variant="primary")
                with gr.Column():
                    image_output = gr.Image(label="Detected Emotion")
                    image_text = gr.Markdown()

            image_button.click(
                fn=detect_emotion_from_image,
                inputs=image_input,
                outputs=[image_output, image_text]
            )

        with gr.Tab("🎥 Webcam Detection"):
            gr.Markdown("### Use your webcam for real-time emotion detection")
            with gr.Row():
                with gr.Column():
                    webcam_input = gr.Image(sources=["webcam"], type="pil", label="Webcam", streaming=True)
                with gr.Column():
                    webcam_output = gr.Image(label="Live Detection")
                    webcam_text = gr.Markdown()

            webcam_input.stream(
                fn=detect_emotion_from_webcam,
                inputs=webcam_input,
                outputs=[webcam_output, webcam_text],
                stream_every=0.5
            )

    gr.Markdown(
        """
        ---
        ### How it works:
        - **Upload Mode:** Upload a photo and click "Analyze Emotion" to detect facial expressions
        - **Webcam Mode:** Enable your webcam for real-time emotion detection
        - The app uses deep learning to identify emotions from facial features
        """
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=5000)
