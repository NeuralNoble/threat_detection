import cv2
import torch
import cvzone
import math
import numpy as np
from ultralytics import YOLO
import gradio as gr

# Check for the appropriate device
device = torch.device("mps" if torch.has_mps else "cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = YOLO("yolo-weights/gun.pt").to(device)

# Class names
classnames = ['gun', 'person']

def draw_transparent_overlay(frame, x1, y1, x2, y2, color=(0, 0, 255), alpha=0.5):
    """Draw a transparent overlay on the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def is_in_extended_area(px1, py1, px2, py2, gx1, gy1, gx2, gy2, extension=50):
    """Check if the gun is in the extended area around the person."""
    ex1, ey1, ex2, ey2 = px1 - extension, py1 - extension, px2 + extension, py2 + extension
    return (gx1 < ex2 and gx2 > ex1 and gy1 < ey2 and gy2 > ey1)

def process_frame(frame):
    results = model(frame)
    persons = []
    threats = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            label = f'{classnames[cls]} {conf}'
            if classnames[cls] == 'person':
                persons.append((x1, y1, x2, y2))
            elif classnames[cls] == 'gun' and conf > 0.4:
                threats.append((x1, y1, x2, y2))

    for (px1, py1, px2, py2) in persons:
        for (gx1, gy1, gx2, gy2) in threats:
            if is_in_extended_area(px1, py1, px2, py2, gx1, gy1, gx2, gy2):
                draw_transparent_overlay(frame, px1, py1, px2, py2)
                cvzone.putTextRect(frame, 'Threat', (px1, py1 - 10), scale=2, thickness=3)

    return frame

def video_processing(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 640))
        frame = process_frame(frame)
        frames.append(frame)
    cap.release()
    return frames

def image_processing(image):
    image = cv2.resize(image, (640, 640))
    image = process_frame(image)
    return image

def gradio_video_interface(video_path):
    frames = video_processing(video_path)
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 640))
    for frame in frames:
        out.write(frame)
    out.release()
    return 'output.mp4'

def gradio_image_interface(image):
    image = image_processing(image)
    return image

video_interface = gr.Interface(
    fn=gradio_video_interface,
    inputs=gr.Video(label="Input Video"),
    outputs=gr.Video(label="Output Video"),
    title="Gun Detection in Video"
)

image_interface = gr.Interface(
    fn=gradio_image_interface,
    inputs=gr.Image(type="numpy", label="Input Image"),
    outputs=gr.Image(type="numpy", label="Output Image"),
    title="Gun Detection in Image"
)

demo = gr.TabbedInterface([video_interface, image_interface], ["Video Detection", "Image Detection"])

if __name__ == "__main__":
    demo.launch()
