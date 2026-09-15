import cv2
import numpy as np
import os


def detect_crack(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def detect_honeycombing(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def mark_honeycombing(frame, contours):
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small areas
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, 'Honeycombing', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def process_video(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(output_dir, 'output.mp4')
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Processing video: {video_path}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        crack_contours = detect_crack(frame)
        cv2.drawContours(frame, crack_contours, -1, (0, 255, 0), 2)

        honeycombing_contours = detect_honeycombing(frame)
        mark_honeycombing(frame, honeycombing_contours)

        output_video.write(frame)  # Save processed frame into the video

    cap.release()
    output_video.release()

    print(f"Processing complete. Processed video saved at: {output_video_path}")


if __name__ == "__main__":
    video_path = "Recording 2025-03-27 182550.mp4"  # Update with your video path
    output_dir = "output_frames"
    process_video(video_path, output_dir)
