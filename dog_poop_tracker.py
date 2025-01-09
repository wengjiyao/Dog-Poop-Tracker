import os
import cv2
import torch
import numpy as np
from datetime import datetime
from playsound import playsound
from ultralytics import YOLO
import threading

from dog_pooping_detector.detector import PoopingDetector

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

MODEL_PATH = 'yolov8s.pt'        # YOLO model (change if you have a different YOLO model)
CAMERA_SOURCE = 'video/example_input.MOV'
# For real camera, you can use CAMERA_SOURCE = 0 or an IP camera URL if desired

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
# For Jetson / Nvidia devices, you may set DEVICE = 'cuda' if supported, else 'cpu'

OUTPUT_FOLDER = 'captured'
SOUND_PATH = 'sound/iphone-camera-capture-6448.mp3'

# Ratio for the dog's bounding box overlay (3:2)
RATIO = 3.0 / 2.0
TARGET_W = 600
TARGET_H = 400
MARGIN_PERCENT = 0  # e.g., 0.05 if you want a 5% margin

FLASH_FRAMES_TRIGGER = 3    # number of frames to show the white flash
FREEZE_FRAMES_TRIGGER = 24  # number of frames to freeze after flash
COOLDOWN_PERIOD = 25 * 5    # Poop detection cooldown (frames); set to ~5 seconds at 25 FPS

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def play_camera_click(sound_path: str = SOUND_PATH):
    """
    Play the camera shutter sound in a separate thread to avoid blocking.
    """
    playsound(sound_path)


def expand_to_ratio_with_margin(x1, y1, x2, y2, frame_w, frame_h,
                                ratio=RATIO, margin_percent=MARGIN_PERCENT):
    """
    Expand the bounding box to a target ratio (3:2) with optional margin,
    ensuring the bounding box stays within frame boundaries.
    """
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return x1, y1, x2, y2  # Degenerate bounding box => do nothing

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    current_ratio = w / h

    # 1) Expand bounding box to maintain at least 3:2 ratio
    if current_ratio < ratio:
        # "Too tall" => expand width
        desired_w = int(ratio * h)
        x1_new = cx - desired_w // 2
        x2_new = x1_new + desired_w
        y1_new, y2_new = y1, y2
    else:
        # "Too wide" => expand height
        desired_h = int(w / ratio)
        y1_new = cy - desired_h // 2
        y2_new = y1_new + desired_h
        x1_new, x2_new = x1, x2

    # 2) Add a margin around the bounding box
    box_w = x2_new - x1_new
    box_h = y2_new - y1_new
    margin_x = int(box_w * margin_percent)
    margin_y = int(box_h * margin_percent)

    x1_new -= margin_x
    y1_new -= margin_y
    x2_new += margin_x
    y2_new += margin_y

    # 3) Clamp to frame boundaries
    x1_new = max(0, x1_new)
    y1_new = max(0, y1_new)
    x2_new = min(frame_w, x2_new)
    y2_new = min(frame_h, y2_new)

    return int(x1_new), int(y1_new), int(x2_new), int(y2_new)


def main():
    """
    Main function to open video/camera stream, detect dog using YOLOv8,
    and determine whether the dog is pooping via the PoopingDetector.
    """
    print("MPS Available:", torch.backends.mps.is_available())
    print("Using device:", DEVICE)

    # YOLO model
    model = YOLO(MODEL_PATH)

    # Detector for dog-pooping
    detector = PoopingDetector(N=4)  # change N if you want stricter/looser detection

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"Unable to open video source: {CAMERA_SOURCE}")
        return

    # Create the output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    pooping_detected = False
    flash_frames = 0
    freeze_frames = 0
    frozen_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]

        # Inference
        results = model(frame, device=DEVICE)
        result = results[0]

        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")  # (n, 4)
        classes = np.array(result.boxes.cls.cpu(), dtype="int")  # (n,)

        dog_roi_shown = False

        for cls, bbox in zip(classes, bboxes):
            x1, y1, x2, y2 = bbox

            # COCO dataset 'dog' label = class 16
            if cls == 16:
                # 1) Pad ROI to make it square (for classifier)
                width = x2 - x1
                height = y2 - y1

                # Safeguard if bounding box is invalid
                if width < 1 or height < 1:
                    continue

                if width > height:
                    pad = (width - height) // 2
                    y1_padded = max(0, y1 - pad)
                    y2_padded = min(frame.shape[0], y2 + pad)
                    dog_roi = frame[y1_padded:y2_padded, x1:x2]
                else:
                    pad = (height - width) // 2
                    x1_padded = max(0, x1 - pad)
                    x2_padded = min(frame.shape[1], x2 + pad)
                    dog_roi = frame[y1:y2, x1_padded:x2_padded]

                # 2) Expand bounding box for 3:2 overlay
                nx1, ny1, nx2, ny2 = expand_to_ratio_with_margin(
                    x1, y1, x2, y2, frame_w, frame_h, RATIO, MARGIN_PERCENT
                )

                dog_roi_with_margin = frame[ny1:ny2, nx1:nx2]
                if dog_roi_with_margin.size > 0:
                    dog_resized = cv2.resize(
                        dog_roi_with_margin, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA
                    )

                    # 3) Overlay the dog's ROI (600x400) onto the bottom-right corner
                    h_small, w_small = dog_resized.shape[:2]
                    overlay_x = frame_w - w_small - 10
                    overlay_y = frame_h - h_small - 10
                    overlay_x = max(0, overlay_x)
                    overlay_y = max(0, overlay_y)

                    # Draw a background rectangle (dark gray)
                    FRAME_COLOR = (50, 50, 50)
                    BORDER_THICKNESS = 5
                    bg_x1 = overlay_x - BORDER_THICKNESS
                    bg_y1 = overlay_y - BORDER_THICKNESS
                    bg_x2 = overlay_x + w_small + BORDER_THICKNESS
                    bg_y2 = overlay_y + h_small + BORDER_THICKNESS

                    bg_x1 = max(0, bg_x1)
                    bg_y1 = max(0, bg_y1)
                    bg_x2 = min(frame_w, bg_x2)
                    bg_y2 = min(frame_h, bg_y2)

                    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), FRAME_COLOR, cv2.FILLED)
                    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 2)

                    # Overlay dog ROI
                    frame_part = frame[overlay_y:overlay_y + h_small, overlay_x:overlay_x + w_small]
                    if frame_part.shape[0] == h_small and frame_part.shape[1] == w_small:
                        frame[overlay_y:overlay_y + h_small, overlay_x:overlay_x + w_small] = dog_resized

                    dog_roi_shown = True

                # 4) Check if dog is pooping
                if detector.check_if_pooping(dog_roi):
                    # Trigger flash & freeze logic if newly detected
                    if not pooping_detected:
                        pooping_detected = True
                        flash_frames = FLASH_FRAMES_TRIGGER
                        freeze_frames = FREEZE_FRAMES_TRIGGER

                        # Play camera shutter sound in another thread
                        sound_thread = threading.Thread(
                            target=play_camera_click, args=(SOUND_PATH,)
                        )
                        sound_thread.start()

                        # Save the "pooping" frame
                        frozen_frame = frame.copy()
                        cv2.rectangle(frozen_frame, (x1, y1), (x2, y2), (0, 255, 255), 7)

                        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
                        output_path = os.path.join(OUTPUT_FOLDER, f"{timestamp}.jpg")
                        cv2.imwrite(output_path, frozen_frame)

                    # White flash
                    if flash_frames > 0:
                        frame[:, :] = (255, 255, 255)
                        flash_frames -= 1

                    # After flash, freeze the frame
                    if flash_frames == 0 and freeze_frames > 0:
                        freeze_frames -= 1
                        frame = frozen_frame

                else:
                    pooping_detected = False

                # 5) Draw bounding box & text
                if pooping_detected:
                    # Avoid overlaying "POOPING!" during flash or freeze
                    if flash_frames == 0 and freeze_frames == 0:
                        font_scale = 3
                        font_thickness = 5
                        text = "POOPING!"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame, text, (x1, y1), font, font_scale,
                                    (0, 255, 255), font_thickness)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 7)
                else:
                    # Normal red rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow("Main Frame", frame)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
