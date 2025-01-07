import threading

import cv2
import torch
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import os
from playsound import playsound
from dog_pooping_detector import PoopingDetector

print(torch.backends.mps.is_available())

model = YOLO('yolov8s.pt')  # Ensure you have the model file

# Desired output ratio of width : height = 3 : 2 (which corresponds to 300 : 200, but we'll make it bigger)
RATIO = 3.0 / 2.0
# Let's make it bigger: 600×400 instead of 300×200
TARGET_W = 600
TARGET_H = 400

# We'll leave a 5% margin around the dog's bounding box (based on the final expanded box size)
MARGIN_PERCENT = 0  # 0.05 if you want margin

detector = PoopingDetector(N=4)

def play_camera_click():
    playsound('sound/iphone-camera-capture-6448.mp3')
def expand_to_ratio_with_margin(x1, y1, x2, y2, frame_w, frame_h, ratio=RATIO, margin_percent=MARGIN_PERCENT):
    """
    1) Expand the bounding box to at least `ratio` (3:2) so we do NOT cut out any part of the dog.
       - If it's too tall, expand width.
       - If it's too wide, expand height.
    2) Add a margin around the expanded bounding box (relative to the box size).
    3) Clamp to frame boundaries.
    """
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return x1, y1, x2, y2  # Degenerate bounding box => do nothing

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    current_ratio = w / h

    # 1) Expand bounding box to 3:2 ratio (no shrinking)
    if current_ratio < ratio:
        # "Too tall" => expand width
        desired_w = int(ratio * h)
        x1_new = cx - desired_w // 2
        x2_new = x1_new + desired_w

        y1_new = y1
        y2_new = y2
    else:
        # "Too wide" => expand height
        desired_h = int(w / ratio)
        y1_new = cy - desired_h // 2
        y2_new = y1_new + desired_h

        x1_new = x1
        x2_new = x2

    # 2) Add margin around the bounding box
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


cap = cv2.VideoCapture("video/IMG_5881.MOV")
pooping_detected = False
flash_frames = 0
freeze_frames = 0
frozen_frame = None

# Create the "captured" folder if it doesn't exist
output_folder = "captured"
os.makedirs(output_folder, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]

    # Inference (use device='mps' on Apple Silicon if desired)
    results = model(frame, device='mps')
    result = results[0]

    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")  # (n, 4)
    classes = np.array(result.boxes.cls.cpu(), dtype="int")  # (n,)

    dog_roi_shown = False

    for cls, bbox in zip(classes, bboxes):
        x1, y1, x2, y2 = bbox

        # COCO 'dog' = class 16
        if cls == 16:
            # Extract the dog's ROI
            dog_roi = None  #frame[y1:y2, x1:x2]
            width = x2 - x1
            height = y2 - y1

            if width > height:
                pad = (width - height) // 2
                # Ensure padding does not go out of bounds
                y1_padded = max(0, y1 - pad)
                y2_padded = min(frame.shape[0], y2 + pad)
                dog_roi = frame[y1_padded:y2_padded, x1:x2]
            else:
                pad = (height - width) // 2
                # Ensure padding does not go out of bounds
                x1_padded = max(0, x1 - pad)
                x2_padded = min(frame.shape[1], x2 + pad)
                dog_roi = frame[y1:y2, x1_padded:x2_padded]


            # Expand bounding box to 3:2 + margin
            nx1, ny1, nx2, ny2 = expand_to_ratio_with_margin(
                x1, y1, x2, y2,
                frame_w, frame_h,
                ratio=RATIO,
                margin_percent=MARGIN_PERCENT
            )

            # Extract that region from the frame
            dog_roi_with_margin = frame[ny1:ny2, nx1:nx2]

            if dog_roi_with_margin.size > 0:
                # Resize to 600×400 (bigger window)
                dog_resized = cv2.resize(dog_roi_with_margin, (TARGET_W, TARGET_H),
                                         interpolation=cv2.INTER_AREA)

                # Determine where to place the small image in the main frame
                h_small, w_small = dog_resized.shape[:2]
                overlay_x = frame_w - w_small - 10  # 10 px from the right edge
                overlay_y = frame_h - h_small - 10  # 10 px from the bottom edge

                # Make sure we don't go out of bounds
                overlay_x = max(0, overlay_x)
                overlay_y = max(0, overlay_y)

                # Before we copy the dog image, let's create a framed rectangle behind it
                # Choose a color for the frame background, e.g. dark gray (50,50,50)
                FRAME_COLOR = (50, 50, 50)
                BORDER_THICKNESS = 5

                # Coordinates for the framed background rectangle
                bg_x1 = overlay_x - BORDER_THICKNESS
                bg_y1 = overlay_y - BORDER_THICKNESS
                bg_x2 = overlay_x + w_small + BORDER_THICKNESS
                bg_y2 = overlay_y + h_small + BORDER_THICKNESS

                # Clamp them in case
                bg_x1 = max(0, bg_x1)
                bg_y1 = max(0, bg_y1)
                bg_x2 = min(frame_w, bg_x2)
                bg_y2 = min(frame_h, bg_y2)

                # Draw the filled background rectangle
                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), FRAME_COLOR, cv2.FILLED)

                # Optionally, add a line border in a different color, e.g. white
                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 2)

                # Now copy the dog_resized onto the main frame
                frame_part = frame[overlay_y:overlay_y + h_small, overlay_x:overlay_x + w_small]

                # If you want a simple copy (no alpha blending):
                if frame_part.shape[0] == h_small and frame_part.shape[1] == w_small:
                    frame[overlay_y:overlay_y + h_small, overlay_x:overlay_x + w_small] = dog_resized

                dog_roi_shown = True

            # Check if the dog is pooping
            if detector.check_if_pooping(dog_roi):
                # Flash effect (white screen)
                if not pooping_detected:
                    # play kamera click sound effect
                    sound_thread = threading.Thread(target=play_camera_click)
                    sound_thread.start()

                    pooping_detected = True
                    flash_frames = 3  # Trigger flash effect for 2 frames
                    freeze_frames = 24

                    frozen_frame = frame.copy()
                    cv2.rectangle(frozen_frame, (x1, y1), (x2, y2), (0, 255, 255), 7)

                    # save captured pooping image to folder "captured"
                    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")

                    # Create the full path for the output image
                    output_path = os.path.join(output_folder, f"{timestamp}.jpg")

                    # Save the frame as an image
                    cv2.imwrite(output_path, frozen_frame)

                if flash_frames > 0:
                    frame[:, :] = (255, 255, 255)
                    flash_frames -= 1

                if flash_frames == 0:
                    if freeze_frames > 0:
                        freeze_frames -= 1
                        frame = frozen_frame

            else:
                pooping_detected = False

            # draw boundingbox
            if pooping_detected:
                # Display "POOPING" text
                font_scale = 3
                font_thickness = 5
                text = "POOPING!"
                font = cv2.FONT_HERSHEY_SIMPLEX

                if flash_frames == 0 and freeze_frames == 0:
                    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                    cv2.putText(frame, text, (x1, y1), font, font_scale, (0, 255, 255), font_thickness)

                    # Draw a thick yellow rectangle if pooping
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 7)
            else:
                # Draw a normal red rectangle if not pooping
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the main frame with the overlayed dog ROI
    cv2.imshow("Main Frame", frame)

    # Press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
