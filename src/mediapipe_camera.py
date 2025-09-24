# -------- src/mediapipe_camera.py --------
# PURPOSE: Open the webcam and draw MediaPipe face + hand landmarks with distance overlays and HUD. Press 'q' to quit.

import cv2
import mediapipe as mp
import time
import numpy as np
import math

def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if cap.isOpened(): return cap
    cap = cv2.VideoCapture(0)
    if cap.isOpened(): return cap
    for idx in [1, 2, -1]:
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap.isOpened(): return cap
        cap = cv2.VideoCapture(idx)
        if cap.isOpened(): return cap
    return None

def calculate_distance_from_face_size(landmarks, frame_width, frame_height):
    """
    Calculate approximate distance based on face size in the frame.
    This is a rough estimation based on the assumption that faces are typically 
    around 15-20cm wide when at normal viewing distance.
    """
    if not landmarks:
        return None
    
    # Get face bounding box
    x_coords = [landmark.x for landmark in landmarks.landmark]
    y_coords = [landmark.y for landmark in landmarks.landmark]
    
    face_width = (max(x_coords) - min(x_coords)) * frame_width
    face_height = (max(y_coords) - min(y_coords)) * frame_height
    
    # Average face dimensions in cm (typical adult face)
    real_face_width = 16.0  # cm
    real_face_height = 20.0  # cm
    
    # Calculate distance using similar triangles
    # Distance = (Real Size * Focal Length) / Pixel Size
    # Using face width as reference (more stable than height)
    estimated_distance = (real_face_width * 525) / face_width  # 525 is approximate focal length
    
    return max(30, min(200, estimated_distance))  # Clamp between 30-200cm

def calculate_hand_distance(landmarks, frame_width, frame_height):
    """
    Calculate approximate distance based on hand size in the frame.
    """
    if not landmarks:
        return None
    
    # Get hand bounding box
    x_coords = [landmark.x for landmark in landmarks.landmark]
    y_coords = [landmark.y for landmark in landmarks.landmark]
    
    hand_width = (max(x_coords) - min(x_coords)) * frame_width
    hand_height = (max(y_coords) - min(y_coords)) * frame_height
    
    # Average hand dimensions in cm
    real_hand_width = 10.0  # cm
    real_hand_height = 18.0  # cm
    
    # Calculate distance using hand width as reference
    estimated_distance = (real_hand_width * 525) / hand_width
    
    return max(20, min(150, estimated_distance))  # Clamp between 20-150cm

def draw_distance_overlay(frame, landmarks, distance, label, position):
    """Draw distance information overlay on the frame"""
    if distance is None:
        return frame
    
    # Position for the overlay
    x, y = position
    text = f"{label}: {distance:.0f}cm"
    
    # Background rectangle for better visibility
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x-5, y-text_height-10), (x+text_width+5, y+5), (0, 0, 0), -1)
    
    # Distance-based color coding
    if distance < 50:
        color = (0, 0, 255)  # Red - too close
    elif distance < 80:
        color = (0, 255, 255)  # Yellow - close
    else:
        color = (0, 255, 0)  # Green - good distance
    
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

def draw_hud(frame, fps, detection_status, frame_count):
    """Draw the HUD overlay on the frame"""
    height, width = frame.shape[:2]
    
    # Semi-transparent background for HUD
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # FPS counter
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Detection status
    cv2.putText(frame, f"Status: {detection_status}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Frame count
    cv2.putText(frame, f"Frame: {frame_count}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Instructions
    cv2.putText(frame, "Press 'q' to quit", (width - 200, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

def main():
    mp_holistic = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils

    cap = open_camera()
    if cap is None:
        print("ERROR: Could not open the webcam.")
        return

    # FPS calculation variables
    prev_time = 0
    frame_count = 0
    
    with mp_holistic.Holistic(
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False
    ) as holistic:
        print("MediaPipe running with distance overlays and HUD — press 'q' to quit.")
        while True:
            ok, frame = cap.read()
            if not ok: break
            
            frame_count += 1
            current_time = time.time()
            
            # Calculate FPS
            if prev_time != 0:
                fps = 1.0 / (current_time - prev_time)
            else:
                fps = 0.0
            prev_time = current_time

            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            # Determine detection status
            detections = []
            if results.face_landmarks:
                detections.append("Face")
            if results.left_hand_landmarks:
                detections.append("Left Hand")
            if results.right_hand_landmarks:
                detections.append("Right Hand")
            
            detection_status = ", ".join(detections) if detections else "No detections"

            # Draw MediaPipe landmarks
            if results.face_landmarks:
                mp_draw.draw_landmarks(frame, results.face_landmarks,
                                       mp_holistic.FACEMESH_TESSELATION)
                
                # Calculate and display face distance
                face_distance = calculate_distance_from_face_size(results.face_landmarks, width, height)
                if face_distance:
                    # Position overlay near the face
                    face_center_x = int(sum([lm.x for lm in results.face_landmarks.landmark]) / len(results.face_landmarks.landmark) * width)
                    face_center_y = int(sum([lm.y for lm in results.face_landmarks.landmark]) / len(results.face_landmarks.landmark) * height)
                    frame = draw_distance_overlay(frame, results.face_landmarks, face_distance, "Face", (face_center_x + 50, face_center_y))

            if results.left_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.left_hand_landmarks,
                                       mp_holistic.HAND_CONNECTIONS)
                
                # Calculate and display left hand distance
                left_hand_distance = calculate_hand_distance(results.left_hand_landmarks, width, height)
                if left_hand_distance:
                    # Position overlay near the hand
                    hand_center_x = int(sum([lm.x for lm in results.left_hand_landmarks.landmark]) / len(results.left_hand_landmarks.landmark) * width)
                    hand_center_y = int(sum([lm.y for lm in results.left_hand_landmarks.landmark]) / len(results.left_hand_landmarks.landmark) * height)
                    frame = draw_distance_overlay(frame, results.left_hand_landmarks, left_hand_distance, "L Hand", (hand_center_x + 30, hand_center_y))

            if results.right_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.right_hand_landmarks,
                                       mp_holistic.HAND_CONNECTIONS)
                
                # Calculate and display right hand distance
                right_hand_distance = calculate_hand_distance(results.right_hand_landmarks, width, height)
                if right_hand_distance:
                    # Position overlay near the hand
                    hand_center_x = int(sum([lm.x for lm in results.right_hand_landmarks.landmark]) / len(results.right_hand_landmarks.landmark) * width)
                    hand_center_y = int(sum([lm.y for lm in results.right_hand_landmarks.landmark]) / len(results.right_hand_landmarks.landmark) * height)
                    frame = draw_distance_overlay(frame, results.right_hand_landmarks, right_hand_distance, "R Hand", (hand_center_x - 100, hand_center_y))

            # Draw HUD
            frame = draw_hud(frame, fps, detection_status, frame_count)

            cv2.imshow("mediapipe-camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()