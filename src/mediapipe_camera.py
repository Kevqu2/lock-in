# -------- src/mediapipe_camera.py --------
# PURPOSE: Open the webcam and draw MediaPipe face + hand landmarks. Press 'q' to quit.

import cv2
import mediapipe as mp

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

def main():
    mp_holistic = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils

    cap = open_camera()
    if cap is None:
        print("ERROR: Could not open the webcam.")
        return

    with mp_holistic.Holistic(
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False
    ) as holistic:
        print("MediaPipe running — press 'q' to quit.")
        while True:
            ok, frame = cap.read()
            if not ok: break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            if results.face_landmarks:
                mp_draw.draw_landmarks(frame, results.face_landmarks,
                                       mp_holistic.FACEMESH_TESSELATION)
            if results.left_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.left_hand_landmarks,
                                       mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.right_hand_landmarks,
                                       mp_holistic.HAND_CONNECTIONS)

            cv2.imshow("mediapipe-camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()