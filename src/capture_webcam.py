# --------- capture_webcam.py (minimal) ----------
# PURPOSE: Open your webcam and show the live feed. Press 'q' to quit.

import cv2 # OpenCV gives us access to the camera and display

def open_camera():
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        return cap
    
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        return cap
    
    return None

def main():
    cap = open_camera()
    if cap is None:
        print("ERROR: Could not open the webcam.")
        return
    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read() # grabs one frame from the camera
        if not ok:
            print("ERROR: fram grab failed.")
            break

        cv2.imshow("webcam", frame)

        # wait 1 ms for a key; if its 'q', exit the loop and close the camera
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
