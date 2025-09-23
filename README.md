Anxiety Habits Detector

A personal prototype to detect anxiety-related habits (like nail-biting, hair-pulling, face-touching, and skin-picking) using AI and real-time camera landmarks.
The system runs locally, processes face and hand landmarks (no raw video stored), and uses a lightweight ML model to classify short movements as possible habits.

Features

Capture webcam sessions with MediaPipe landmarks (hands + face)

Save sessions as landmarks only (NDJSON format for privacy)

Label events (start/end times with habit type)

Convert sessions into training data with simple features

Train baseline models (MLP, GRU, or TCN) on personal data

Run live inference with habit alerts
