#!/usr/bin/env python3
import os
import sys
import time
import json
import signal
from typing import Dict, Any, List, Tuple

import numpy as np
import cv2
import click

try:
    import orjson
    dumps = orjson.dumps  # type: ignore
except Exception:  # pragma: no cover
    dumps = lambda obj: json.dumps(obj, separators=(",", ":")).encode("utf-8")

try:
    import mediapipe as mp
except Exception as exc:
    print("ERROR: mediapipe is required. Install with `pip install mediapipe`.", file=sys.stderr)
    raise


def _serialize_landmarks(landmarks) -> List[Dict[str, float]]:
    serialized: List[Dict[str, float]] = []
    if landmarks is None:
        return serialized
    for lm in landmarks:
        serialized.append({
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(getattr(lm, "z", 0.0)),
            "visibility": float(getattr(lm, "visibility", 0.0)),
        })
    return serialized


def _make_session_paths(output_dir: str, session_name: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, f"{session_name}.jsonl")
    meta_path = os.path.join(output_dir, f"{session_name}.meta.json")
    return jsonl_path, meta_path


def _write_meta(meta_path: str, meta: Dict[str, Any]) -> None:
    tmp = meta_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(dumps(meta))
    os.replace(tmp, meta_path)


class GracefulKiller:
    def __init__(self):
        self.received_signal = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):  # type: ignore[no-untyped-def]
        self.received_signal = True


@click.command()
@click.option("--camera-index", "camera_index", default=0, show_default=True, type=int, help="OpenCV camera index")
@click.option("--session-name", "session_name", default=None, help="Session name; default is timestamp")
@click.option("--label", "label", default=None, help="Optional behavior label for the entire session (e.g., nail_biting)")
@click.option("--note", "note", default=None, help="Free-form note to store in session metadata")
@click.option("--output-dir", "output_dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "raw"), show_default=True, help="Directory for JSONL output")
@click.option("--draw", is_flag=True, help="Render landmarks on preview window")
@click.option("--no-window", is_flag=True, help="Do not open a preview window (headless)")
@click.option("--max-seconds", type=float, default=0, show_default=True, help="Auto-stop after N seconds; 0 means unlimited")
def main(camera_index: int, session_name: str | None, label: str | None, note: str | None, output_dir: str, draw: bool, no_window: bool, max_seconds: float) -> None:
    start_unix = time.time()
    if session_name is None:
        session_name = time.strftime("%Y%m%d-%H%M%S")

    # Resolve output directory relative to repo root if given like ../data/raw
    output_dir = os.path.abspath(output_dir)
    jsonl_path, meta_path = _make_session_paths(output_dir, session_name)

    # Initialize MediaPipe solutions
    mp_face = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    drawing_utils = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {camera_index}", file=sys.stderr)
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    meta: Dict[str, Any] = {
        "session_name": session_name,
        "created_unix": start_unix,
        "camera_index": camera_index,
        "frame_width": width,
        "frame_height": height,
        "fps": fps,
        "notes": note or "landmarks-only capture; no raw frames are stored",
        "label": label,
        "schema": {
            "frame": {
                "t": "float seconds since start",
                "face": "MediaPipe FaceMesh 468+ landmarks x,y,z,visibility",
                "hands": "MediaPipe Hands multi-hand landmarks x,y,z"
            }
        }
    }
    _write_meta(meta_path, meta)

    killer = GracefulKiller()
    last_meta_write = time.time()
    start_time_monotonic = time.monotonic()

    with open(jsonl_path, "ab", buffering=1024 * 1024) as out:
        while True:
            if killer.received_signal:
                break
            if max_seconds and (time.monotonic() - start_time_monotonic) >= max_seconds:
                break

            ret, frame = cap.read()
            if not ret:
                print("WARN: failed to read frame; stopping", file=sys.stderr)
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_result = face_mesh.process(rgb)
            hands_result = hands.process(rgb)

            now = time.monotonic()
            t_rel = float(now - start_time_monotonic)

            face_landmarks = None
            if face_result.multi_face_landmarks:
                face_landmarks = _serialize_landmarks(face_result.multi_face_landmarks[0].landmark)

            hands_all: List[List[Dict[str, float]]] = []
            if hands_result.multi_hand_landmarks:
                for hand_lms in hands_result.multi_hand_landmarks:
                    hands_all.append(_serialize_landmarks(hand_lms.landmark))

            record = {
                "t": t_rel,
                "face": face_landmarks,
                "hands": hands_all,
            }
            out.write(dumps(record) + b"\n")

            if not no_window:
                display = frame.copy()
                if draw:
                    if face_result.multi_face_landmarks:
                        for face_lms in face_result.multi_face_landmarks:
                            drawing_utils.draw_landmarks(
                                image=display,
                                landmark_list=face_lms,
                                connections=mp_face.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
                            )
                    if hands_result.multi_hand_landmarks:
                        for hand_lms in hands_result.multi_hand_landmarks:
                            drawing_utils.draw_landmarks(
                                display,
                                hand_lms,
                                mp_hands.HAND_CONNECTIONS,
                                drawing_styles.get_default_hand_landmarks_style(),
                                drawing_styles.get_default_hand_connections_style(),
                            )
                cv2.imshow("Landmark Capture (q to quit)", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            # Refresh meta periodically in case properties change
            if (time.time() - last_meta_write) > 10:
                last_meta_write = time.time()
                _write_meta(meta_path, meta)

    cap.release()
    if not no_window:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

