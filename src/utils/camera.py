"""
Camera utilities for the Anxiety Habits Detector
"""

import cv2
from typing import Optional


def open_camera(camera_index: int = 0) -> Optional[cv2.VideoCapture]:
    """Open camera with fallback options"""
    # Try macOS AVFoundation first
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        return cap
    
    # Fallback to default backend
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        return cap
    
    # Try other camera indices
    for idx in [1, 2, -1]:
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            return cap
    
    return None


def setup_camera(cap: cv2.VideoCapture, width: int = 640, height: int = 480) -> bool:
    """Setup camera properties"""
    if not cap.isOpened():
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    return True


def get_camera_info(cap: cv2.VideoCapture) -> dict:
    """Get camera information"""
    if not cap.isOpened():
        return {}
    
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "backend": cap.getBackendName()
    }

