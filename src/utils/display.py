"""
Display utilities for the Anxiety Habits Detector
"""

import cv2
import time
from typing import Dict, List, Tuple, Optional


class DisplayManager:
    """Manages display overlays and HUD for the anxiety habits detector"""
    
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history = []
    
    def draw_hud(self, frame, fps: float, detection_status: str, 
                recording_status: bool, session_info: Optional[Dict] = None) -> None:
        """Draw the main HUD overlay with clean layout"""
        height, width = frame.shape[:2]
        
        # Define colors (BGR format)
        colors = {
            'background': (20, 20, 20),      # Dark gray background
            'title': (255, 255, 255),        # White title
            'text': (200, 200, 200),         # Light gray text
            'recording': (0, 150, 255),      # Orange for recording
            'good': (0, 200, 0),             # Green for good status
            'warning': (0, 200, 255),        # Yellow for warning
            'error': (0, 0, 255),            # Red for error
            'accent': (255, 100, 0)          # Blue accent
        }
        
        # HUD dimensions and positioning
        hud_width = 380
        hud_height = 120
        hud_x = 15
        hud_y = 15
        
        # Draw main HUD background with border
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_width, hud_y + hud_height), 
                     colors['background'], -1)
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_width, hud_y + hud_height), 
                     colors['accent'], 2)
        
        # Title section
        cv2.putText(frame, "ANXIETY HABITS DETECTOR", (hud_x + 15, hud_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['title'], 2)
        
        # Status indicators section
        y_offset = hud_y + 45
        line_height = 18
        
        # FPS indicator
        if fps > 25:
            fps_color = colors['good']
        elif fps > 15:
            fps_color = colors['warning']
        else:
            fps_color = colors['error']
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (hud_x + 15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
        
        # Detection status
        y_offset += line_height
        if detection_status and detection_status != "No detections":
            status_color = colors['good']
            status_text = f"Monitoring: {detection_status}"
        else:
            status_color = colors['text']
            status_text = "Waiting for detection..."
        
        cv2.putText(frame, status_text, (hud_x + 15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Recording status
        y_offset += line_height
        if recording_status and session_info:
            cv2.putText(frame, f"● RECORDING: {session_info['session_id'][:15]}...", 
                       (hud_x + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['recording'], 1)
        else:
            cv2.putText(frame, "○ Not recording", (hud_x + 15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1)
        
        # Instructions section (bottom right)
        instructions = "Controls: 'r'=record, 'q'=quit, 'h'=help"
        text_size = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.putText(frame, instructions, (width - text_size[0] - 15, height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['text'], 1)
    
    def draw_habit_alert(self, frame, habit_events: List, position: Tuple[int, int] = None) -> None:
        """Draw habit detection alerts"""
        if not habit_events:
            return
        
        height, width = frame.shape[:2]
        
        # Default position if not specified
        if position is None:
            position = (width // 2 - 150, height // 2 - 50)
        
        x, y = position
        
        # Draw alert background
        cv2.rectangle(frame, (x-10, y-10), (x+300, y+80), (0, 0, 100), -1)
        cv2.rectangle(frame, (x-10, y-10), (x+300, y+80), (0, 0, 255), 3)
        
        # Draw alert text
        cv2.putText(frame, "HABIT DETECTED!", (x, y+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
        # List detected habits
        for i, event in enumerate(habit_events[:3]):  # Show max 3 habits
            habit_text = f"- {event.habit_type.value.replace('_', ' ').title()}"
            cv2.putText(frame, habit_text, (x, y+45+i*15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def draw_proximity_overlay(self, frame, proximity_data: Dict, positions: Dict[str, Tuple[int, int]]) -> None:
        """Draw proximity overlays for hands with smart positioning"""
        height, width = frame.shape[:2]
        
        # Smart positioning to avoid overlap
        overlay_positions = {}
        y_base = 150  # Start below HUD
        
        if 'left' in proximity_data and proximity_data['left']:
            overlay_positions['left'] = (20, y_base)
            y_base += 80  # Space for next overlay
        
        if 'right' in proximity_data and proximity_data['right']:
            overlay_positions['right'] = (width - 220, y_base)
        
        # Draw overlays with smart positioning
        for hand_type, proximity in proximity_data.items():
            if proximity and hand_type in overlay_positions:
                self._draw_single_proximity_overlay(frame, proximity, overlay_positions[hand_type])
    
    def _draw_single_proximity_overlay(self, frame, proximity, position: Tuple[int, int]) -> None:
        """Draw a single proximity overlay with clean design"""
        x, y = position
        
        # Define colors
        colors = {
            'background': (15, 15, 15),      # Dark background
            'border': (60, 60, 60),          # Gray border
            'text': (220, 220, 220),         # Light text
            'safe': (0, 180, 0),             # Green for safe
            'warning': (0, 150, 255),        # Orange for warning
            'danger': (0, 0, 255),           # Red for danger
        }
        
        # Determine status and colors
        if proximity.is_close:
            status_color = colors['danger']
            status_text = "HABIT DETECTED"
            bg_color = (0, 0, 50)  # Dark red background
        elif proximity.normalized_distance < 0.25:
            status_color = colors['warning']
            status_text = "CLOSE"
            bg_color = (0, 50, 50)  # Dark orange background
        else:
            status_color = colors['safe']
            status_text = "SAFE"
            bg_color = colors['background']
        
        # Overlay dimensions
        overlay_width = 200
        overlay_height = 70
        
        # Draw background with border
        cv2.rectangle(frame, (x, y), (x + overlay_width, y + overlay_height), 
                     bg_color, -1)
        cv2.rectangle(frame, (x, y), (x + overlay_width, y + overlay_height), 
                     status_color, 2)
        
        # Draw content
        padding = 8
        
        # Hand label
        hand_text = f"{proximity.hand_type.upper()} HAND"
        cv2.putText(frame, hand_text, (x + padding, y + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1)
        
        # Distance info
        distance_text = f"Dist: {proximity.normalized_distance:.2f}"
        cv2.putText(frame, distance_text, (x + padding, y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['text'], 1)
        
        # Region info
        region_text = f"Region: {proximity.face_region}"
        cv2.putText(frame, region_text, (x + padding, y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['text'], 1)
        
        # Status indicator
        status_x = x + overlay_width - 100
        cv2.putText(frame, status_text, (status_x, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
    
    def draw_landmarks(self, frame, results, mp_holistic, mp_drawing) -> None:
        """Draw MediaPipe landmarks with clean, professional styling"""
        # Define landmark colors (BGR format)
        colors = {
            'face': (0, 255, 150),      # Teal for face
            'left_hand': (255, 100, 0), # Orange for left hand
            'right_hand': (0, 150, 255) # Blue for right hand
        }
        
        # Draw face landmarks with subtle styling
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=colors['face'], thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=colors['face'], thickness=1)
            )
        
        # Draw left hand landmarks
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=colors['left_hand'], thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=colors['left_hand'], thickness=2)
            )
        
        # Draw right hand landmarks
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=colors['right_hand'], thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=colors['right_hand'], thickness=2)
            )
