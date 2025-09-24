"""
Background Display Manager for Anxiety Habits Detector

Minimal display interface optimized for background operation.
"""

import cv2
import time
from typing import Dict, List, Tuple, Optional
from .display import DisplayManager


class BackgroundDisplayManager(DisplayManager):
    """Minimal display manager for background operation"""
    
    def __init__(self, show_minimal_ui: bool = True):
        super().__init__()
        self.show_minimal_ui = show_minimal_ui
        self.last_habit_time = 0
        self.habit_flash_duration = 2.0  # seconds
        
    def draw_minimal_hud(self, frame, fps: float, recording_status: bool, 
                        session_info: Optional[Dict] = None) -> None:
        """Draw minimal HUD for background operation"""
        if not self.show_minimal_ui:
            return
            
        height, width = frame.shape[:2]
        
        # Minimal color scheme
        colors = {
            'background': (10, 10, 10),      # Very dark background
            'text': (150, 150, 150),         # Muted text
            'recording': (0, 100, 200),      # Subtle blue for recording
            'good': (0, 150, 0),             # Green for good status
            'warning': (0, 100, 200),        # Blue for warnings
        }
        
        # Very small HUD in top-right corner
        hud_width = 200
        hud_height = 60
        hud_x = width - hud_width - 10
        hud_y = 10
        
        # Draw minimal background
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_width, hud_y + hud_height), 
                     colors['background'], -1)
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_width, hud_y + hud_height), 
                     colors['text'], 1)
        
        # App name (small)
        cv2.putText(frame, "HABIT DETECTOR", (hud_x + 5, hud_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['text'], 1)
        
        # FPS (very small)
        fps_color = colors['good'] if fps > 15 else colors['warning']
        cv2.putText(frame, f"FPS: {fps:.0f}", (hud_x + 5, hud_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, fps_color, 1)
        
        # Recording status
        if recording_status:
            cv2.putText(frame, "● REC", (hud_x + 5, hud_y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors['recording'], 1)
        else:
            cv2.putText(frame, "○ IDLE", (hud_x + 5, hud_y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors['text'], 1)
        
        # Minimal instructions
        cv2.putText(frame, "'r'=rec 'q'=quit", (width - 120, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors['text'], 1)
    
    def draw_habit_flash(self, frame, habit_events: List) -> None:
        """Draw a subtle flash when habits are detected"""
        if not habit_events:
            return
        
        current_time = time.time()
        
        # Only flash for a short duration after habit detection
        if current_time - self.last_habit_time < self.habit_flash_duration:
            height, width = frame.shape[:2]
            
            # Draw a subtle border flash
            flash_color = (0, 0, 255)  # Red flash
            thickness = 3
            
            # Fade effect based on time elapsed
            time_elapsed = current_time - self.last_habit_time
            fade_factor = 1.0 - (time_elapsed / self.habit_flash_duration)
            alpha = int(255 * fade_factor)
            
            # Draw flashing border
            cv2.rectangle(frame, (0, 0), (width-1, height-1), flash_color, thickness)
            
            # Add subtle corner indicators
            corner_size = 20
            cv2.rectangle(frame, (0, 0), (corner_size, corner_size), flash_color, -1)
            cv2.rectangle(frame, (width-corner_size, 0), (width, corner_size), flash_color, -1)
            cv2.rectangle(frame, (0, height-corner_size), (corner_size, height), flash_color, -1)
            cv2.rectangle(frame, (width-corner_size, height-corner_size), (width, height), flash_color, -1)
    
    def update_habit_flash(self, habit_events: List) -> None:
        """Update the habit flash timer"""
        if habit_events:
            self.last_habit_time = time.time()
    
    def draw_subtle_proximity_indicators(self, frame, proximity_data: Dict) -> None:
        """Draw very subtle proximity indicators"""
        if not proximity_data:
            return
        
        height, width = frame.shape[:2]
        
        # Small corner indicators instead of full overlays
        indicator_size = 8
        corner_offset = 15
        
        for hand_type, proximity in proximity_data.items():
            if not proximity:
                continue
            
            # Determine color based on proximity
            if proximity.is_close:
                color = (0, 0, 255)  # Red for close
            elif proximity.normalized_distance < 0.25:
                color = (0, 150, 255)  # Orange for close
            else:
                color = (0, 200, 0)  # Green for safe
            
            # Position indicators in corners
            if hand_type == 'left':
                x, y = corner_offset, height - corner_offset - indicator_size
            else:  # right hand
                x, y = width - corner_offset - indicator_size, height - corner_offset - indicator_size
            
            # Draw small indicator
            cv2.circle(frame, (x + indicator_size//2, y + indicator_size//2), 
                      indicator_size//2, color, -1)
    
    def draw_landmarks_subtle(self, frame, results, mp_holistic, mp_drawing) -> None:
        """Draw landmarks with very subtle styling for background mode"""
        if not self.show_minimal_ui:
            return
        
        # Very subtle colors for background operation
        colors = {
            'face': (0, 100, 100),        # Very muted teal
            'left_hand': (100, 50, 0),    # Very muted orange
            'right_hand': (0, 50, 100)    # Very muted blue
        }
        
        # Draw face landmarks (very subtle)
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=colors['face'], thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=colors['face'], thickness=1)
            )
        
        # Draw hand landmarks (subtle)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=colors['left_hand'], thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=colors['left_hand'], thickness=1)
            )
        
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=colors['right_hand'], thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=colors['right_hand'], thickness=1)
            )

