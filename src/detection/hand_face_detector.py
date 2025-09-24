"""
Hand-to-Face Anxiety Habits Detector

Detects anxiety-related habits like nail-biting, hair-pulling, face-touching, 
and skin-picking by analyzing proximity between hands and face landmarks.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class HabitType(Enum):
    """Types of anxiety habits that can be detected"""
    NONE = "none"
    FACE_TOUCHING = "face_touching"
    HAIR_PULLING = "hair_pulling"
    NAIL_BITING = "nail_biting"
    SKIN_PICKING = "skin_picking"


@dataclass
class HandFaceProximity:
    """Data structure for hand-face proximity analysis"""
    hand_type: str  # 'left' or 'right'
    distance: float  # Distance in pixels
    normalized_distance: float  # Normalized distance (0-1)
    face_region: str  # Which part of face is closest
    is_close: bool  # Whether hand is close enough to be touching
    confidence: float  # Confidence in the detection


@dataclass
class AnxietyHabitEvent:
    """Data structure for detected anxiety habit events"""
    habit_type: HabitType
    start_time: float
    end_time: float
    duration: float
    confidence: float
    hand_involved: str  # 'left', 'right', or 'both'
    face_region: str


class HandFaceDetector:
    """Detects hand-to-face proximity and anxiety habits"""
    
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Face regions for more specific detection
        self.face_regions = {
            'forehead': [10, 151, 9, 10],
            'left_eye': [33, 7, 163, 144],
            'right_eye': [362, 382, 381, 380],
            'nose': [1, 2, 5, 4],
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375],
            'chin': [18, 200, 199, 175],
            'left_cheek': [116, 117, 118, 119, 120],
            'right_cheek': [345, 346, 347, 348, 349]
        }
        
        # Thresholds for habit detection
        self.close_proximity_threshold = 0.15  # Normalized distance
        self.habit_duration_threshold = 3.0  # seconds (increased to avoid false positives)
        self.confidence_threshold = 0.7  # Restored confidence threshold
        
        # State tracking
        self.current_habits: Dict[str, AnxietyHabitEvent] = {}
        self.habit_history: List[AnxietyHabitEvent] = []
    
    def calculate_hand_face_proximity(self, 
                                    hand_landmarks, 
                                    face_landmarks, 
                                    frame_width: int, 
                                    frame_height: int,
                                    hand_type: str) -> Optional[HandFaceProximity]:
        """Calculate proximity between hand and face landmarks"""
        if not hand_landmarks or not face_landmarks:
            return None
        
        # Get hand center
        hand_x = np.mean([lm.x for lm in hand_landmarks.landmark])
        hand_y = np.mean([lm.y for lm in hand_landmarks.landmark])
        
        # Calculate distances to different face regions
        min_distance = float('inf')
        closest_region = 'unknown'
        
        for region_name, landmark_indices in self.face_regions.items():
            region_points = []
            for idx in landmark_indices:
                if idx < len(face_landmarks.landmark):
                    lm = face_landmarks.landmark[idx]
                    region_points.append([lm.x * frame_width, lm.y * frame_height])
            
            if region_points:
                region_center = np.mean(region_points, axis=0)
                distance = np.sqrt((hand_x * frame_width - region_center[0])**2 + 
                                 (hand_y * frame_height - region_center[1])**2)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_region = region_name
        
        # Normalize distance by frame diagonal
        frame_diagonal = np.sqrt(frame_width**2 + frame_height**2)
        normalized_distance = min_distance / frame_diagonal
        
        # Determine if hand is close enough to be touching
        is_close = normalized_distance < self.close_proximity_threshold
        
        return HandFaceProximity(
            hand_type=hand_type,
            distance=min_distance,
            normalized_distance=normalized_distance,
            face_region=closest_region,
            is_close=is_close,
            confidence=1.0 - min(normalized_distance, 1.0)
        )
    
    def classify_habit_type(self, proximity: HandFaceProximity) -> HabitType:
        """Classify the type of anxiety habit based on proximity data"""
        if not proximity.is_close:
            return HabitType.NONE
        
        # Classify based on face region and hand proximity
        if proximity.face_region in ['mouth', 'chin']:
            return HabitType.NAIL_BITING
        elif proximity.face_region in ['forehead', 'left_eye', 'right_eye']:
            return HabitType.HAIR_PULLING
        elif proximity.face_region in ['left_cheek', 'right_cheek', 'nose']:
            return HabitType.SKIN_PICKING
        else:
            return HabitType.FACE_TOUCHING
    
    def detect_habits(self, results, frame_width: int, frame_height: int, timestamp: float) -> List[AnxietyHabitEvent]:
        """Detect anxiety habits from MediaPipe results"""
        detected_events = []
        
        # Check left hand proximity
        if results.left_hand_landmarks and results.face_landmarks:
            left_proximity = self.calculate_hand_face_proximity(
                results.left_hand_landmarks, results.face_landmarks, 
                frame_width, frame_height, 'left'
            )
            
            if left_proximity:
                habit_type = self.classify_habit_type(left_proximity)
                self._process_habit_detection('left', habit_type, left_proximity, timestamp)
        
        # Check right hand proximity
        if results.right_hand_landmarks and results.face_landmarks:
            right_proximity = self.calculate_hand_face_proximity(
                results.right_hand_landmarks, results.face_landmarks, 
                frame_width, frame_height, 'right'
            )
            
            if right_proximity:
                habit_type = self.classify_habit_type(right_proximity)
                self._process_habit_detection('right', habit_type, right_proximity, timestamp)
        
        # Return completed events
        completed_events = self._get_completed_events()
        return completed_events
    
    def _process_habit_detection(self, hand: str, habit_type: HabitType, proximity: HandFaceProximity, timestamp: float):
        """Process habit detection and update state"""
        habit_key = f"{hand}_{habit_type.value}"
        
        if habit_type != HabitType.NONE and proximity.is_close:
            # Start or continue habit
            if habit_key not in self.current_habits:
                print(f"🎯 STARTING HABIT: {habit_type.value} ({hand} hand)")
                self.current_habits[habit_key] = AnxietyHabitEvent(
                    habit_type=habit_type,
                    start_time=timestamp,
                    end_time=timestamp,
                    duration=0.0,
                    confidence=proximity.confidence,
                    hand_involved=hand,
                    face_region=proximity.face_region
                )
            else:
                # Update existing habit
                self.current_habits[habit_key].end_time = timestamp
                self.current_habits[habit_key].duration = timestamp - self.current_habits[habit_key].start_time
                self.current_habits[habit_key].confidence = max(
                    self.current_habits[habit_key].confidence, 
                    proximity.confidence
                )
                # Show progress every 0.2 seconds and complete habit if duration threshold reached
                if int(self.current_habits[habit_key].duration * 5) != int((self.current_habits[habit_key].duration - 0.1) * 5):
                    print(f"⏱️  HABIT PROGRESS: {habit_type.value} ({self.current_habits[habit_key].duration:.1f}s/{self.habit_duration_threshold:.1f}s)")
                
                # Complete habit if duration threshold is reached (even while ongoing)
                if self.current_habits[habit_key].duration >= self.habit_duration_threshold:
                    print(f"✅ HABIT COMPLETED: {habit_type.value} ({self.current_habits[habit_key].duration:.1f}s)")
                    self.habit_history.append(self.current_habits[habit_key])
                    del self.current_habits[habit_key]
        else:
            # End habit if it exists and meets duration threshold
            if habit_key in self.current_habits:
                event = self.current_habits[habit_key]
                if event.duration >= self.habit_duration_threshold:
                    print(f"✅ HABIT COMPLETED: {event.habit_type.value} ({event.duration:.1f}s)")
                    self.habit_history.append(event)
                else:
                    print(f"❌ HABIT CANCELLED: {event.habit_type.value} ({event.duration:.1f}s < {self.habit_duration_threshold:.1f}s)")
                del self.current_habits[habit_key]
    
    def _get_completed_events(self) -> List[AnxietyHabitEvent]:
        """Get and clear completed habit events"""
        events = self.habit_history.copy()
        self.habit_history.clear()
        return events
    
    def get_current_habits(self) -> Dict[str, AnxietyHabitEvent]:
        """Get currently active habits"""
        return self.current_habits.copy()
    
    def draw_proximity_overlay(self, frame, proximity: HandFaceProximity, position: Tuple[int, int]):
        """Draw proximity information overlay on frame"""
        if not proximity:
            return frame
        
        x, y = position
        text = f"{proximity.hand_type.upper()} HAND"
        distance_text = f"Distance: {proximity.distance:.1f}px"
        region_text = f"Region: {proximity.face_region}"
        
        # Color based on proximity
        if proximity.is_close:
            color = (0, 0, 255)  # Red for close
            alert_text = "HABIT DETECTED!"
        else:
            color = (0, 255, 0)  # Green for safe distance
            alert_text = "SAFE"
        
        # Draw background
        cv2.rectangle(frame, (x-10, y-60), (x+300, y+10), (0, 0, 0), -1)
        cv2.rectangle(frame, (x-10, y-60), (x+300, y+10), color, 2)
        
        # Draw text
        cv2.putText(frame, text, (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, distance_text, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, region_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw alert if close
        if proximity.is_close:
            cv2.putText(frame, alert_text, (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame

