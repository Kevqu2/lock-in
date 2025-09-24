#!/usr/bin/env python3
"""
Hand-Face Proximity Detector for Anxiety Habits

This module detects when hands are close to the face and classifies
the type of anxiety habit being performed.
"""

import numpy as np
import math
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class HabitType(Enum):
    """Types of anxiety habits that can be detected"""
    NONE = "none"
    NAIL_BITING = "nail_biting"
    HAIR_PULLING = "hair_pulling"
    SKIN_PICKING = "skin_picking"
    FACE_TOUCHING = "face_touching"


@dataclass
class HandFaceProximity:
    """Data structure for hand-face proximity information"""
    hand_type: str
    distance: float
    normalized_distance: float
    face_region: str
    is_close: bool
    confidence: float


@dataclass
class AnxietyHabitEvent:
    """Represents a detected anxiety habit event"""
    habit_type: HabitType
    start_time: float
    end_time: float
    duration: float
    confidence: float
    hand_involved: str
    face_region: str


class HandFaceDetector:
    """Detects hand-to-face proximity and classifies anxiety habits"""
    
    def __init__(self):
        # Thresholds for habit detection
        self.close_proximity_threshold = 0.15  # Normalized distance
        self.habit_duration_threshold = 3.0  # seconds (increased to avoid false positives)
        self.confidence_threshold = 0.7  # Restored confidence threshold
        
        # State tracking
        self.current_habits = {}  # habit_key -> AnxietyHabitEvent
        self.habit_history = []
        
        # Face regions for habit classification
        self.face_regions = {
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'chin': [18, 175, 199, 175, 199, 175, 199, 175, 199],
            'left_cheek': [116, 117, 118, 119, 120, 121, 126, 142, 36, 205],
            'right_cheek': [345, 346, 347, 348, 349, 350, 355, 371, 266, 425],
            'nose': [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 360, 279, 331, 294, 460, 328, 459, 457, 438, 439, 278, 344, 440, 275, 4, 45, 51, 134, 102, 49, 220, 305, 281, 360, 279, 331, 294, 460, 328, 459, 457, 438, 439, 278, 344, 440, 275],
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'forehead': [10, 151, 9, 8, 107, 55, 65, 52, 53, 46]
        }
    
    def calculate_hand_face_proximity(self, hand_landmarks, face_landmarks, 
                                    frame_width: int, frame_height: int,
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
    
    def _process_habit_detection(self, hand: str, habit_type: HabitType, 
                               proximity: HandFaceProximity, timestamp: float) -> None:
        """Process habit detection and manage habit events"""
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
    
    def detect_habits(self, results, frame_width: int, frame_height: int) -> list:
        """Main method to detect habits from MediaPipe results"""
        timestamp = time.time()
        detected_events = []
        
        # Process left hand
        if results.left_hand_landmarks and results.face_landmarks:
            left_proximity = self.calculate_hand_face_proximity(
                results.left_hand_landmarks, results.face_landmarks, 
                frame_width, frame_height, 'left'
            )
            if left_proximity:
                habit_type = self.classify_habit_type(left_proximity)
                self._process_habit_detection('left', habit_type, left_proximity, timestamp)
        
        # Process right hand
        if results.right_hand_landmarks and results.face_landmarks:
            right_proximity = self.calculate_hand_face_proximity(
                results.right_hand_landmarks, results.face_landmarks, 
                frame_width, frame_height, 'right'
            )
            if right_proximity:
                habit_type = self.classify_habit_type(right_proximity)
                self._process_habit_detection('right', habit_type, right_proximity, timestamp)
        
        # Return any newly completed habits
        recent_habits = [h for h in self.habit_history if h.end_time > timestamp - 1.0]
        self.habit_history = [h for h in self.habit_history if h.end_time <= timestamp - 1.0]
        
        return recent_habits