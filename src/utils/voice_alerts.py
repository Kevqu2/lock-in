#!/usr/bin/env python3
"""
Voice Alert System for Anxiety Habits Detector

This module provides voice notifications using macOS native 'say' command.
It's designed to be privacy-friendly and work in the background.
"""

import subprocess
import time
from typing import Optional
from enum import Enum


class HabitType(Enum):
    """Types of anxiety habits that can be detected"""
    NAIL_BITING = "nail_biting"
    HAIR_PULLING = "hair_pulling"
    SKIN_PICKING = "skin_picking"
    FACE_TOUCHING = "face_touching"


class AnxietyHabitEvent:
    """Represents a detected anxiety habit event"""
    
    def __init__(self, habit_type: HabitType, start_time: float, end_time: float,
                 duration: float, confidence: float, hand_involved: str, face_region: str):
        self.habit_type = habit_type
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration
        self.confidence = confidence
        self.hand_involved = hand_involved
        self.face_region = face_region


class VoiceAlertManager:
    """Manages voice alerts for anxiety habits detection"""
    
    def __init__(self, enabled: bool = True, cooldown_seconds: int = 30):
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_time = {}
        
        # Test if 'say' command is available
        if not self._test_say_command():
            print("Warning: 'say' command not available. Voice alerts will be disabled.")
            self.enabled = False
    
    def _test_say_command(self) -> bool:
        """Test if the macOS 'say' command is available"""
        try:
            result = subprocess.run(['say', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def send_voice_alert(self, message: str) -> bool:
        """Send a basic voice alert"""
        if not self.enabled:
            return False
            
        try:
            subprocess.run(['say', message], timeout=10)
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def send_habit_voice_alert(self, habit_event: AnxietyHabitEvent) -> bool:
        """Send a voice alert for a specific habit"""
        if not self.enabled:
            return False
        
        # Check cooldown
        habit_key = habit_event.habit_type.value
        current_time = time.time()
        
        if habit_key in self.last_alert_time:
            time_since_last = current_time - self.last_alert_time[habit_key]
            if time_since_last < self.cooldown_seconds:
                return False
        
        # Create personalized message
        habit_name = habit_event.habit_type.value.replace('_', ' ')
        message = f"Please stop {habit_name}. You've been doing this for {habit_event.duration:.1f} seconds."
        
        success = self.send_voice_alert(message)
        if success:
            self.last_alert_time[habit_key] = current_time
            
        return success
    
    def send_summary_voice_alert(self, total_habits: int, session_duration: float) -> bool:
        """Send a session summary voice alert"""
        if not self.enabled or total_habits == 0:
            return False
            
        message = f"Session complete. You had {total_habits} habit episodes over {session_duration:.1f} minutes."
        return self.send_voice_alert(message)
    
    def enable_voice_alerts(self) -> None:
        """Enable voice alerts"""
        self.enabled = True
    
    def disable_voice_alerts(self) -> None:
        """Disable voice alerts"""
        self.enabled = False
    
    def set_cooldown(self, seconds: int) -> None:
        """Set cooldown period between voice alerts"""
        self.cooldown_seconds = seconds