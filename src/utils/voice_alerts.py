"""
Voice announcement system for Anxiety Habits Detector

Provides voice alerts when notifications don't work.
"""

import subprocess
import time
from typing import List
from ..detection.hand_face_detector import AnxietyHabitEvent, HabitType


class VoiceAlertManager:
    """Manages voice announcements for habit detection"""
    
    def __init__(self, enabled: bool = True, cooldown_seconds: int = 30):
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_time = {}
        self.alert_count = 0
        
    def send_voice_alert(self, message: str) -> bool:
        """Send a voice announcement using macOS 'say' command"""
        if not self.enabled:
            return False
        
        try:
            # Use the macOS 'say' command
            result = subprocess.run(['say', message], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print("Voice alert timeout")
            return False
        except Exception as e:
            print(f"Voice alert error: {e}")
            return False
    
    def send_habit_voice_alert(self, habit_event: AnxietyHabitEvent) -> bool:
        """Send voice alert for detected habit"""
        if not self.enabled:
            return False
        
        # Check cooldown (skip if cooldown is 0)
        habit_key = habit_event.habit_type.value
        current_time = time.time()
        
        if self.cooldown_seconds > 0:
            if habit_key in self.last_alert_time:
                time_since_last = current_time - self.last_alert_time[habit_key]
                if time_since_last < self.cooldown_seconds:
                    return False
        
        # Update last alert time
        self.last_alert_time[habit_key] = current_time
        self.alert_count += 1
        
        # Create voice message
        habit_name = habit_event.habit_type.value.replace('_', ' ')
        message = f"Habit detected: {habit_name}. Please stop."
        
        return self.send_voice_alert(message)
    
    def send_summary_voice_alert(self, session_habits: List[AnxietyHabitEvent], session_duration: float) -> bool:
        """Send voice summary at end of session"""
        if not self.enabled or not session_habits:
            return False
        
        total_habits = len(session_habits)
        session_minutes = session_duration / 60
        
        if total_habits == 1:
            message = f"Session complete. One habit detected in {session_minutes:.1f} minutes."
        else:
            message = f"Session complete. {total_habits} habits detected in {session_minutes:.1f} minutes."
        
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
    
    def test_voice_alert(self) -> bool:
        """Test voice alert system"""
        return self.send_voice_alert("Voice alert test. Anxiety habits detector is working.")
    
    def get_stats(self) -> dict:
        """Get voice alert statistics"""
        return {
            "enabled": self.enabled,
            "cooldown_seconds": self.cooldown_seconds,
            "alerts_sent": self.alert_count,
            "last_alerts": self.last_alert_time
        }

