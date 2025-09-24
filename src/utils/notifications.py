"""
Notification system for Anxiety Habits Detector

Provides system notifications for habit detection events.
"""

import time
from typing import List, Dict, Optional
from plyer import notification
from ..detection.hand_face_detector import AnxietyHabitEvent, HabitType


class NotificationManager:
    """Manages system notifications for habit detection"""
    
    def __init__(self, enabled: bool = True, cooldown_seconds: int = 30):
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.last_notification_time = {}
        self.notification_count = 0
        
        # Notification settings
        self.app_name = "Anxiety Habits Detector"
        self.app_icon = None  # Could add an icon file path here
        
    def send_habit_notification(self, habit_event: AnxietyHabitEvent) -> bool:
        """Send a notification for a detected habit"""
        if not self.enabled:
            return False
        
        # Check cooldown to avoid spam
        habit_key = habit_event.habit_type.value
        current_time = time.time()
        
        if habit_key in self.last_notification_time:
            time_since_last = current_time - self.last_notification_time[habit_key]
            if time_since_last < self.cooldown_seconds:
                return False  # Still in cooldown period
        
        # Update last notification time
        self.last_notification_time[habit_key] = current_time
        self.notification_count += 1
        
        # Prepare notification content
        title = self._get_notification_title(habit_event)
        message = self._get_notification_message(habit_event)
        
        try:
            notification.notify(
                title=title,
                message=message,
                app_name=self.app_name,
                app_icon=self.app_icon,
                timeout=10,  # Notification duration in seconds
                toast=False  # Use system notification instead of toast
            )
            return True
        except Exception as e:
            print(f"Failed to send notification: {e}")
            return False
    
    def _get_notification_title(self, habit_event: AnxietyHabitEvent) -> str:
        """Get notification title based on habit type"""
        titles = {
            HabitType.NAIL_BITING: "🦷 Nail Biting Detected",
            HabitType.HAIR_PULLING: "💇 Hair Pulling Detected", 
            HabitType.FACE_TOUCHING: "🤚 Face Touching Detected",
            HabitType.SKIN_PICKING: "👆 Skin Picking Detected"
        }
        return titles.get(habit_event.habit_type, "⚠️ Habit Detected")
    
    def _get_notification_message(self, habit_event: AnxietyHabitEvent) -> str:
        """Get notification message with helpful information"""
        duration_text = f"{habit_event.duration:.1f}s" if habit_event.duration > 0 else "ongoing"
        confidence_text = f"{habit_event.confidence*100:.0f}% confidence"
        
        # Add helpful suggestions based on habit type
        suggestions = {
            HabitType.NAIL_BITING: "Try using fidget toys or stress balls",
            HabitType.HAIR_PULLING: "Consider wearing a hat or using hair accessories",
            HabitType.FACE_TOUCHING: "Keep hands busy with other activities",
            HabitType.SKIN_PICKING: "Apply lotion or use gentle skin care products"
        }
        
        suggestion = suggestions.get(habit_event.habit_type, "Take a deep breath and relax")
        
        return f"Duration: {duration_text} • {confidence_text}\n\n💡 Suggestion: {suggestion}"
    
    def send_summary_notification(self, session_habits: List[AnxietyHabitEvent], session_duration: float) -> bool:
        """Send a summary notification at the end of a session"""
        if not self.enabled or not session_habits:
            return False
        
        # Count habits by type
        habit_counts = {}
        for habit in session_habits:
            habit_type = habit.habit_type.value
            habit_counts[habit_type] = habit_counts.get(habit_type, 0) + 1
        
        # Create summary message
        total_habits = len(session_habits)
        session_minutes = session_duration / 60
        
        if total_habits == 1:
            title = "📊 Session Complete - 1 Habit Detected"
        else:
            title = f"📊 Session Complete - {total_habits} Habits Detected"
        
        message = f"Session duration: {session_minutes:.1f} minutes\n\n"
        
        # Add breakdown by habit type
        for habit_type, count in habit_counts.items():
            message += f"• {habit_type.replace('_', ' ').title()}: {count}\n"
        
        message += f"\nTotal notifications sent: {self.notification_count}"
        
        try:
            notification.notify(
                title=title,
                message=message,
                app_name=self.app_name,
                app_icon=self.app_icon,
                timeout=15,
                toast=False
            )
            return True
        except Exception as e:
            print(f"Failed to send summary notification: {e}")
            return False
    
    def enable_notifications(self) -> None:
        """Enable notifications"""
        self.enabled = True
    
    def disable_notifications(self) -> None:
        """Disable notifications"""
        self.enabled = False
    
    def set_cooldown(self, seconds: int) -> None:
        """Set notification cooldown period"""
        self.cooldown_seconds = max(5, seconds)  # Minimum 5 seconds
    
    def get_stats(self) -> Dict:
        """Get notification statistics"""
        return {
            "enabled": self.enabled,
            "cooldown_seconds": self.cooldown_seconds,
            "notifications_sent": self.notification_count,
            "last_notifications": self.last_notification_time
        }
    
    def reset_stats(self) -> None:
        """Reset notification statistics"""
        self.notification_count = 0
        self.last_notification_time = {}


class BackgroundNotificationManager(NotificationManager):
    """Enhanced notification manager for background operation"""
    
    def __init__(self, enabled: bool = True, cooldown_seconds: int = 45):
        super().__init__(enabled, cooldown_seconds)
        self.background_mode = True
        self.quiet_hours = (22, 8)  # 10 PM to 8 AM
        self.weekend_mode = False
        
    def send_habit_notification(self, habit_event: AnxietyHabitEvent) -> bool:
        """Send notification with background mode considerations"""
        if not self._should_notify():
            return False
        
        return super().send_habit_notification(habit_event)
    
    def _should_notify(self) -> bool:
        """Check if notifications should be sent based on background settings"""
        if not self.enabled:
            return False
        
        current_hour = time.localtime().tm_hour
        is_quiet_time = self.quiet_hours[0] <= current_hour or current_hour < self.quiet_hours[1]
        
        # Skip notifications during quiet hours unless it's a critical habit
        if is_quiet_time:
            return False
        
        return True
    
    def set_quiet_hours(self, start_hour: int, end_hour: int) -> None:
        """Set quiet hours for notifications"""
        self.quiet_hours = (start_hour, end_hour)
    
    def enable_weekend_mode(self, enabled: bool = True) -> None:
        """Enable/disable weekend-specific notification settings"""
        self.weekend_mode = enabled

