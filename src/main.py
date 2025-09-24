"""
Main application for Anxiety Habits Detector

A personal prototype to detect anxiety-related habits using AI and real-time camera landmarks.
"""

import cv2
import mediapipe as mp
import time
import argparse
import sys
from typing import Optional

from .detection.hand_face_detector import HandFaceDetector, AnxietyHabitEvent
from .recording.landmark_recorder import LandmarkRecorder
from .utils.camera import open_camera, setup_camera, get_camera_info
from .utils.display import DisplayManager
from .utils.background_display import BackgroundDisplayManager
from .utils.voice_alerts import VoiceAlertManager


class AnxietyHabitsDetector:
    """Main application class for the Anxiety Habits Detector"""
    
    def __init__(self, output_dir: str = "sessions", background_mode: bool = True):
        # Initialize components
        self.detector = HandFaceDetector()
        self.recorder = LandmarkRecorder(output_dir)
        self.voice_alerts = VoiceAlertManager(enabled=True, cooldown_seconds=5)  # Reduced from 30 to 5 seconds
        
        # Choose display manager based on mode
        if background_mode:
            self.display = BackgroundDisplayManager(show_minimal_ui=True)
        else:
            self.display = DisplayManager()
        
        # MediaPipe setup
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Application state
        self.running = False
        self.camera = None
        self.current_habits = []
        self.background_mode = background_mode
        self.session_habits = []  # Track all habits in current session
        
    def initialize_camera(self) -> bool:
        """Initialize camera"""
        print("Initializing camera...")
        self.camera = open_camera()
        
        if self.camera is None:
            print("ERROR: Could not open camera")
            return False
        
        if not setup_camera(self.camera, 640, 480):
            print("WARNING: Could not set optimal camera settings")
        
        camera_info = get_camera_info(self.camera)
        print(f"Camera initialized: {camera_info}")
        return True
    
    def run(self):
        """Main application loop"""
        if not self.initialize_camera():
            return
        
        if self.background_mode:
            print("\n🔔 ANXIETY HABITS DETECTOR - BACKGROUND MODE 🔔")
            print("Monitoring for: nail-biting, hair-pulling, face-touching, skin-picking")
            print("You will hear voice alerts when habits are detected.")
            print("\nControls:")
            print("  'r' - Start/Stop recording session")
            print("  'q' - Quit application")
            print("  'v' - Toggle voice alerts")
            print("  'c' - Cycle cooldown (5s → 0s → 5s)")
            print("  'h' - Show help")
        else:
            print("\n🚨 ANXIETY HABITS DETECTOR ACTIVE 🚨")
            print("Monitoring for: nail-biting, hair-pulling, face-touching, skin-picking")
            print("\nControls:")
            print("  'r' - Start/Stop recording session")
            print("  'q' - Quit application")
            print("  'h' - Show help")
        
        self.running = True
        prev_time = 0
        frame_count = 0
        
        with self.mp_holistic.Holistic(
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False
        ) as holistic:
            
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    print("ERROR: Failed to read frame")
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Calculate FPS
                fps = 1.0 / (current_time - prev_time) if prev_time > 0 else 0.0
                prev_time = current_time
                
                height, width = frame.shape[:2]
                
                # Process frame with MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb_frame)
                
                # Detect anxiety habits
                habit_events = self.detector.detect_habits(results, width, height, current_time)
                
                # Process habit events
                if habit_events:
                    print(f"🚨 HABIT DETECTED: {len(habit_events)} events")
                    for event in habit_events:
                        print(f"   - {event.habit_type.value} ({event.hand_involved} hand, {event.duration:.1f}s, {event.confidence:.2f} confidence)")
                    
                    self.current_habits.extend(habit_events)
                    self.session_habits.extend(habit_events)
                    
                    # Send voice alerts for each habit
                    for habit_event in habit_events:
                        success = self.voice_alerts.send_habit_voice_alert(habit_event)
                        if success:
                            print(f"🔊 Voice alert sent: {habit_event.habit_type.value}")
                        else:
                            print(f"🔕 Voice alert skipped (cooldown or disabled)")
                    
                    # Update display flash
                    if hasattr(self.display, 'update_habit_flash'):
                        self.display.update_habit_flash(habit_events)
                
                # Get current proximity data for display
                proximity_data = {}
                if results.left_hand_landmarks and results.face_landmarks:
                    proximity_data['left'] = self.detector.calculate_hand_face_proximity(
                        results.left_hand_landmarks, results.face_landmarks, width, height, 'left'
                    )
                    # Debug output for left hand (reduced frequency)
                    if proximity_data['left'] and proximity_data['left'].is_close and frame_count % 30 == 0:
                        print(f"👈 LEFT HAND CLOSE: {proximity_data['left'].face_region} (dist: {proximity_data['left'].normalized_distance:.3f})")
                
                if results.right_hand_landmarks and results.face_landmarks:
                    proximity_data['right'] = self.detector.calculate_hand_face_proximity(
                        results.right_hand_landmarks, results.face_landmarks, width, height, 'right'
                    )
                    # Debug output for right hand (reduced frequency)
                    if proximity_data['right'] and proximity_data['right'].is_close and frame_count % 30 == 0:
                        print(f"👉 RIGHT HAND CLOSE: {proximity_data['right'].face_region} (dist: {proximity_data['right'].normalized_distance:.3f})")
                
                # Record frame if recording
                if self.recorder.is_recording():
                    habit_data = [event.__dict__ for event in habit_events] if habit_events else None
                    self.recorder.record_frame(results, current_time, habit_data)
                
                # Prepare display data
                detection_status = self._get_detection_status(results)
                session_info = self.recorder.get_session_info()
                
                # Draw overlays based on mode
                if self.background_mode:
                    # Background mode - minimal display
                    if hasattr(self.display, 'draw_landmarks_subtle'):
                        self.display.draw_landmarks_subtle(frame, results, self.mp_holistic, self.mp_drawing)
                    if hasattr(self.display, 'draw_minimal_hud'):
                        self.display.draw_minimal_hud(frame, fps, self.recorder.is_recording(), session_info)
                    if hasattr(self.display, 'draw_subtle_proximity_indicators'):
                        self.display.draw_subtle_proximity_indicators(frame, proximity_data)
                    if hasattr(self.display, 'draw_habit_flash'):
                        self.display.draw_habit_flash(frame, self.current_habits[-1:])  # Flash for latest habit
                else:
                    # Full display mode
                    self.display.draw_landmarks(frame, results, self.mp_holistic, self.mp_drawing)
                    self.display.draw_hud(frame, fps, detection_status, 
                                        self.recorder.is_recording(), session_info)
                    self.display.draw_proximity_overlay(frame, proximity_data, {})
                    
                    # Draw habit alerts
                    if self.current_habits:
                        self.display.draw_habit_alert(frame, self.current_habits[-3:])  # Show last 3 habits
                
                # Display frame
                cv2.imshow("Anxiety Habits Detector", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self._toggle_recording()
                elif key == ord('v') and self.background_mode:
                    self._toggle_voice_alerts()
                elif key == ord('c') and self.background_mode:
                    self._cycle_cooldown()
                elif key == ord('h'):
                    self._show_help()
                
                # Clear old habits from display (keep only recent ones)
                if len(self.current_habits) > 10:
                    self.current_habits = self.current_habits[-5:]
        
        self.cleanup()
    
    def _get_detection_status(self, results) -> str:
        """Get current detection status string"""
        detections = []
        if results.face_landmarks:
            detections.append("Face")
        if results.left_hand_landmarks:
            detections.append("Left Hand")
        if results.right_hand_landmarks:
            detections.append("Right Hand")
        
        return ", ".join(detections) if detections else "No detections"
    
    def _toggle_recording(self):
        """Toggle recording session"""
        if self.recorder.is_recording():
            session = self.recorder.stop_session()
            if session:
                print(f"\n📹 Recording stopped: {session.session_id}")
                print(f"   Duration: {session.duration:.1f}s")
                print(f"   Frames: {session.total_frames}")
                print(f"   Habits detected: {session.habit_count}")
                
                # Send summary voice alert if in background mode
                if self.background_mode and self.session_habits:
                    self.voice_alerts.send_summary_voice_alert(self.session_habits, session.duration)
                
                # Reset session habits
                self.session_habits = []
        else:
            session_id = self.recorder.start_session()
            print(f"\n🔴 Recording started: {session_id}")
            self.session_habits = []  # Reset session habits
    
    def _toggle_voice_alerts(self):
        """Toggle voice alert system"""
        if self.voice_alerts.enabled:
            self.voice_alerts.disable_voice_alerts()
            print("\n🔕 Voice alerts disabled")
        else:
            self.voice_alerts.enable_voice_alerts()
            print("\n🔊 Voice alerts enabled")
    
    def _cycle_cooldown(self):
        """Cycle through cooldown settings"""
        current_cooldown = self.voice_alerts.cooldown_seconds
        if current_cooldown == 5:
            self.voice_alerts.set_cooldown(0)
            print("\n⚡ Cooldown disabled - voice alerts will play immediately")
        elif current_cooldown == 0:
            self.voice_alerts.set_cooldown(5)
            print("\n⏰ Cooldown set to 5 seconds")
        else:
            self.voice_alerts.set_cooldown(5)
            print("\n⏰ Cooldown set to 5 seconds")
    
    def _show_help(self):
        """Show help information"""
        print("\n" + "="*60)
        print("ANXIETY HABITS DETECTOR - HELP")
        print("="*60)
        print("This application detects anxiety-related habits by monitoring")
        print("hand-to-face proximity using computer vision.")
        print("\nDETECTED HABITS:")
        print("  • Nail biting (hands near mouth)")
        print("  • Hair pulling (hands near forehead/eyes)")
        print("  • Face touching (hands on cheeks/nose)")
        print("  • Skin picking (hands on face)")
        
        if self.background_mode:
            print("\nBACKGROUND MODE FEATURES:")
            print("  • Voice alerts when habits are detected")
            print("  • Minimal visual interface")
            print("  • 3-second minimum duration to avoid false positives")
            print(f"  • {self.voice_alerts.cooldown_seconds}-second cooldown between voice alerts")
            print("  • Session summary voice alerts")
            print("\nCONTROLS:")
            print("  'r' - Start/Stop recording session")
            print("  'v' - Toggle voice alerts on/off")
            print("  'c' - Cycle cooldown (5s → 0s → 5s)")
            print("  'q' - Quit application")
            print("  'h' - Show this help")
        else:
            print("\nCONTROLS:")
            print("  'r' - Start/Stop recording session")
            print("  'q' - Quit application")
            print("  'h' - Show this help")
        
        print("\nRECORDING:")
        print("  Sessions are saved as NDJSON files containing only landmark")
        print("  coordinates and habit events (no raw video data).")
        
        if self.background_mode:
            print("\nVOICE ALERTS:")
            print("  You'll hear voice announcements when habits are detected.")
            print("  Each alert tells you the specific habit and asks you to stop.")
            print("  Habits must last at least 3 seconds to avoid false positives.")
            print(f"  Voice alerts have a {self.voice_alerts.cooldown_seconds}-second cooldown to avoid spam.")
            print("  Use 'c' to cycle cooldown settings (5s → 0s → 5s).")
        
        print("="*60)
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        
        # Stop recording if active
        if self.recorder.is_recording():
            session = self.recorder.stop_session()
            if session:
                print(f"Recording stopped: {session.session_id}")
        
        # Release camera
        if self.camera:
            self.camera.release()
        
        # Close windows
        cv2.destroyAllWindows()
        
        print("Cleanup complete.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Anxiety Habits Detector")
    parser.add_argument("--output-dir", default="sessions", 
                       help="Directory to save recording sessions")
    parser.add_argument("--list-sessions", action="store_true",
                       help="List all recorded sessions")
    parser.add_argument("--foreground", action="store_true",
                       help="Run in foreground mode with full display")
    parser.add_argument("--background", action="store_true", default=True,
                       help="Run in background mode with notifications (default)")
    
    args = parser.parse_args()
    
    # Determine mode
    background_mode = args.background and not args.foreground
    
    # Create detector instance
    detector = AnxietyHabitsDetector(args.output_dir, background_mode=background_mode)
    
    # Handle list sessions command
    if args.list_sessions:
        sessions = detector.recorder.list_sessions()
        if sessions:
            print("\nRecorded Sessions:")
            print("-" * 80)
            for session in sessions:
                print(f"ID: {session['session_id']}")
                print(f"  Start: {session.get('start_time', 'Unknown')}")
                print(f"  Duration: {session.get('duration', 0):.1f}s")
                print(f"  Frames: {session.get('total_frames', 0)}")
                print(f"  Habits: {session.get('habit_count', 0)}")
                print(f"  Notes: {session.get('notes', 'None')}")
                print()
        else:
            print("No recorded sessions found.")
        return
    
    # Run the detector
    try:
        detector.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        detector.cleanup()
    except Exception as e:
        print(f"\nError: {e}")
        detector.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
