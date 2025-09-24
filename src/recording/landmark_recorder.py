"""
Landmark Recorder for Anxiety Habits Detector

Records MediaPipe landmarks and habit events in NDJSON format for privacy-preserving data collection.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import mediapipe as mp


@dataclass
class LandmarkData:
    """Data structure for landmark information"""
    timestamp: float
    frame_number: int
    face_landmarks: Optional[List[Dict]] = None
    left_hand_landmarks: Optional[List[Dict]] = None
    right_hand_landmarks: Optional[List[Dict]] = None
    habit_events: Optional[List[Dict]] = None


@dataclass
class SessionMetadata:
    """Metadata for a recording session"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    total_frames: int = 0
    habit_count: int = 0
    notes: Optional[str] = None


class LandmarkRecorder:
    """Records landmarks and habit events in NDJSON format"""
    
    def __init__(self, output_dir: str = "sessions"):
        self.output_dir = output_dir
        self.current_session: Optional[SessionMetadata] = None
        self.session_file = None
        self.frame_count = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def start_session(self, session_id: Optional[str] = None, notes: Optional[str] = None) -> str:
        """Start a new recording session"""
        if session_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = f"session_{timestamp}"
        
        self.current_session = SessionMetadata(
            session_id=session_id,
            start_time=datetime.now().timestamp(),
            notes=notes
        )
        
        # Create session file
        session_filename = f"{session_id}.ndjson"
        self.session_file = open(os.path.join(self.output_dir, session_filename), 'w')
        
        # Write session header
        header = {
            "type": "session_header",
            "session_id": session_id,
            "start_time": self.current_session.start_time,
            "notes": notes,
            "version": "1.0"
        }
        self.session_file.write(json.dumps(header) + '\n')
        
        self.frame_count = 0
        return session_id
    
    def stop_session(self) -> Optional[SessionMetadata]:
        """Stop the current recording session"""
        if not self.current_session or not self.session_file:
            return None
        
        end_time = datetime.now().timestamp()
        self.current_session.end_time = end_time
        self.current_session.duration = end_time - self.current_session.start_time
        self.current_session.total_frames = self.frame_count
        
        # Write session footer
        footer = {
            "type": "session_footer",
            "session_id": self.current_session.session_id,
            "end_time": end_time,
            "duration": self.current_session.duration,
            "total_frames": self.current_session.total_frames,
            "habit_count": self.current_session.habit_count
        }
        self.session_file.write(json.dumps(footer) + '\n')
        
        # Close file
        self.session_file.close()
        
        # Reset state
        completed_session = self.current_session
        self.current_session = None
        self.session_file = None
        self.frame_count = 0
        
        return completed_session
    
    def record_frame(self, 
                    results, 
                    timestamp: float,
                    habit_events: Optional[List] = None) -> bool:
        """Record landmarks for a single frame"""
        if not self.current_session or not self.session_file:
            return False
        
        self.frame_count += 1
        
        # Convert landmarks to serializable format
        landmark_data = LandmarkData(
            timestamp=timestamp,
            frame_number=self.frame_count,
            habit_events=habit_events or []
        )
        
        # Extract face landmarks
        if results.face_landmarks:
            landmark_data.face_landmarks = [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": getattr(lm, 'visibility', 1.0)}
                for lm in results.face_landmarks.landmark
            ]
        
        # Extract left hand landmarks
        if results.left_hand_landmarks:
            landmark_data.left_hand_landmarks = [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": getattr(lm, 'visibility', 1.0)}
                for lm in results.left_hand_landmarks.landmark
            ]
        
        # Extract right hand landmarks
        if results.right_hand_landmarks:
            landmark_data.right_hand_landmarks = [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": getattr(lm, 'visibility', 1.0)}
                for lm in results.right_hand_landmarks.landmark
            ]
        
        # Write to file as NDJSON
        self.session_file.write(json.dumps(asdict(landmark_data)) + '\n')
        
        # Update habit count
        if habit_events:
            self.current_session.habit_count += len(habit_events)
        
        return True
    
    def is_recording(self) -> bool:
        """Check if currently recording a session"""
        return self.current_session is not None and self.session_file is not None
    
    def get_session_info(self) -> Optional[Dict]:
        """Get current session information"""
        if not self.current_session:
            return None
        
        return {
            "session_id": self.current_session.session_id,
            "start_time": self.current_session.start_time,
            "frame_count": self.frame_count,
            "duration": datetime.now().timestamp() - self.current_session.start_time if self.current_session else 0,
            "notes": self.current_session.notes
        }
    
    def list_sessions(self) -> List[Dict]:
        """List all recorded sessions"""
        sessions = []
        
        if not os.path.exists(self.output_dir):
            return sessions
        
        for filename in os.listdir(self.output_dir):
            if filename.endswith('.ndjson'):
                session_id = filename[:-7]  # Remove .ndjson extension
                
                # Try to get session info from file
                filepath = os.path.join(self.output_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        first_line = f.readline()
                        last_line = None
                        for line in f:
                            last_line = line
                        
                        header = json.loads(first_line) if first_line else {}
                        footer = json.loads(last_line) if last_line else {}
                        
                        sessions.append({
                            "session_id": session_id,
                            "start_time": header.get("start_time", 0),
                            "duration": footer.get("duration", 0),
                            "total_frames": footer.get("total_frames", 0),
                            "habit_count": footer.get("habit_count", 0),
                            "notes": header.get("notes", ""),
                            "filename": filename
                        })
                except (json.JSONDecodeError, IOError):
                    # If we can't read the file, just add basic info
                    sessions.append({
                        "session_id": session_id,
                        "filename": filename,
                        "error": "Could not read session file"
                    })
        
        return sorted(sessions, key=lambda x: x.get("start_time", 0), reverse=True)
