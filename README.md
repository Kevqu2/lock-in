# 🧠 Lock-In

A real-time computer vision app that detects and alerts you about anxiety-related habits like nail-biting, hair-pulling, face-touching, and skin-picking. Uses MediaPipe and voice alerts to help build awareness and break unwanted behaviors.

## ✨ Features

- **Real-time Detection**: Monitors hand-to-face proximity using computer vision
- **Voice Alerts**: macOS native voice notifications when habits are detected
- **Background Mode**: Runs discretely with minimal visual interface
- **Privacy-First**: Only saves landmark coordinates, no images stored
- **Session Recording**: Track your progress with NDJSON data export

## 🎮 Controls

| Key | Action |
|-----|--------|
| `r` | Start/Stop recording session |
| `v` | Toggle voice alerts |
| `c` | Cycle cooldown (5s → 0s → 5s) |
| `h` | Show help |
| `q` | Quit |

## 🔬 How It Works

The app uses MediaPipe to detect face and hand landmarks, then calculates proximity between them. When your hand stays close to specific face regions for 3+ seconds, it triggers a voice alert:

- **Mouth/Chin** → Nail biting
- **Eyes/Forehead** → Hair pulling  
- **Cheeks/Nose** → Skin picking
- **Other areas** → Face touching

## 📊 Technical Details

- **30 FPS** real-time processing
- **3-second minimum** duration to avoid false positives
- **NDJSON format** for privacy-preserving data recording
- **macOS voice synthesis** for native alerts
- **Modular architecture** with separate detection, recording, and display modules
- python3 run_detector.py to run it

## ⚠️ Disclaimer

This tool is for awareness and educational purposes only. It's not a substitute for professional mental health treatment.

---

**Built with ❤️ for mental health awareness**
