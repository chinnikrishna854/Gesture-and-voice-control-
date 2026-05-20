"""
virtual_mouse_local.py
======================
Real-time gesture- and voice-controlled virtual mouse.
Implements the system from:
    Bhukya et al., "Gesture and Voice-Controlled Virtual Mouse:
    An AI-based Approach to Hands-Free Navigation" (IEEE, VIT-AP/SJSU).

Runs on YOUR machine (Windows/Linux/macOS) — NOT on Colab.
Needs a working webcam, microphone, and OS-level cursor control.

INSTALL (Windows / cmd):
    pip install mediapipe opencv-python pyautogui SpeechRecognition pyttsx3 numpy
    # PyAudio is needed by SpeechRecognition for mic input:
    pip install pipwin && pipwin install pyaudio
    # (on Linux: sudo apt install portaudio19-dev && pip install pyaudio)

RUN:
    python virtual_mouse_local.py

KEYS while running:
    q   quit
    v   toggle voice listener
    m   toggle mouse control

Gestures (paper §III.A):
    Index + middle up, apart  -> move cursor
    Index + middle up, touching -> left click
    Index up alone            -> right click
    Open palm                 -> neutral
    Thumb up only             -> scroll up
    Thumb down only           -> scroll down

Voice commands (a subset of the paper's set):
    "click"            left click at current pos
    "double click"     double click
    "right click"      right click
    "scroll up/down"   scroll
    "open <app>"       launches notepad / chrome / calculator
    "search <query>"   opens browser search
    "stop listening"   pauses voice listener
"""

import sys
import time
import threading
import queue
import webbrowser
import subprocess
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import pyautogui

# --- Voice imports are optional; the gesture half still works without them
try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_AVAILABLE = True
except Exception as e:
    print(f"[warn] voice modules not available: {e}")
    VOICE_AVAILABLE = False

# ---------- pyautogui safety ----------
pyautogui.FAILSAFE = True            # move cursor to top-left corner to abort
pyautogui.PAUSE = 0.0                # we add our own smoothing

SCREEN_W, SCREEN_H = pyautogui.size()
CAM_W, CAM_H = 640, 480

# ---------- MediaPipe ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# Landmark indices (standard 21-point hand model)
TIP = {'thumb': 4, 'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20}
PIP = {'thumb': 3, 'index': 6, 'middle': 10, 'ring': 14, 'pinky': 18}


def fingers_up(lm):
    """Which fingers are extended? lm is list of normalized (x,y,z)."""
    up = {}
    for f in ['index', 'middle', 'ring', 'pinky']:
        up[f] = lm[TIP[f]].y < lm[PIP[f]].y
    # Thumb: heuristic via x-coordinate (right hand assumed)
    up['thumb'] = lm[TIP['thumb']].x < lm[PIP['thumb']].x
    return up


def classify_gesture(lm):
    """Map landmarks to a cursor action (paper §III.A, Fig. 2)."""
    f = fingers_up(lm)
    if f['index'] and f['middle'] and not f['ring'] and not f['pinky']:
        ix, iy = lm[TIP['index']].x, lm[TIP['index']].y
        mx, my = lm[TIP['middle']].x, lm[TIP['middle']].y
        d = ((ix - mx) ** 2 + (iy - my) ** 2) ** 0.5
        return 'LEFT_CLICK' if d < 0.04 else 'MOVE'
    if f['index'] and not f['middle'] and not f['ring'] and not f['pinky']:
        return 'RIGHT_CLICK'
    if all(f.values()):
        return 'NEUTRAL'
    if f['thumb'] and not f['index'] and not f['middle']:
        return 'SCROLL_UP' if lm[TIP['thumb']].y < lm[0].y else 'SCROLL_DOWN'
    return 'NEUTRAL'


# ============================================================
# Voice assistant (runs in a background thread)
# ============================================================
class VoiceAssistant(threading.Thread):
    def __init__(self, command_queue):
        super().__init__(daemon=True)
        self.q = command_queue
        self.running = True
        self.enabled = True
        if VOICE_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.tts = pyttsx3.init()
            self.tts.setProperty('rate', 175)

    def speak(self, text):
        if not VOICE_AVAILABLE:
            return
        try:
            self.tts.say(text)
            self.tts.runAndWait()
        except Exception:
            pass

    def run(self):
        if not VOICE_AVAILABLE:
            return
        with sr.Microphone() as src:
            try:
                self.recognizer.adjust_for_ambient_noise(src, duration=0.8)
            except Exception:
                pass
            self.speak("Voice assistant ready")
            while self.running:
                if not self.enabled:
                    time.sleep(0.3); continue
                try:
                    audio = self.recognizer.listen(src, timeout=4, phrase_time_limit=4)
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"[voice] heard: {text!r}")
                    self.q.put(text)
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    print(f"[voice] API error: {e}"); time.sleep(1)
                except Exception as e:
                    print(f"[voice] error: {e}"); time.sleep(0.5)


def handle_voice_command(cmd, va):
    """Execute a recognised voice command."""
    c = cmd.strip().lower()
    if 'stop listening' in c or 'pause listening' in c:
        va.enabled = False; va.speak("Voice paused"); return
    if 'start listening' in c or 'resume listening' in c:
        va.enabled = True; va.speak("Listening"); return
    if 'double click' in c:
        pyautogui.doubleClick(); va.speak("Double clicked"); return
    if 'right click' in c:
        pyautogui.rightClick(); va.speak("Right clicked"); return
    if c == 'click' or c.endswith(' click') or c.startswith('click'):
        pyautogui.click(); va.speak("Clicked"); return
    if 'scroll up' in c:
        pyautogui.scroll(400); va.speak("Scrolling up"); return
    if 'scroll down' in c:
        pyautogui.scroll(-400); va.speak("Scrolling down"); return
    if c.startswith('open '):
        target = c[5:].strip()
        apps = {'notepad': 'notepad.exe', 'calculator': 'calc.exe',
                'chrome': 'chrome.exe', 'browser': 'chrome.exe'}
        if target in apps:
            try:
                subprocess.Popen(apps[target]); va.speak(f"Opening {target}"); return
            except Exception as e:
                print(f"[voice] open failed: {e}")
        webbrowser.open(f"https://www.google.com/search?q={target.replace(' ', '+')}")
        va.speak(f"Searching {target}"); return
    if c.startswith('search '):
        query = c[7:].strip()
        webbrowser.open(f"https://www.google.com/search?q={query.replace(' ', '+')}")
        va.speak(f"Searching {query}"); return
    if 'what time' in c or 'time' == c.strip():
        va.speak("It is " + time.strftime("%I:%M %p")); return
    print(f"[voice] no action for: {c!r}")


# ============================================================
# Main loop
# ============================================================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        print("ERROR: cannot open webcam"); sys.exit(1)

    hands = mp_hands.Hands(model_complexity=1, max_num_hands=1,
                           min_detection_confidence=0.6,
                           min_tracking_confidence=0.5)

    cmd_q = queue.Queue()
    va = VoiceAssistant(cmd_q)
    if VOICE_AVAILABLE:
        va.start()
    voice_on = VOICE_AVAILABLE
    mouse_on = True

    # Smoothing buffer for cursor (paper §III.A "refinement")
    smooth_x = deque(maxlen=4)
    smooth_y = deque(maxlen=4)

    # Click debouncing
    last_click_t = 0.0
    CLICK_COOLDOWN = 0.6
    last_scroll_t = 0.0
    SCROLL_COOLDOWN = 0.25

    # Region inside the camera frame that maps to the full screen.
    # This margin makes corners reachable without hyperextending the arm.
    MARGIN = 100
    map_x0, map_y0 = MARGIN, MARGIN
    map_x1, map_y1 = CAM_W - MARGIN, CAM_H - MARGIN

    print(f"Screen: {SCREEN_W}x{SCREEN_H}  | Camera: {CAM_W}x{CAM_H}")
    print("Keys: q=quit, v=toggle voice, m=toggle mouse")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)                       # mirror — feels natural
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            gesture = 'NO_HAND'
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(
                    frame, lm, mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style())
                gesture = classify_gesture(lm.landmark)

                # Cursor position from the index-fingertip
                ix = int(lm.landmark[TIP['index']].x * CAM_W)
                iy = int(lm.landmark[TIP['index']].y * CAM_H)

                if mouse_on and gesture == 'MOVE':
                    # Map camera region -> screen
                    sx = np.interp(ix, [map_x0, map_x1], [0, SCREEN_W])
                    sy = np.interp(iy, [map_y0, map_y1], [0, SCREEN_H])
                    smooth_x.append(sx); smooth_y.append(sy)
                    try:
                        pyautogui.moveTo(np.mean(smooth_x), np.mean(smooth_y), duration=0.0)
                    except pyautogui.FailSafeException:
                        print("[safety] failsafe triggered — exiting")
                        break

                now = time.time()
                if mouse_on and gesture == 'LEFT_CLICK' and (now - last_click_t) > CLICK_COOLDOWN:
                    pyautogui.click(); last_click_t = now
                    print("[gesture] LEFT_CLICK")
                elif mouse_on and gesture == 'RIGHT_CLICK' and (now - last_click_t) > CLICK_COOLDOWN:
                    pyautogui.rightClick(); last_click_t = now
                    print("[gesture] RIGHT_CLICK")
                elif mouse_on and gesture == 'SCROLL_UP' and (now - last_scroll_t) > SCROLL_COOLDOWN:
                    pyautogui.scroll(60); last_scroll_t = now
                elif mouse_on and gesture == 'SCROLL_DOWN' and (now - last_scroll_t) > SCROLL_COOLDOWN:
                    pyautogui.scroll(-60); last_scroll_t = now

            # Drain voice commands
            while not cmd_q.empty():
                handle_voice_command(cmd_q.get(), va)

            # HUD
            cv2.rectangle(frame, (map_x0, map_y0), (map_x1, map_y1), (40, 200, 40), 1)
            hud_lines = [
                f"Gesture: {gesture}",
                f"Mouse:   {'ON' if mouse_on else 'OFF'}",
                f"Voice:   {'ON' if voice_on else 'OFF'}{'' if VOICE_AVAILABLE else ' (unavailable)'}",
                "q quit | m mouse | v voice",
            ]
            for i, t in enumerate(hud_lines):
                cv2.putText(frame, t, (10, 24 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 220), 2, cv2.LINE_AA)

            cv2.imshow('Virtual Mouse — press q to quit', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                mouse_on = not mouse_on; print(f"mouse -> {mouse_on}")
            elif key == ord('v') and VOICE_AVAILABLE:
                voice_on = not voice_on; va.enabled = voice_on
                print(f"voice -> {voice_on}")
    finally:
        if VOICE_AVAILABLE:
            va.running = False
        cap.release()
        cv2.destroyAllWindows()
        print("Bye.")


if __name__ == '__main__':
    main()
