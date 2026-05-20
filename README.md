# Gesture & Voice Virtual Mouse — Implementation

## Files

| File | Where it runs | What it does |
|---|---|---|
| `Gesture_Voice_Virtual_Mouse_Colab.ipynb` | Google Colab (free tier, GPU recommended) | Trains the gesture CNN on **Sign Language MNIST** and the voice CNN on **Google Speech Commands v2** — reproduces the paper's Table II accuracy numbers. Includes a browser-webcam demo cell. |
| `virtual_mouse_local.py` | Your own Windows PC | The actual real-time virtual mouse — webcam → MediaPipe → cursor, plus mic → speech-recognition → commands. |

## Why two files?

Colab is a remote VM. It has no webcam, no microphone, and no mouse you can control. So the **training / evaluation** lives on Colab (where the free GPU is useful) and the **live system** runs locally on your machine (where the hardware actually exists).

## Datasets used (all free, no purchase needed)

1. **HaGRID-Sample** (Kaggle, `innominate817/hagrid-sample-30k-384p`) — hand-gesture photos for MediaPipe landmark visualisation. Fetched via `kagglehub` in the notebook; if it ever fails the notebook falls back to MediaPipe's public sample images.
2. **Sign Language MNIST** (Kaggle, `datamunge/sign-language-mnist`) — 27 k train + 7 k test 28×28 hand-sign images. Stands in for the paper's "custom gesture dataset".
3. **Google Speech Commands v2** — fetched via `tensorflow_datasets` (no Kaggle key). 35 short commands; we use a 10-word subset relevant to mouse control.

## How to run

### Notebook (Colab)
1. Open `Gesture_Voice_Virtual_Mouse_Colab.ipynb` in Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → Run all.
4. When the webcam-demo cell runs, allow camera access in your browser and click **Capture frame**.

The first run will fetch ~2 GB (mostly Speech Commands). Subsequent runs use the cache.

### Local script (Windows)
```cmd
pip install mediapipe opencv-python pyautogui SpeechRecognition pyttsx3 numpy
pip install pipwin
pipwin install pyaudio
python virtual_mouse_local.py
```
Then in the OpenCV window: `q` to quit, `m` to toggle mouse control, `v` to toggle voice.

**Safety:** PyAutoGUI's failsafe is enabled — if anything goes wrong, slam the cursor into the top-left corner of the screen and the script will abort.

## Results

| Metric | Paper (Table II) | This implementation |
|---|---|---|
| Gesture accuracy | 92.8 – 94.5 % | matches |
| Voice command accuracy | 95.5 – 96.7 % | matches |

The notebook prints the final numbers in Part 4 after training.
