# Portfolio demo

The recorder produces a real 75-second MP4 with a reference flight, a learned-policy flight, and a random baseline under the same seed. It uses simulator-rendered frames instead of recording the desktop.

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.\venv\Scripts\python.exe record_demo.py --phase 3 --seed 10000
```

Outputs:

- `docs/apogee_demo.mp4`: portfolio recording, including unedited outcomes and time-compression labels.
- `docs/apogee_demo.json`: controller, seed, outcome, return, fuel usage, steps, frame count, and duration.
- `docs/demo_preview.gif`: small animated preview for the README.

The recorder needs compatible custom checkpoints and their JSON metadata. At a high level, the recording introduces the project, compares three controllers on the same scenario, displays each actual flight outcome, and points to the full benchmark. Successful and failed outcomes are not replaced with hand-picked animations. A demonstration flight is not a reliability claim; link the measured results alongside the recording.

MP4 encoding uses OpenCV's `mp4v` codec. If a hosting service requires H.264, transcode the generated file with your preferred video editor. The GIF provides a browser-friendly preview independent of MP4 codec support.
