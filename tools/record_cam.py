"""Record a clean clip straight from a camera (no dashboard boxes or banners on it).

  python tools\record_cam.py "rtsp://user:pass@192.168.31.90:554/stream1" --name gloves_bare --seconds 90
  python tools\record_cam.py 0 --name gloves_bare_webcam --seconds 90      (0 = laptop webcam)

Clips are saved in recordings\<name>.mp4. Press q in the preview window to stop early.
"""
import argparse, os, time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("source", help="RTSP address, or 0 for the laptop webcam")
ap.add_argument("--name", required=True)
ap.add_argument("--seconds", type=int, default=90)
a = ap.parse_args()

src = int(a.source) if a.source.isdigit() else a.source
cap = cv2.VideoCapture(src)
ok, frame = cap.read()
if not ok:
    raise SystemExit("Could not read from the camera. Check the address / close other apps using it.")
h, w = frame.shape[:2]
os.makedirs("recordings", exist_ok=True)
out = os.path.join("recordings", a.name + ".mp4")
wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
t0 = last = time.time()
while time.time() - t0 < a.seconds:
    ok, frame = cap.read()
    if not ok:
        continue
    if time.time() - last >= 1 / 15:       # save about 15 frames a second
        wr.write(frame)
        last = time.time()
    shown = frame.copy()
    cv2.putText(shown, "REC %ds" % (a.seconds - (time.time() - t0)), (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("recording - q to stop", cv2.resize(shown, (960, int(960 * h / w))))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
wr.release(); cap.release(); cv2.destroyAllWindows()
print("saved", out)
