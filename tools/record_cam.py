"""Record a clean clip straight from a camera (no dashboard boxes or banners on it).

  python tools/record_cam.py --name seat_nocap --seconds 60
      uses the camera currently selected in the dashboard (the RTSP_URL line in .env)
  python tools/record_cam.py 0 --name bare_webcam --seconds 60
      0 = the laptop webcam
  python tools/record_cam.py "rtsp://user:pass@192.168.7.27:554/stream1" --name x
      a camera address typed in full

Clips are saved in the recordings folder as <name>.mp4. Press q in the preview window to stop early.
"""
import argparse, os, time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("source", nargs="?", default=None, help="RTSP address, or 0 for the laptop webcam. Leave out to use RTSP_URL from .env")
ap.add_argument("--name", required=True)
ap.add_argument("--seconds", type=int, default=90)
a = ap.parse_args()

source = a.source
if source is None:
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
    except ImportError:
        pass
    source = os.environ.get("RTSP_URL", "").strip()
    if not source:
        raise SystemExit("No camera given and RTSP_URL is not set in .env. Select a camera in the dashboard first, or give the address.")
    print("using the camera from .env (RTSP_URL)")
if "<" in source:
    raise SystemExit("Replace the <...> placeholder with the real camera address, or leave the address out to use the one in .env.")
src = int(source) if source.isdigit() else source
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
