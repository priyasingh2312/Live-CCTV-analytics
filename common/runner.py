"""Command-line runner used by every detector script and by run_all.py."""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from .base import EventSink, _atomic_write, load_config
from .stream import StreamReader


class _LiveHub:
    """Holds the newest live JPEG so any number of browsers can watch it as a video stream."""

    def __init__(self):
        self.cond = threading.Condition()
        self.seq, self.jpg, self.viewers = 0, None, 0

    def publish(self, jpg):
        with self.cond:
            self.seq, self.jpg = self.seq + 1, jpg
            self.cond.notify_all()

    def wait(self, seq, timeout):
        with self.cond:
            if self.seq == seq:
                self.cond.wait(timeout)
            return self.seq, self.jpg


def _start_stream_server(hub, host, port, stop):
    """Tiny local web server: http://host:port/live.mjpg is a motion-JPEG video the dashboard plays in an <img>.
    Returns the server, or None when the port is not available (the dashboard then falls back to still pictures)."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/live.jpg":
                _, jpg = hub.wait(-1, 0)
                if not jpg:
                    self.send_error(503)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(jpg)
                return
            if path != "/live.mjpg":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()
            seq, idle = 0, 0
            try:
                while not stop.is_set():
                    new, jpg = hub.wait(seq, 2.0)
                    if new == seq or not jpg:
                        idle += 1
                        if idle > 30:       # no new picture for about a minute: let the browser reconnect
                            break
                        continue
                    seq, idle = new, 0
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    self.wfile.flush()
            except OSError:                 # the browser closed the picture
                pass

    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            pass                            # closed connections are normal

    try:
        srv = Server((host, port), Handler)
    except OSError as e:
        print(f"[warn] live stream not started on {host}:{port} ({e}); the dashboard will use still pictures")
        return None
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _live_loop(reader, detectors, path, stop, every, hub=None):
    """Writes the live picture for the dashboard several times a second, on its own thread, so it stays smooth
    while the AI is busy. Every running use case is drawn on the newest camera frame (boxes are from the last AI run)."""
    last_id, warned, last_file = -1, False, 0.0
    while not stop.is_set():
        t0 = time.time()
        try:
            fid, frame = reader.read()
            if frame is not None and fid != last_id:
                last_id = fid
                for row, d in enumerate(detectors):
                    d.draw(frame, row)
                h, w = frame.shape[:2]
                if w > 960:
                    frame = cv2.resize(frame, (960, int(h * 960 / w)))
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    data = buf.tobytes()
                    if hub is not None:
                        hub.publish(data)
                    # still-picture file: the dashboard's fallback and its "is the camera alive" check
                    if hub is None or time.time() - last_file > 1.0:
                        last_file = time.time()
                        _atomic_write(path, data)
        except Exception as e:  # the live picture must never stop the detectors
            if not warned:
                print(f"[warn] live picture: {e}")
                warned = True
        stop.wait(max(0.02, every - (time.time() - t0)))


def run(detector_classes, cfg, source=None, show=True, max_frames=None):
    sink = EventSink(cfg)
    detectors = []
    for cls in detector_classes:
        try:
            detectors.append(cls(cfg, sink))
            print(f"[ok]   {cls.name} ready")
        except Exception as e:
            print(f"[SKIP] {cls.name}: {e}")
    if not detectors:
        sys.exit("No detector could start. Check that the model weights exist (see README).")

    src = source or os.environ.get("RTSP_URL") or cfg["camera"]["source"]
    shown_src = str(src).split("@")[-1] if "@" in str(src) else src  # hide password in logs
    print(f"[stream] opening {shown_src}")
    reader = StreamReader(src, cfg["camera"].get("reconnect_seconds", 5)).start()

    frame_skip = max(1, int(cfg["runtime"].get("frame_skip", 3)))
    n = len(detectors)
    last_id, tick, last_wait_msg = -1, 0, time.time()
    last_step_warn = 0.0
    live_stop = threading.Event()
    live_every = float(cfg["runtime"].get("live_interval", 0.15))   # seconds between live pictures (0.15 = about 6 per second)
    hub = _LiveHub()
    stream_host = str(cfg["runtime"].get("stream_host", "127.0.0.1"))
    stream_port = int(cfg["runtime"].get("stream_port", 8599))
    stream_info = sink.out / "stream.json"
    stream_info.unlink(missing_ok=True)
    server = _start_stream_server(hub, stream_host, stream_port, live_stop) if cfg["runtime"].get("live_stream", True) else None
    if server is None:
        hub = None
    else:
        _atomic_write(stream_info, json.dumps({"host": stream_host, "port": stream_port, "pid": os.getpid()}).encode())
        print(f"[stream] live video on http://{stream_host}:{stream_port}/live.mjpg")
    threading.Thread(target=_live_loop, daemon=True, args=(
        reader, detectors, sink.out / "frames" / "_combined.jpg", live_stop, live_every, hub)).start()
    window = "CCTV Analytics - press q to quit"
    try:
        while True:
            fid, frame = reader.read()
            if frame is None:
                if time.time() - last_wait_msg > 5:
                    print("[stream] waiting for first frame ...")
                    last_wait_msg = time.time()
                time.sleep(0.1)
                continue
            if fid == last_id:
                time.sleep(0.005)
                continue
            last_id, tick = fid, tick + 1

            # Detectors take turns on a CPU: one detector per "AI slot".
            if tick % frame_skip == 0:
                try:
                    detectors[(tick // frame_skip) % n].step(frame)
                except Exception as e:  # one bad step must not stop the whole run (a restart also forgets the alert pause)
                    if time.time() - last_step_warn > 30:
                        print(f"[warn] {detectors[(tick // frame_skip) % n].name}: {e}")
                        last_step_warn = time.time()

            if show:
                disp = frame
                for row, d in enumerate(detectors):
                    d.draw(disp, row)
                cv2.imshow(window, disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if max_frames and tick >= max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        live_stop.set()
        try:
            (sink.out / "stream.json").unlink(missing_ok=True)
            if server is not None:
                server.shutdown()
        except Exception:
            pass
        reader.stop()
        cv2.destroyAllWindows()
        print("stopped.")


def build_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--source", help="rtsp:// URL, webcam index (0), or video file. "
                                    "Default: RTSP_URL in .env, else config.yaml")
    p.add_argument("--config", help="path to an alternative config.yaml")
    p.add_argument("--no-window", action="store_true", help="do not open the live window")
    p.add_argument("--max-frames", type=int, help="stop after N frames (testing)")
    return p


def main(detector_classes, description="CCTV analytics"):
    args = build_parser(description).parse_args()
    cfg = load_config(args.config)
    show = cfg["runtime"].get("show_window", True) and not args.no_window
    run(detector_classes, cfg, source=args.source, show=show, max_frames=args.max_frames)
