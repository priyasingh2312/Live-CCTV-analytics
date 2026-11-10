"""Streamlit control-panel dashboard for the CCTV analytics (one RTSP camera).

Start it (PowerShell, from the project folder):
    streamlit run dashboard\\app.py

What it does
  * You tick the use cases you want and press Start. The dashboard launches
    `run_all.py --only <ticked> --no-window` in the background. The camera is
    taken from RTSP_URL in .env, exactly like the command-line detectors.
  * To change the selection press Apply (the background process is restarted,
    detectors cannot be added to a running process).
  * Live frames, counts and alerts are read from the files the detectors write
    into outputs\\ . The dashboard never opens the camera itself.
  * The detectors keep running if you close the browser tab. Press Stop to end them.
"""
import base64
import hmac
import html
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import pandas as pd
import streamlit as st
import yaml

try:
    from common.face_db import FaceDB, valid_name
except ImportError:
    FaceDB = None

try:
    from report import build_pdf          # dashboard/report.py
except ImportError:
    build_pdf = None

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))          # so  common.face_db  can be imported
OUT = ROOT / "outputs"
LOG = OUT / "run.log"
for _sub in ("stats", "frames", "snapshots"):
    (OUT / _sub).mkdir(parents=True, exist_ok=True)

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# detector name -> label shown on the dashboard
LABELS = {
    "person_count": "Person count",
    "cap_detection": "Cap",
    "gloves_detection": "Gloves",
    "button_detection": "Shirt buttons",
    "shoes_detection": "Shoes",
    "mask_detection": "Mask",
    "beard_detection": "Beard",
    "object_detection": "Objects (laptop, chair, diary, phone)",
    "face_recognition": "Face recognition",      # button and checkbox are ready; the detector itself is added later
}
HEAVY = {"cap_detection", "mask_detection", "beard_detection"}   # CLIP based, slower on CPU
PRESETS = {
    "Safety kit": ["gloves_detection", "cap_detection", "mask_detection", "shoes_detection"],
    "Office": ["person_count", "object_detection"],
    "Dress code": ["button_detection", "shoes_detection", "cap_detection"],
    "Face recognition": ["face_recognition"],
    "All": list(LABELS),
    "None": [],
}
def _wide():
    # Streamlit 1.50+ uses width="stretch"; older versions use use_container_width=True
    try:
        v = tuple(int(x) for x in st.__version__.split(".")[:2])
    except ValueError:
        v = (1, 0)
    return {"width": "stretch"} if v >= (1, 50) else {"use_container_width": True}


WIDE = _wide()
FRESH_SECONDS = 10      # a detector is "live" if it wrote its stats this recently
MAX_AUTO_RESTARTS = 5


# ------------------------------------------------------------------ config
def load_cfg():
    try:
        return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def availability(cfg):
    """{detector: (ok, note)} - looks only at which model files exist."""
    res = {}
    for name in LABELS:
        block = (cfg.get("detectors") or {}).get(name)
        if not block:
            res[name] = (False, "not set up yet (no block in config.yaml)")
            continue
        missing = [m["weights"] for m in block.get("models", [])
                   if not m.get("coco") and not (ROOT / m["weights"]).exists()]
        clip = block.get("clip")
        clip_ok = isinstance(clip, dict) and clip.get("enabled", True) and name in HEAVY
        if not missing:
            res[name] = (True, "")
        elif clip_ok:
            res[name] = (True, "uses the CLIP fallback (no trained model)")
        else:
            res[name] = (False, "missing " + ", ".join(Path(m).name for m in missing))
    return res


def is_browser_url(u):
    u = str(u).strip()
    return u == "browser" or u.startswith("browser:")


def slot_of_url(u):
    u = str(u).strip()
    return u.split(":", 1)[1] if ":" in u else "default"


def rtsp_source(cfg):
    """(url, shown_text). RTSP_URL from .env wins, then config camera.source."""
    url = os.environ.get("RTSP_URL") or ""
    if not url:
        src = (cfg.get("camera") or {}).get("source")
        if isinstance(src, str) and src.lower().startswith("rtsp://"):
            url = src
    if not url:
        return "", ""
    if url.strip().isdigit():                       # a webcam of this PC: RTSP_URL=0
        return url.strip(), f"Webcam {url.strip()}"
    if is_browser_url(url):                         # the webcam of another laptop, sent by its browser (serve.py)
        return url.strip(), "Browser webcam" + ("" if slot_of_url(url) == "default" else " " + slot_of_url(url))
    p = urlparse(url)
    return url, f"{p.hostname or '?'}:{p.port or 554}"      # never show user/password


def rtsp_parts(url):
    """Split an rtsp:// address into (host, port, user, password, path) for the Change camera form."""
    p = urlparse(url) if url else None
    if not p:
        return "", 554, "", "", "stream1"
    return (p.hostname or "", p.port or 554, unquote(p.username or ""), unquote(p.password or ""),
            (p.path or "/stream1").lstrip("/") + (("?" + p.query) if p.query else ""))


def build_rtsp(host, port, user, pw, path):
    auth = ""
    if user:
        auth = quote(user, safe="") + ((":" + quote(pw, safe="")) if pw else "") + "@"
    return f"rtsp://{auth}{host.strip()}:{int(port)}/{path.strip().lstrip('/')}"


def save_rtsp(url):
    """Write RTSP_URL into .env (other lines kept) and into this process, so the detectors started next use it."""
    env = ROOT / ".env"
    lines = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
    out, done = [], False
    for ln in lines:
        if re.match(r"\s*RTSP_URL\s*=", ln):
            if not done:
                out.append(f"RTSP_URL={url}")
                done = True
        else:
            out.append(ln)
    if not done:
        out.append(f"RTSP_URL={url}")
    tmp = env.with_name(".env.tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    os.replace(tmp, env)
    os.environ["RTSP_URL"] = url


CAM_LOG = OUT / "cameras.json"


def load_cams():
    """The camera log: every camera that was ever added (address without user/password)."""
    try:
        data = json.loads(CAM_LOG.read_text(encoding="utf-8"))
        return [c for c in data if isinstance(c, dict) and c.get("id")]
    except (OSError, ValueError):
        return []


def _write_cams(cams):
    try:
        CAM_LOG.parent.mkdir(parents=True, exist_ok=True)
        tmp = CAM_LOG.with_name("cameras.json.tmp")
        tmp.write_text(json.dumps(cams, indent=2), encoding="utf-8")
        os.replace(tmp, CAM_LOG)
    except OSError:
        pass


def register_cam(url, name=None):
    """Return (log entry, is_new). A camera is matched by IP and port; a new address is added as the next CAM number.
    The address is kept so the camera can be selected again later (this file stays on this PC, like .env)."""
    if str(url).strip().isdigit():
        host, port = "webcam", int(str(url).strip())
    elif is_browser_url(url):
        host, port = "browser:" + slot_of_url(url), 0
    else:
        p = urlparse(url)
        host, port = p.hostname or "", p.port or 554
    cams = load_cams()
    for c in cams:
        if c.get("host") == host and int(c.get("port", 554)) == port:
            if c.get("url") != url or (name and c.get("name") != name):
                c["url"] = url
                if name:
                    c["name"] = name
                _write_cams(cams)
            return c, False
    nums = [int(m.group(1)) for c in cams for m in [re.fullmatch(r"CAM (\d+)", str(c["id"]))] if m]
    entry = {"id": f"CAM {(max(nums) if nums else 0) + 1:02d}", "name": name or "", "host": host, "port": port, "url": url,
             "added": datetime.now().isoformat(timespec="seconds")}
    cams.append(entry)
    _write_cams(cams)
    return entry, True


def update_cam(cid, url, name=None):
    cams = load_cams()
    for c in cams:
        if c["id"] == cid:
            if str(url).strip().isdigit():
                c["host"], c["port"] = "webcam", int(str(url).strip())
            elif is_browser_url(url):
                c["host"], c["port"] = "browser:" + slot_of_url(url), 0
            else:
                p = urlparse(url)
                c["host"], c["port"] = p.hostname or "", p.port or 554
            c["url"], c["name"] = url, name or ""
    _write_cams(cams)


def delete_cam(cid):
    """Remove a camera from the list (its alerts stay in Alert history). If it was the active one, switch to another."""
    cams = [c for c in load_cams() if c["id"] != cid]
    _write_cams(cams)
    if cid == CAM_LABEL:
        nxt = next((c for c in cams if c.get("url")), None)
        save_rtsp(nxt["url"] if nxt else "")


def test_webcam(index):
    code = ("import cv2,sys\ni=int(sys.argv[1])\nc=cv2.VideoCapture(i,cv2.CAP_DSHOW) if sys.platform=='win32' else cv2.VideoCapture(i)\n"
            "ok,f=c.read()\nprint('OK %dx%d' % (f.shape[1], f.shape[0]) if ok else 'FAIL')")
    try:
        r = subprocess.run([sys.executable, "-c", code, str(int(index))], capture_output=True, text=True, timeout=20,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return False, "The webcam did not answer (timed out). Close other apps that use it."
    if r.stdout.strip().startswith("OK"):
        return True, "Webcam works. Got a picture (" + r.stdout.strip()[3:] + ")."
    return False, "Could not read a picture. Try another webcam number (0, 1, 2) and close other apps that use the camera."


def addr_line(txt):
    return esc(txt) if str(txt).startswith(("Webcam", "Browser")) else esc(txt) + " &middot; RTSP"


def test_browser(url="browser"):
    """A browser webcam is fine when the other laptop's page has sent a picture in the last few seconds."""
    f = ROOT / "outputs" / "browser_cam" / (slot_of_url(url) + ".jpg")
    try:
        age = time.time() - f.stat().st_mtime
    except OSError:
        return False, "No picture yet. Open the browser-camera link on the other laptop and press Start camera."
    if age > 10:
        return False, f"The last picture is {int(age)} s old. On the other laptop press Start camera and keep that tab open."
    return True, "Browser webcam works. A picture just arrived."


def next_browser_slot():
    used = {slot_of_url(c.get("url", "")) for c in load_cams() if is_browser_url(c.get("url", ""))}
    n = 1
    while f"web{n}" in used:
        n += 1
    return f"web{n}"


def browser_link(slot):
    """The address to open on the other laptop (built from the address this dashboard is being viewed on, e.g. the ngrok one)."""
    try:
        key = (ROOT / ".browser_cam_key").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    host, scheme = "", "https"
    try:
        h = st.context.headers
        host = h.get("X-Forwarded-Host") or h.get("Host") or ""
        scheme = (h.get("X-Forwarded-Proto") or ("http" if host.startswith(("localhost", "127.")) else "https")).split(",")[0]
    except Exception:
        pass
    return f"{scheme}://{host or '<your-ngrok-address>'}/browser-camera?key={key}&cam={slot}"


def test_rtsp(url):
    """Try to read one frame in a separate process (so a dead address cannot hang the dashboard)."""
    if str(url).strip().isdigit():
        return test_webcam(url)
    if is_browser_url(url):
        return test_browser(url)
    code = ("import cv2,sys\nc=cv2.VideoCapture(sys.argv[1],cv2.CAP_FFMPEG)\nok,f=c.read()\n"
            "print('OK %dx%d' % (f.shape[1], f.shape[0]) if ok else 'FAIL')")
    env = dict(os.environ, OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|stimeout;6000000")
    try:
        r = subprocess.run([sys.executable, "-c", code, url], capture_output=True, text=True, timeout=20, env=env,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return False, "No answer from the camera (timed out). Check the IP address, port and that the PC can reach it."
    if r.stdout.strip().startswith("OK"):
        return True, "Connected. Got a picture (" + r.stdout.strip()[3:] + ")."
    return False, "Could not read a picture. Check the IP, port, user, password and stream name."


# ------------------------------------------------------------------ process control
STATE = OUT / "run_state.json"     # pid of the background process, so a restarted dashboard finds it again


def pid_alive(pid):
    """True if `pid` is a running python/run_all process (works on Windows and Linux)."""
    if not pid:
        return False
    if os.name == "nt":
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True,
                                 text=True, timeout=8, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.lower()
        except Exception:
            return False
        return f'"{pid}"' in out and "python" in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        return "run_all" in Path(f"/proc/{pid}/cmdline").read_text()
    except OSError:
        return True


def kill_pid(pid):
    """Stop a process and its children."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=15,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            import signal
            os.kill(pid, signal.SIGTERM)
    except Exception:
        pass


def _save_state(pid, names, started):
    try:
        STATE.write_text(json.dumps({"pid": pid, "names": names, "started": started}))
    except OSError:
        pass


def _drop_state():
    try:
        STATE.unlink()
    except OSError:
        pass


def _clear_stale(max_age=30):
    """Remove live files that an old, stopped run left behind (they would show as stale tiles)."""
    now = time.time()
    for p in list((OUT / "stats").glob("*.json")):
        try:
            if now - p.stat().st_mtime > max_age:
                p.unlink()
                (OUT / "frames" / (p.stem + ".jpg")).unlink(missing_ok=True)
        except OSError:
            pass


@st.cache_resource
def ctl():
    """One state dict shared by every browser session of this dashboard server.
    If an earlier dashboard left detectors running (it was closed or crashed), they are picked up again."""
    c = {"proc": None, "adopted": None, "names": [], "started": 0.0, "wanted": False,
         "restarts": 0, "log": None, "last_restart": 0.0}
    try:
        st_ = json.loads(STATE.read_text())
        if pid_alive(st_.get("pid")):
            c.update(adopted=st_["pid"], names=list(st_.get("names") or []), started=float(st_.get("started") or 0))
        else:
            _drop_state()
    except (OSError, ValueError):
        pass
    if not c["adopted"]:
        _clear_stale()
    import atexit
    atexit.register(_shutdown, c)       # stopping the dashboard (Ctrl+C) also stops its detectors
    return c


def _shutdown(c):
    try:
        p = c.get("proc")
        if p is not None and p.poll() is None:
            kill_pid(p.pid)
        if c.get("adopted") and pid_alive(c["adopted"]):
            kill_pid(c["adopted"])
        _drop_state()
    except Exception:
        pass


def running():
    c = ctl()
    if c["proc"] is not None and c["proc"].poll() is None:
        return True
    if c["adopted"]:
        if pid_alive(c["adopted"]):
            return True
        c["adopted"], c["names"] = None, []
        _drop_state()
    return False


def _clear_outputs():
    for p in list((OUT / "stats").glob("*.json")) + list((OUT / "frames").glob("*.jpg")):
        try:
            p.unlink()
        except OSError:
            pass


def _spawn(names, fresh=True):
    c = ctl()
    if c["log"]:
        try:
            c["log"].close()
        except OSError:
            pass
    if fresh:
        _clear_outputs()
    c["log"] = open(LOG, "w" if fresh else "a", encoding="utf-8", errors="replace")
    cmd = [sys.executable, "-u", "run_all.py", "--only", ",".join(names), "--no-window"]
    override = os.environ.get("CCTV_TEST_SOURCE")        # testing only, e.g. a phone camera
    if override:
        cmd += ["--source", override]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    c["proc"] = subprocess.Popen(cmd, cwd=str(ROOT), stdout=c["log"],
                                 stderr=subprocess.STDOUT, creationflags=flags)
    c["names"], c["started"] = list(names), time.time()
    _save_state(c["proc"].pid, c["names"], c["started"])


def start(names):
    c = ctl()
    stop()
    c["wanted"], c["restarts"] = True, 0
    _spawn(names)


def stop():
    c = ctl()
    c["wanted"] = False
    p = c["proc"]
    if p is not None and p.poll() is None:
        kill_pid(p.pid)
        try:
            p.wait(8)
        except subprocess.TimeoutExpired:
            p.kill()
    c["proc"] = None
    if c["adopted"]:
        if pid_alive(c["adopted"]):
            kill_pid(c["adopted"])
        c["adopted"] = None
    c["names"] = []
    _drop_state()
    if c["log"]:
        try:
            c["log"].close()
        except OSError:
            pass
        c["log"] = None
    _clear_outputs()


def watchdog():
    """Restart the background process if it died on its own (limited number of tries)."""
    c = ctl()
    p = c["proc"]
    if not c["wanted"] or p is None or p.poll() is None:
        return
    if time.time() - c["started"] > 120:
        c["restarts"] = 0            # it ran fine for a while; allow fresh retries
    if c["restarts"] >= MAX_AUTO_RESTARTS:
        c["wanted"] = False
        return
    if time.time() - c["last_restart"] < 5:
        return
    c["restarts"] += 1
    c["last_restart"] = time.time()
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}|{os.environ.get('CAMERA_NAME', '')}] [dashboard] process exited (code {p.returncode}); restart {c['restarts']}/{MAX_AUTO_RESTARTS}\n")
    _spawn(c["names"], fresh=False)


# ------------------------------------------------------------------ data helpers
def read_stats():
    out = {}
    for p in (OUT / "stats").glob("*.json"):
        try:
            s = json.loads(p.read_text())
            out[s["detector"]] = s
        except (ValueError, OSError, KeyError):
            pass        # file being rewritten; next refresh will get it
    return out


def log_tail(n=40):
    try:
        with open(LOG, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 80000))
            lines = f.read().decode("utf-8", errors="replace").splitlines()
        return lines[-n:]
    except OSError:
        return []


def skipped():
    """{detector: reason} for detectors the runner could not start."""
    res = {}
    for ln in log_tail(200):
        if ln.startswith("[SKIP]"):
            name, _, reason = ln[7:].partition(":")
            res[name.strip()] = reason.strip()
    return res


EVENT_COLS = ["id", "timestamp", "camera", "detector", "event", "count", "snapshot", "status"]
HAS_REVIEW = True


def read_events():
    p = OUT / "events.db"
    empty = pd.DataFrame(columns=EVENT_COLS)
    if not p.exists():
        return empty
    try:
        con = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True, timeout=10)
        try:
            try:
                df = pd.read_sql_query("SELECT id, ts AS timestamp, camera, detector, event, count, snapshot, status "
                                       "FROM events ORDER BY ts", con)
            except Exception:                      # an old database without the status column
                df = pd.read_sql_query("SELECT id, ts AS timestamp, camera, detector, event, count, snapshot, 'new' AS status "
                                       "FROM events ORDER BY ts", con)
        finally:
            con.close()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df.dropna(subset=["timestamp"])
    except Exception:
        return empty


def mark_reviewed(event_id):
    """Set an alert's status to 'reviewed' in the event database."""
    try:
        con = sqlite3.connect(str(OUT / "events.db"), timeout=10)
        try:
            con.execute("PRAGMA busy_timeout=10000")
            con.execute("UPDATE events SET status='reviewed' WHERE id=?", (int(event_id),))
            con.commit()
        finally:
            con.close()
    except Exception as e:
        print("[dashboard] could not mark the alert reviewed:", e)


def fragment(every):
    """Auto-refreshing block (falls back to a plain block on old Streamlit)."""
    if hasattr(st, "fragment"):
        return st.fragment(run_every=every)
    return lambda f: f


def label(name):
    return LABELS.get(name, name.replace("_", " ").title())


# ------------------------------------------------------------------ page
cfg = load_cfg()
_dash = cfg.get("dashboard") or {}
TITLE = _dash.get("title", "CCTV Video Analytics")
SUBTITLE = _dash.get("subtitle", "Live monitoring and alert history")
FOOTER = _dash.get("footer", "")
PRODUCT = _dash.get("product", "Video analytics")
CAMERA_NAME = (cfg.get("camera") or {}).get("name", "camera-1")


def cam_disp(n):
    """camera-1 -> CAM 01 (other names are shown as they are)"""
    m = re.fullmatch(r"(?i)\s*cam(?:era)?[-_ ]*(\d+)\s*", str(n or CAMERA_NAME))
    return f"CAM {int(m.group(1)):02d}" if m else str(n or CAMERA_NAME)


CAM_LABEL = cam_disp(CAMERA_NAME)
LOGO = _dash.get("logo", "")                 # optional image file, relative to the project folder
st.set_page_config(page_title=TITLE, layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
[data-testid="stSidebarCollapseButton"] {display: none;}      /* sidebar always open */
.block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1600px;}

/* ---------- cards ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff; border: 1px solid #E3E8F0 !important; border-radius: 14px;
    box-shadow: 0 1px 2px rgba(15, 27, 45, .05), 0 4px 14px rgba(15, 27, 45, .04);
}
.kpi {background: #fff; border: 1px solid #E3E8F0; border-left: 5px solid var(--c, #1F5FBF); border-radius: 12px;
    padding: 14px 18px; box-shadow: 0 1px 2px rgba(15,27,45,.05), 0 4px 14px rgba(15,27,45,.04);}
.kpi .l {color: #5B6778; font-size: .74rem; font-weight: 700; letter-spacing: .8px; text-transform: uppercase;}
.kpi .v {color: #0F1B2D; font-size: 2rem; font-weight: 700; line-height: 1.15; margin-top: 2px;}
.kpi .s {color: #7A8696; font-size: .8rem; margin-top: 2px;}

/* ---------- top bar ---------- */
.topbar {display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap;
    padding: 4px 0 14px 0; margin-bottom: 10px; border-bottom: 1px solid #E3E8F0;}
.topbar h1 {margin: 0; padding: 0; font-size: 1.6rem; font-weight: 700; color: #0F1B2D; letter-spacing: .2px;}
.topbar .sub {color: #5B6778; font-size: .9rem; margin-top: 2px;}
.chips {display: flex; gap: 8px; flex-wrap: wrap; align-items: center;}

/* ---------- chips ---------- */
.chip {display: inline-flex; align-items: center; gap: 6px; padding: 3px 11px; border-radius: 999px;
    font-size: .76rem; font-weight: 600; border: 1px solid transparent; white-space: nowrap;}
.chip::before {content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor;}
.chip.ok::before {animation: pulse 1.8s infinite;}
@keyframes pulse {0% {box-shadow: 0 0 0 0 rgba(22,122,59,.45);} 70% {box-shadow: 0 0 0 6px rgba(22,122,59,0);} 100% {box-shadow: 0 0 0 0 rgba(22,122,59,0);}}
.chip.ok   {background: #E6F6EC; color: #167A3B; border-color: #BFE5CC;}
.chip.warn {background: #FFF4DD; color: #9A6200; border-color: #F3DDA8;}
.chip.bad  {background: #FDE8E8; color: #B42318; border-color: #F5C2C0;}
.chip.idle {background: #EEF1F6; color: #5B6778; border-color: #DCE2EC;}

/* ---------- live tiles ---------- */
.tile-head {display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;}
.tile-title {font-weight: 650; font-size: 1rem; color: #0F1B2D;}
.counts {display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0 2px 0;}
.count {background: #F1F5FB; border: 1px solid #DDE6F3; border-radius: 10px; padding: 4px 12px; font-size: .85rem; color: #1B2430;}
.count b {color: #0F1B2D; margin-left: 4px;}
.alert-banner {margin-top: 10px; padding: 8px 12px; border-radius: 10px; background: #FDE8E8; color: #B42318;
    border: 1px solid #F5C2C0; font-weight: 600; font-size: .9rem;}
.quiet {color: #7A8696; font-size: .85rem; margin-top: 10px;}
.empty {text-align: center; padding: 56px 12px; color: #5B6778;}
.empty h3 {color: #0F1B2D; margin-bottom: 4px;}

/* ---------- recent alerts feed ---------- */
.panel-title {font-weight: 700; font-size: .95rem; color: #0F1B2D; margin-bottom: 10px;
    display: flex; justify-content: space-between; align-items: center;}
.feed-item {display: flex; gap: 10px; padding: 9px 0; border-top: 1px solid #EDF0F5; align-items: center;}
.feed-item:first-of-type {border-top: none;}
.feed-item img {width: 84px; height: 56px; object-fit: cover; border-radius: 8px; border: 1px solid #E3E8F0; flex: none;}
.feed-item .th {position: relative; display: block; width: 84px; height: 56px; flex: none; cursor: zoom-in;}
/* hover a thumbnail: the same picture opens large right there, over the page (like the alert log popup) */
.feed-item .th:hover {z-index: 999999;}
.feed-item .th:hover img {position: absolute; right: -4px; top: -4px; width: 480px; max-width: 60vw; height: auto;
  border: 1px solid #3B82F6; border-radius: 6px; background: #fff; box-shadow: 0 8px 28px rgba(0,0,0,.35);
  object-fit: contain; pointer-events: none; z-index: 999999;}
.feed-item .ph {width: 84px; height: 56px; border-radius: 8px; background: #EEF1F6; flex: none;}
.feed-item .t {font-weight: 650; font-size: .86rem; color: #0F1B2D; line-height: 1.2;}
.feed-item .e {font-size: .8rem; color: #B42318; font-weight: 600;}
.feed-item .w {font-size: .74rem; color: #7A8696;}
.feed-item .why {font-size: .76rem; color: #3D4A5C; line-height: 1.25; margin: 1px 0 2px;}
.feed-item .t {font-size: .95rem;}

/* ---------- tabs ---------- */
[data-baseweb="tab-list"] {gap: 6px;}
[data-baseweb="tab"] {font-weight: 600; padding: 8px 14px;}

/* ---------- dark sidebar ---------- */
[data-testid="stSidebar"] {background: #0F1B2D; border-right: none;}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span:not(.chip),
[data-testid="stSidebar"] li {color: #DCE5F3 !important;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color: #8EA0BB !important;}
[data-testid="stSidebar"] .stButton button {background: #1A2B46; border: 1px solid #2C4163; color: #E6ECF5; border-radius: 9px; font-weight: 600;}
[data-testid="stSidebar"] .stButton button:hover {border-color: #4F7BC4; color: #fff;}
[data-testid="stSidebar"] .stButton button[kind="primary"] {background: #2468D8; border-color: #2468D8; color: #fff;}
[data-testid="stSidebar"] .stButton button:disabled {opacity: .4;}
[data-testid="stSidebar"] .stButton button p {white-space: normal; line-height: 1.15;}      /* long names like "Face recognition" wrap instead of being cut */
[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button {background: #1A2B46; border: 1px solid #2C4163; color: #E6ECF5 !important; border-radius: 9px; font-weight: 600;}
[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button:hover {border-color: #4F7BC4; color: #fff !important;}
[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {background: #2468D8; border-color: #2468D8; color: #fff !important;}
[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button p {color: inherit !important;}
[data-testid="stSidebar"] [data-testid="stExpander"] details {background: #14233B; border: 1px solid #2C4163; border-radius: 10px;}
[data-testid="stSidebar"] [data-testid="stExpander"] summary, [data-testid="stSidebar"] [data-testid="stExpander"] summary * {color: #E6ECF5 !important; background: transparent !important;}
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stForm"] {border-color: #2C4163;}
.brand {display: flex; align-items: center; gap: 10px; margin: 2px 0 14px 0;}
.brand .name {font-weight: 700; font-size: 1.1rem; color: #FFFFFF; line-height: 1.15;}
.brand .tag {color: #8EA0BB; font-size: .76rem;}
.side-label {font-size: .7rem; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase;
    color: #6F85A6; margin: 20px 0 6px 0;}
.footer {color: #6F85A6; font-size: .76rem; text-align: center; margin-top: 28px;}

/* ---------- login / forms ---------- */
[data-testid="stForm"] {border: none; padding: 0;}
</style>
""", unsafe_allow_html=True)

DASH_USER = os.environ.get("DASHBOARD_USER", "admin")
DASH_PASS = os.environ.get("DASHBOARD_PASSWORD", "")


def esc(x):
    return html.escape(str(x))


def pretty(text):
    return str(text).replace("_", " ").strip().capitalize()


_WORN = {"cap", "gloves", "shoes", "mask", "helmet", "vest", "apron", "goggles", "hairnet", "glasses"}


def alert_reason(event, count=None):
    """(short label, colour, reason sentence) for an alert, e.g. no_cap -> Not wearing / No cap detected"""
    e = str(event).strip()
    low = e.lower()
    try:
        n = int(count)
    except (TypeError, ValueError):
        n = 0
    if low == "unknown":
        return "Unknown", "idle", "Face not in the saved faces"
    if low.startswith("no_") or low.startswith("no "):
        item = low[3:].replace("_", " ").strip()
        why = f"No {item} detected" + (f" on {n} people" if n > 1 else "")
        return ("Not wearing" if item in _WORN else "Missing"), "bad", why
    if " above " in low:
        lab, lim = low.split(" above ", 1)
        return "Threshold exceeded", "warn", f"{pretty(lab)} count {n} is above the limit of {lim}"
    if " below " in low:
        lab, lim = low.split(" below ", 1)
        return "Below minimum", "warn", f"{pretty(lab)} count is below the minimum of {lim}"
    if low == "beard":
        return "Beard visible", "bad", "Unshaved face seen" + (f" ({n} people)" if n > 1 else "")
    return pretty(e), "bad", pretty(e) + (f" ({n})" if n > 1 else "")


def chip(text, kind="idle"):
    return f'<span class="chip {kind}">{esc(text)}</span>'


def kpi(label_, value, sub="", color="#1F5FBF"):
    return (f'<div class="kpi" style="--c:{color}"><div class="l">{esc(label_)}</div>'
            f'<div class="v">{esc(value)}</div><div class="s">{esc(sub) or "&nbsp;"}</div></div>')


def ago(ts):
    secs = max(0, int((datetime.now() - ts).total_seconds()))
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60} min ago"
    if secs < 86400:
        return f"{secs // 3600} h ago"
    return ts.strftime("%d %b, %H:%M")


@st.cache_data(max_entries=400, show_spinner=False)
def thumb_uri(path_str, mtime, width=160):
    """Small base64 thumbnail of a snapshot (cached) for the feed and the alert table."""
    import base64
    import cv2
    im = cv2.imread(path_str)
    if im is None:
        return ""
    h, w = im.shape[:2]
    im = cv2.resize(im, (width, max(1, int(h * width / w))))
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def snap_uri(name, width=160):
    p = OUT / "snapshots" / str(name)
    try:
        return thumb_uri(str(p), p.stat().st_mtime, width) if p.exists() else ""
    except OSError:
        return ""


def brand_html():
    return (f'<div class="brand"><div><div class="name">{esc(TITLE)}</div>'
            f'<div class="tag">{esc(SUBTITLE)}</div></div></div>')


def login_gate():
    """Simple client login. User and password come from .env (DASHBOARD_USER / DASHBOARD_PASSWORD)."""
    if not DASH_PASS or st.session_state.get("auth"):
        return
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        st.write("")
        st.write("")
        with st.container(border=True):
            has_logo = bool(LOGO) and (ROOT / LOGO).exists()
            if has_logo:
                st.image(str(ROOT / LOGO), width=260)
            st.markdown(f'<div class="topbar"><div>{"" if has_logo else "<h1>" + esc(TITLE) + "</h1>"}'
                        f'<div class="sub">Please sign in to continue</div></div></div>', unsafe_allow_html=True)
            with st.form("login"):
                u = st.text_input("User")
                pw = st.text_input("Password", type="password")
                go = st.form_submit_button("Sign in", type="primary", **WIDE)
            if go:
                ok = hmac.compare_digest(u.encode(), DASH_USER.encode()) & \
                    hmac.compare_digest(pw.encode(), DASH_PASS.encode())
                if ok:
                    st.session_state["auth"] = True
                    st.rerun()
                time.sleep(1.5)                     # slows down password guessing
                st.error("Wrong user or password.")
    st.stop()


login_gate()

avail = availability(cfg)
url, shown = rtsp_source(cfg)
test_src = os.environ.get("CCTV_TEST_SOURCE")
if url and not test_src:                       # the detectors tag every alert with this camera
    CAMERA_NAME = register_cam(url)[0]["id"]
    CAM_LABEL = cam_disp(CAMERA_NAME)
    os.environ["CAMERA_NAME"] = CAMERA_NAME

for n in LABELS:
    st.session_state.setdefault(f"uc_{n}", False)
if "init_done" not in st.session_state:          # a new browser tab shows what is already running
    st.session_state["init_done"] = True
    if running():
        for n in LABELS:
            st.session_state[f"uc_{n}"] = n in ctl()["names"] and avail[n][0]


def _apply_preset(names):
    for n in LABELS:
        st.session_state[f"uc_{n}"] = n in names and avail[n][0]


st.markdown("""
<style>
/* ===================== restyle: top bar, sidebar navigation, cards ===================== */
header[data-testid="stHeader"] {display: none;}
.block-container {padding-top: 4.3rem !important; max-width: 1700px;}
.swbar {position: fixed; top: 0; left: 0; right: 0; height: 64px; z-index: 1000002; background: #fff;
    border-bottom: 1px solid #E3E8F0; display: flex; align-items: center; gap: 18px; padding: 0 24px;}
.swbar .logo {font-weight: 800; font-size: 1.45rem; color: #0F1B2D; letter-spacing: -.2px; white-space: nowrap;}
.swbar .logo {color: #0B2A5B;} .swbar .logo .w2 {color: #1F6FE0;}
.swbar .logo img {height: 32px; vertical-align: middle; display: block;}
.swbar .sep {width: 1px; height: 28px; background: #E3E8F0;}
.swbar .prod {color: #5B6778; font-size: 1.05rem; white-space: nowrap;}
.swbar .sp {flex: 1;}
.swbar .date {color: #5B6778; font-size: .9rem; white-space: nowrap;}
.swbar .av {width: 38px; height: 38px; border-radius: 50%; background: #DCE8FB; color: #1F5FBF; font-weight: 700;
    display: flex; align-items: center; justify-content: center; font-size: .95rem;}
.swbar a.out {color: #0F1B2D; font-weight: 600; text-decoration: none;}
.swbar a.out:hover {color: #1F5FBF;}
@media (max-width: 900px) {.swbar .prod, .swbar .date {display: none;}}

/* sidebar sits below the top bar */
[data-testid="stSidebar"] {top: 64px !important; height: calc(100vh - 64px) !important;}
[data-testid="stSidebar"] > div:first-child {height: 100%;}
[data-testid="stSidebarHeader"] {display: none;}
[data-testid="stSidebarUserContent"] {padding-top: 1rem;}
[data-testid="stSidebar"] .stButton button {width: 100%; justify-content: flex-start; text-align: left; background: transparent !important;
    border: none !important; color: #DCE5F3 !important; padding: 10px 12px; font-weight: 600; box-shadow: none;}
[data-testid="stSidebar"] .stButton button:hover {background: #1A2B46 !important; color: #fff !important;}
[data-testid="stSidebar"] .stButton button[kind="primary"] {background: #1F3556 !important; color: #fff !important;}
[data-testid="stSidebar"] .stButton button p {color: inherit !important;}
[data-testid="stSidebar"] [data-testid="stCheckbox"] label {width: 100%; flex-direction: row !important; justify-content: space-between;
    padding: 6px 8px; border-radius: 8px; gap: 8px;}
[data-testid="stSidebar"] [data-testid="stCheckbox"] label:hover {background: #14233B;}
[data-testid="stSidebar"] [data-testid="stCheckbox"] [data-testid="stWidgetLabel"] {order: 1; flex: 1; width: 100%;}
[data-testid="stSidebar"] [data-testid="stCheckbox"] label > div:first-of-type {order: 2; margin-left: auto; flex: none;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap: .3rem;}
[data-testid="stSidebar"] [data-testid="stCheckbox"], [data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stCheckbox"]) {width: 100% !important;}
[data-testid="stSidebar"] [data-testid="stCheckbox"] label p {font-size: 1rem; display: flex; align-items: center; gap: 12px;}
[data-testid="stSidebar"] span[data-testid="stIconMaterial"] {font-size: 1.4rem !important; color: #A9B8D0;}
[data-testid="stSidebar"] .stButton button > div {justify-content: flex-start !important; width: 100%;}
[data-testid="stSidebar"] .stButton button p {text-align: left; font-size: 1rem; display: flex; align-items: center; gap: 12px;}
.side-sep {height: 1px; background: #223552; margin: 12px 0;}
.side-out {text-decoration: none !important; display: block; margin-top: 18px; padding: 10px 12px; color: #DCE5F3 !important; text-decoration: none; font-weight: 600; border-radius: 8px;}
.side-out:hover {background: #1A2B46;}

/* ---------- KPI row ---------- */
.kp {display: flex; gap: 14px; align-items: center; background: #fff; border: 1px solid #E3E8F0; border-radius: 14px;
    padding: 14px 18px; box-shadow: 0 1px 2px rgba(15,27,45,.05), 0 4px 14px rgba(15,27,45,.04);}
.kp .ic {width: 46px; height: 46px; border-radius: 12px; background: var(--bg, #E8F0FD); color: var(--c, #1F5FBF);
    display: flex; align-items: center; justify-content: center; flex: none;}
.kp .v.long {font-size: 1.25rem; padding: 6px 0;}
.kp .v {font-size: 1.9rem; font-weight: 700; color: #0F1B2D; line-height: 1.05;}
.kp .l {color: var(--c, #1F5FBF); font-weight: 600; font-size: .9rem;}
.kp .s {color: #7A8696; font-size: .78rem;}

/* ---------- banner, camera placeholder, camera list ---------- */
.warnbar {display: flex; align-items: center; gap: 10px; background: #FFF6E0; border: 1px solid #F3DDA8; border-left: 4px solid #E8A317;
    color: #9A6200; padding: 11px 14px; border-radius: 10px; font-size: .92rem; margin-bottom: 6px;}
.camwrap {position: relative; border-radius: 12px; overflow: hidden; line-height: 0; background: #0A2C55;}
.camwrap img {width: 100%; display: block;}
.camwrap .camaddr {position: absolute; left: 0; right: 0; bottom: 0; padding: 26px 14px 10px; line-height: 1.2; color: #fff; font-size: .82rem;
    background: linear-gradient(0deg, rgba(5,20,40,.78), rgba(5,20,40,0));}
.camwrap a.camfs, .camwrap a.camclose {position: absolute; right: 10px; width: 34px; height: 34px; border-radius: 8px; color: #fff !important;
    display: flex; align-items: center; justify-content: center; background: rgba(5,20,40,.45); text-decoration: none !important;}
.camwrap a.camfs {bottom: 8px;}
.camwrap a.camclose {top: 10px; display: none;}
.camwrap a.camfs:hover, .camwrap a.camclose:hover {background: rgba(15,157,143,.9);}
.camwrap:target {position: fixed; inset: 0; z-index: 2147483000; border-radius: 0; background: #000; display: flex; align-items: center; justify-content: center;}
.camwrap:target img {width: 100%; height: 100vh; object-fit: contain;}
.camwrap:target a.camfs {display: none;}
.camwrap:target a.camclose {display: flex;}
.selcam {color: #1F5FBF; font-size: .85rem; margin: 8px 0 4px;}
.phcam {background: linear-gradient(180deg, #14233B, #0F1B2D); border-radius: 12px; color: #DCE5F3; text-align: center;
    display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative; padding: 24px 12px;}
.phcam .t {font-weight: 700; font-size: 1.15rem; color: #fff; margin-top: 10px;}
.phcam .s {color: #8EA0BB; font-size: .88rem; margin-top: 4px;}
.phcam .addr {position: absolute; left: 14px; bottom: 10px; color: #A9B8D0; font-size: .8rem;}
.cam {border: 1px solid #E3E8F0; border-radius: 12px; padding: 11px 13px; margin-bottom: 10px; background: #fff;}
.cam.sel {border-color: #2468D8; background: #EEF4FF;}
.cam .n {font-weight: 700; color: #0F1B2D;}
.cam .n small {color: #5B6778; font-weight: 600; margin-right: 6px;}
.cam .st {font-size: .8rem; color: #7A8696; display: flex; align-items: center; gap: 6px; margin: 3px 0 4px 0;}
.tag {display: inline-block; background: #EEF1F6; border-radius: 7px; padding: 2px 9px; font-size: .74rem; color: #1B2430; margin: 3px 5px 0 0;}
.dot {width: 8px; height: 8px; border-radius: 50%; background: #98A2B3; display: inline-block; flex: none;}
.dot.ok {background: #167A3B;} .dot.bad {background: #D92D20;} .dot.sel {background: #2468D8;} .dot.wait {background: #E8A317;}
.card-title {font-weight: 700; font-size: 1.05rem; color: #0F1B2D;}
.card-title small {color: #1F5FBF; font-weight: 600; margin-left: 8px; font-size: .86rem;}

/* ---------- detection workspace ---------- */
.wsgrid {display: grid; grid-template-columns: repeat(3, 1fr); gap: 0 26px; margin-top: 6px;}
@media (max-width: 1100px) {.wsgrid {grid-template-columns: 1fr;}}
.wsrow {display: flex; align-items: center; gap: 10px; padding: 9px 4px; border-bottom: 1px solid #EDF0F5;}
.wsrow .nm {flex: 1; color: #0F1B2D; font-weight: 600; font-size: .92rem; line-height: 1.2;}
.wsrow .nm small {display: block; color: #7A8696; font-weight: 400; font-size: .74rem;}
.wsrow .stt {font-size: .82rem; color: #5B6778; display: flex; align-items: center; gap: 7px; white-space: nowrap;}
.wsfoot {color: #5B6778; font-size: .86rem; margin-top: 10px; display: flex; align-items: center; gap: 8px;}
.st-key-presets button {border-radius: 999px !important; padding: 2px 14px !important; min-height: 38px;}
.st-key-presets button[kind="primary"] {background: #E7F0FE !important; color: #1F5FBF !important; border: 1px solid #9DBDF2 !important;}
.st-key-presets button[kind="secondary"] {background: #fff; color: #1B2430; border: 1px solid #DCE2EC;}

/* ---------- recent alerts ---------- */
.feed-item .pill {display: inline-block; font-size: .72rem; font-weight: 600; padding: 1px 9px; border-radius: 999px; margin: 2px 0;}
.pill.bad {background: #FDE8E8; color: #B42318;} .pill.idle {background: #EEF1F6; color: #5B6778;} .pill.warn {background: #FFF4DD; color: #9A6200;}
.pill.ok {background: #E6F6EC; color: #167A3B;}

/* ---------- alert details / logs ---------- */
.kv {display: flex; justify-content: space-between; gap: 12px; padding: 8px 0; border-top: 1px solid #EDF0F5; font-size: .9rem;}
.kv span:first-child {color: #5B6778;} .kv span:last-child {color: #0F1B2D; font-weight: 600; text-align: right;}
.lvl {display: inline-block; font-size: .72rem; font-weight: 700; padding: 2px 9px; border-radius: 7px;}
.lvl.INFO {background: #E0F4F1; color: #0B7A6B;} .lvl.WARN {background: #FFF4DD; color: #9A6200;} .lvl.ERROR {background: #FDE8E8; color: #B42318;}
.camchip {background: #E3ECFB; color: #1F4F9F; font-weight: 700; font-size: .78rem; padding: 3px 10px; border-radius: 8px; align-self: flex-start;}
.logwrap {max-height: 380px; overflow: auto;}
table.logt {width: 100%; border-collapse: collapse; font-size: .92rem; border: none !important;}
table.logt tr, table.logt tbody tr:nth-child(even) {background: transparent !important;}
table.logt th, table.logt td {border-left: none !important; border-right: none !important;}
table.logt th {position: sticky; top: 0; background: #fff; text-align: left; color: #3D4A5C; font-weight: 600; padding: 10px 12px; border-bottom: 1px solid #E3E8F0;}
table.logt td {padding: 9px 12px; border-top: 1px solid #EDF0F5; color: #1B2430; vertical-align: middle;}
table.logt td.tm {color: #3D4A5C; white-space: nowrap;}
.lvb {display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 8px; font-weight: 700; font-size: .78rem;}
.lbicon {width: 52px; height: 52px; border-radius: 12px; background: #FFF1E0; color: #E8890C; display: flex; align-items: center; justify-content: center;}
.scard {display: flex; align-items: center; gap: 14px; background: #fff; border: 1px solid #E3E8F0; border-radius: 14px; padding: 16px 18px;
    box-shadow: 0 1px 2px rgba(15,27,45,.05), 0 4px 14px rgba(15,27,45,.04);}
.scard .ic {width: 46px; height: 46px; border-radius: 12px; display: flex; align-items: center; justify-content: center; flex: none;
    background: var(--bg, #E8F0FD); color: var(--c, #1F5FBF);}
.scard .k {color: #5B6778; font-size: .82rem;} .scard .v {color: #0F1B2D; font-weight: 700; font-size: 1.15rem;} .scard .d {color: #7A8696; font-size: .8rem;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- new-look helpers
_V = tuple(int(x) for x in (st.__version__.split(".") + ["0", "0"])[:2])
HAS_ICONS = _V >= (1, 37)        # :material/name: icons inside labels
HAS_KEY = _V >= (1, 39)          # st.container(key=...) gives a CSS class


def ic(name, text):
    """Label with a material icon when this Streamlit supports it."""
    return f":material/{name}: {text}" if HAS_ICONS else text


def kcontainer(key, **kw):
    return st.container(key=key, **kw) if HAS_KEY else st.container(**kw)


_SVG = {
    "err": '<circle cx="12" cy="12" r="9"/><path d="M12 7v6M12 16v.5"/>',
    "play": '<path d="M7 4.5v15l12-7.5z"/>',
    "pause": '<path d="M8 5v14M16 5v14"/>',
    "download": '<path d="M12 4v11M7 11l5 5 5-5M5 20h14"/>',
    "pencil": '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13.500 6.500l4 4"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8v.5"/>',
    "expand": '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
    "close": '<path d="M5 5l14 14M19 5 5 19"/>',
    "layers": '<path d="M12 3 3 8l9 5 9-5-9-5z"/><path d="M3 13l9 5 9-5"/>',
    "alert": '<path d="M12 3 2 20h20L12 3z"/><path d="M12 10v4M12 17v.5"/>',
    "bell": '<path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10 21h4"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "cam": '<path d="M3 7h12a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2H3z"/><path d="M17 11l4-3v8l-4-3"/>',
    "wifi": '<path d="M2 9a15 15 0 0 1 20 0M5 12.5a10 10 0 0 1 14 0M8.5 16a5 5 0 0 1 7 0"/><path d="M12 19.5v.5"/>',
}


def svg(name, size=22):
    return (f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="currentColor" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round">{_SVG[name]}</svg>')


def kp(icon, value, label_, sub, color="#1F5FBF", bg="#E8F0FD"):
    return (f'<div class="kp" style="--c:{color};--bg:{bg}"><div class="ic">{svg(icon)}</div><div>'
            f'<div class="v{" long" if len(str(value)) > 7 else ""}">{esc(value)}</div><div class="l">{esc(label_)}</div><div class="s">{esc(sub) or "&nbsp;"}</div></div></div>')


def short(n):
    return label(n).split(" (")[0]


UC_ICON = {"person_count": "person", "cap_detection": "sports_baseball", "gloves_detection": "back_hand",
           "button_detection": "checkroom", "shoes_detection": "steps", "mask_detection": "masks",
           "beard_detection": "face_6", "object_detection": "deployed_code", "face_recognition": "face"}
UC_SUB = {"object_detection": "laptop, chair, diary, phone"}
PRESET_ORDER = ["All", "Safety kit", "Office", "Dress code", "Face recognition", "None"]


def _goto(page):
    st.session_state["page"] = page


def _go_main():
    """Dashboard View button: back to the main page, Live view tab."""
    st.session_state["page"] = "dash"
    st.session_state["_goto_live"] = True


def _use_cam(cid):
    c = next((x for x in load_cams() if x["id"] == cid), None)
    if running() or ctl()["wanted"]:
        st.session_state["cam_msg"] = ("error", "Stop the analytics first (Stop button below), then select another camera.")
    elif not c or not c.get("url"):
        st.session_state["cam_msg"] = ("error", f"{cid} has no saved address. Add it again with + Add camera.")
    else:
        save_rtsp(c["url"])
        st.session_state["cam_msg"] = ("success", f"{cid} selected. Press Start to use it.")


def _toggle_cam_edit():
    st.session_state["cam_edit"] = not st.session_state.get("cam_edit")
    st.session_state["cam_target"] = None


def _edit_cam(cid):
    if running() or ctl()["wanted"]:
        st.session_state["cam_msg"] = ("error", "Stop the analytics first (Stop button below), then edit the camera.")
        return
    st.session_state["cam_edit"], st.session_state["cam_target"] = True, cid
    st.session_state["cam_msg"] = (None, "")


def _ask_del(cid):
    if running() or ctl()["wanted"]:
        st.session_state["cam_msg"] = ("error", "Stop the analytics first (Stop button below), then remove the camera.")
        return
    st.session_state["cam_del"] = cid


def _open_cam_edit():
    st.session_state["page"] = "dash"
    st.session_state["cam_edit"] = True


def cam_status(stats, now):
    """(text, dot class) for the camera."""
    c = ctl()
    if test_src:
        return "Test source", "wait"
    if not url:
        return "Not set", "bad"
    if running():
        fresh = any(now - s["timestamp"] < FRESH_SECONDS for s in stats.values())
        return ("Receiving frames", "ok") if fresh else ("Waiting for frames", "wait")
    if c["wanted"]:
        return "Restarting", "wait"
    return "Stopped", "off"


def active_names(stats, now):
    c = ctl()
    if running() or c["wanted"]:
        return list(c["names"])
    return [n for n in stats if now - stats[n]["timestamp"] < FRESH_SECONDS]   # run from another window


def camwrap(jpg, addr=""):
    b64 = base64.b64encode(jpg).decode()
    return ('<div class="camwrap" id="camfs">'
            f'<img src="data:image/jpeg;base64,{b64}" alt="camera">'
            + (f'<div class="camaddr">{addr_line(addr)}</div>' if addr else "")
            + f'<a class="camfs" href="#camfs" title="Full screen">{svg("expand", 20)}</a>'
            f'<a class="camclose" href="#" title="Close full screen">{svg("close", 20)}</a></div>')


def camwrap_stream(url, addr=""):
    """Same card as camwrap(), but the picture is the live video stream. The HTML is identical on every refresh,
    so the browser keeps playing the stream (no flicker, also not in full screen)."""
    return ('<div class="camwrap" id="camfs">'
            f'<img src="{esc(url)}" alt="camera">'
            + (f'<div class="camaddr">{addr_line(addr)}</div>' if addr else "")
            + f'<a class="camfs" href="#camfs" title="Full screen">{svg("expand", 20)}</a>'
            f'<a class="camclose" href="#" title="Close full screen">{svg("close", 20)}</a></div>')


def stream_url():
    """Address of the live video stream (written by the detectors), or None -> the dashboard shows still pictures."""
    try:
        f = OUT / "stream.json"
        info = json.loads(f.read_text())
        if f.stat().st_mtime < ctl()["started"] - 5:
            return None                         # left over from an earlier run
        port, bind = int(info["port"]), str(info.get("host") or "127.0.0.1")
    except (OSError, ValueError, KeyError, TypeError):
        return None
    loop = ("127.0.0.1", "localhost", "::1")
    try:
        host = (st.context.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
    except Exception:
        host = ""
    host = host or "127.0.0.1"
    if host in loop:
        host = "127.0.0.1"
    elif bind in loop:
        return None                             # opened from another computer, but the stream is local-only
    return f"http://{host}:{port}/live.mjpg"


def selcam():
    return f'<div class="selcam">Selected camera: {esc(CAM_LABEL)}</div>'


def placeholder(title, sub, height, addr=""):
    return (f'<div class="phcam" style="min-height:{height}px">{svg("cam", 52)}<div class="t">{esc(title)}</div>'
            f'<div class="s">{esc(sub)}</div>' + (f'<div class="addr">{esc(addr)}</div>' if addr else "") + '</div>')


# ---- system log parsing (run.log has no time stamps, so the table shows level / source / message)
_STAMP = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\|([^\]]*)\]\s?(.*)$")


def parse_log(lines):
    out = []
    for ln in lines:
        t = ln.strip()
        if not t:
            continue
        when, cam = None, ""
        m = _STAMP.match(t)
        if m:
            try:
                when = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                when = None
            cam, t = cam_disp(m.group(2)) if m.group(2) else "", m.group(3).strip()
        low = t.lower()
        if t.startswith("[SKIP]"):
            lvl, src, msg = "ERROR", "Analytics", t[6:].strip()
        elif t.startswith("[ok]"):
            lvl, src, msg = "INFO", "Analytics", t[4:].strip()
        elif t.startswith("[stream]"):
            lvl, src, msg = ("WARN" if "waiting" in low or "lost" in low or "retry" in low or "cannot" in low else "INFO"), "Camera", t[8:].strip()
        elif t.startswith("[EVENT]"):
            lvl, src, msg = "INFO", "Alerts", t[7:].strip()
        elif t.startswith("[dashboard]"):
            lvl, src, msg = ("WARN" if "exited" in low else "INFO"), "System", t[11:].strip()
        elif t.startswith("[warn]") or "warning" in low:
            lvl, src, msg = "WARN", "System", t.replace("[warn]", "").strip()
        elif "traceback" in low or "error" in low or "exception" in low:
            lvl, src, msg = "ERROR", "System", t
        else:
            lvl, src, msg = "INFO", "System", t
        out.append({"When": when, "Camera": cam, "Level": lvl, "Source": src, "Message": msg})
    return out[::-1]                                  # newest first


# ---------------------------------------------------------------- colour theme + line icons (as in the design)
_I = {
    "dash": '<rect x="3" y="3" width="7" height="7" rx="1.6"/><rect x="14" y="3" width="7" height="7" rx="1.6"/><rect x="3" y="14" width="7" height="7" rx="1.6"/><rect x="14" y="14" width="7" height="7" rx="1.6"/>',
    "person": '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4.4 3.6-7 8-7s8 2.6 8 7"/>',
    "cap": '<path d="M4.5 16a7.5 7.5 0 0 1 15 0"/><path d="M2.5 16h19l-2 3h-15z"/><path d="M12 8.5V7"/>',
    "gloves": '<path d="M8 13V6a1.5 1.5 0 0 1 3 0v5M11 11V4.5a1.5 1.5 0 0 1 3 0V11M14 11V6a1.5 1.5 0 0 1 3 0v7"/><path d="M17 13v-1.5a1.5 1.5 0 0 1 3 0V15c0 4-3 7-7 7h-1c-3 0-4.6-1.8-6-4.2L3.5 14.600a1.600 1.600 0 0 1 2.700-1.700L8 15.500"/>',
    "shirt": '<path d="M8.500 3 3 6l2.500 4L8 9v12h8V9l2.500 1L21 6l-5.500-3a3.500 3.500 0 0 1-7 0z"/>',
    "shoes": '<path d="M3 17v-5l5-1 3 3h4c3 0 6 1 6 3v0z"/><path d="M3 20h18"/><path d="M8 11V8"/>',
    "mask": '<path d="M4 8c5-2 11-2 16 0v5c0 3.500-3.500 6-8 6s-8-2.500-8-6z"/><path d="M2 10.500h2M20 10.500h2"/><path d="M8 12.500h8M8.500 15.500h7"/>',
    "beard": '<path d="M5 9a7 7 0 0 1 14 0v3c0 5-3 9-7 9s-7-4-7-9z"/><path d="M9.500 10v.5M14.500 10v.5"/><path d="M9 15.500c1.500 1.200 4.500 1.200 6 0"/>',
    "objects": '<path d="M12 3 3 7.500v9L12 21l9-4.500v-9z"/><path d="M3 7.500 12 12l9-4.500M12 12v9"/>',
    "face": '<path d="M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M8 20H6a2 2 0 0 1-2-2v-2"/><path d="M9 10v1M15 10v1M12 10v3.500h-1M9.500 16c1.500 1 3.500 1 5 0"/>',
    "doc": '<path d="M7 3h8l4 4v14H7z"/><path d="M15 3v4h4M10 12h6M10 16h6"/>',
    "shield": '<path d="M12 3 4 6v6c0 5 3.500 8 8 9 4.500-1 8-4 8-9V6z"/><path d="M9 12l2 2 4-4"/>',
    "briefcase": '<rect x="3" y="8" width="18" height="12" rx="2"/><path d="M9 8V5.500h6V8M3 13.500h18"/>',
    "pencil": '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13.500 6.500l4 4"/>',
    "trash": '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
    "none": '<circle cx="12" cy="12" r="9"/><path d="M5.600 5.600l12.800 12.800"/>',
    "chev_l": '<path d="M13 6l-6 6 6 6M19 6l-6 6 6 6"/>',
    "chev_r": '<path d="M5 6l6 6-6 6M11 6l6 6-6 6"/>',
}


def _icon_css(selector, key):
    from urllib.parse import quote
    svg_ = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="1.7" '
            'stroke-linecap="round" stroke-linejoin="round">' + _I[key] + '</svg>')
    uri = 'url("data:image/svg+xml;utf8,' + quote(svg_, safe="/:=' ") + '")'
    return (f'{selector} p::before {{content: ""; display: inline-block; width: 22px; height: 22px; flex: none; '
            f'background-color: currentColor; -webkit-mask: {uri} center / contain no-repeat; mask: {uri} center / contain no-repeat;}}\n')


_WS_KEY = {"person_count": "person", "cap_detection": "cap", "gloves_detection": "gloves", "button_detection": "shirt",
           "shoes_detection": "shoes", "mask_detection": "mask", "beard_detection": "beard", "object_detection": "objects",
           "face_recognition": "face"}


def _wsicon(n):
    return ('<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.7" '
            f'stroke-linecap="round" stroke-linejoin="round">{_I[_WS_KEY[n]]}</svg>') if n in _WS_KEY else ""


if "side_compact" not in st.session_state:
    st.session_state["side_compact"] = True          # auto-collapse by default
SIDE_COMPACT = bool(HAS_KEY and st.session_state["side_compact"])


def _toggle_side():
    st.session_state["side_compact"] = not st.session_state.get("side_compact", False)


_icons = ""
if HAS_KEY:
    _uc = {"person_count": "person", "cap_detection": "cap", "gloves_detection": "gloves", "button_detection": "shirt",
           "shoes_detection": "shoes", "mask_detection": "mask", "beard_detection": "beard", "object_detection": "objects",
           "face_recognition": "face"}
    for _n, _k in _uc.items():
        _icons += _icon_css(f'[data-testid="stSidebar"] .st-key-uc_{_n} label', _k)
    _icons += _icon_css('[class*="st-key-camed_"] button', "pencil")
    _icons += _icon_css('[class*="st-key-camdel_"] button', "trash")
    _icons += _icon_css('[data-testid="stSidebar"] .st-key-nav_dash button', "dash")
    _icons += _icon_css('[data-testid="stSidebar"] .st-key-nav_logs button', "doc")
    _icons += _icon_css('[data-testid="stSidebar"] .st-key-side_toggle button', "chev_r" if SIDE_COMPACT else "chev_l")
    for _p, _k in {"All": "dash", "Safety kit": "shield", "Office": "briefcase", "Dress code": "shirt",
                   "Face recognition": "face", "None": "none"}.items():
        _icons += _icon_css(f'.st-key-preset_{_p.replace(" ", "-")} button', _k)

st.markdown("""
<style>
:root {--teal: #0F9D8F; --teal-d: #0B8478; --teal-bg: #E3F5F2; --navy: #0B3A6E; --navy-2: #1D4F86; --navy-3: #12467F;}
.stApp {background: #F3F7FB;}
[data-testid="stSidebar"] {background: var(--navy) !important;}
[data-testid="stSidebar"] .stButton button:hover, [data-testid="stSidebar"] [data-testid="stCheckbox"] label:hover {background: var(--navy-3) !important;}
[data-testid="stSidebar"] .stButton button[kind="primary"] {background: var(--navy-2) !important;}
[data-testid="stSidebar"] [data-testid="stCheckbox"] label[data-selected="true"] > div:first-of-type {background-color: var(--teal) !important; border-color: var(--teal) !important;}
[data-testid="stSidebar"] span[data-testid="stIconMaterial"] {color: #CFE0F5;}
.side-sep {background: #2D62A3;} .side-label {color: #9DB6D8;}
[data-testid="stSidebar"] [data-testid="stCheckbox"] label p, [data-testid="stSidebar"] .stButton button p {color: #EAF2FC !important;}
[data-testid="stMain"] .stButton button[kind="primary"], [data-testid="stMain"] [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {
    background: var(--teal); border-color: var(--teal); color: #fff;}
[data-testid="stMain"] .stButton button[kind="primary"]:hover {background: var(--teal-d); border-color: var(--teal-d);}
[data-testid="stMain"] [data-testid="stDownloadButton"] button {background: #fff; color: var(--teal-d); border: 1px solid var(--teal);}
[data-testid="stMain"] [data-testid="stDownloadButton"] button:hover {background: var(--teal-bg);}
.st-key-presets button[kind="primary"] {background: var(--teal-bg) !important; color: var(--teal-d) !important; border: 1px solid #8BD3C9 !important;}
.st-key-presets button p {display: flex; align-items: center; gap: 6px; font-size: .84rem; white-space: nowrap; overflow: visible; text-overflow: clip;}
.st-key-presets button {padding-left: 8px; padding-right: 8px;}
[data-baseweb="tab-highlight"] {background-color: var(--teal) !important;}
button[role="tab"][aria-selected="true"], button[role="tab"][aria-selected="true"] p {color: var(--teal-d) !important;}
.cam.sel {border-color: var(--teal); background: #EAF8F5;}
.subcam {color: #5B6778; font-size: .8rem; margin: -4px 0 4px;}
.st-key-viewall button {background: transparent !important; border: none !important; box-shadow: none !important;
    color: var(--teal-d) !important; font-weight: 700; justify-content: flex-end; padding-right: 0;}
.st-key-viewall button:hover {text-decoration: underline;}
.wsgrid {gap: 10px 14px; align-items: start;}
.wsrow {background: #fff; border: 1px solid #DDE5F0; border-radius: 10px; padding: 10px 14px; gap: 12px;}
.wsrow .wi {color: #1F5FBF; display: flex; flex: none;}
.wsrow .nm small {margin-top: 2px;}
.cam .n {display: flex; align-items: center;}
.cam .n {white-space: nowrap;}
[class*="st-key-camsel_"], [class*="st-key-camact_"] {position: relative;}
[class*="st-key-use_"] {position: absolute; inset: 0 0 10px 0; width: 100%; height: auto; margin: 0; z-index: 2;}
[class*="st-key-use_"] button {width: 100%; height: 100%; opacity: 0; cursor: pointer; min-height: 0;}
[class*="st-key-use_"] button:disabled {cursor: not-allowed;}
[class*="st-key-camed_"], [class*="st-key-camdel_"] {position: absolute; top: 8px; width: auto; z-index: 3;}
[class*="st-key-camed_"] {right: 42px;} [class*="st-key-camdel_"] {right: 10px;}
[class*="st-key-camed_"] button, [class*="st-key-camdel_"] button {background: transparent !important; border: none !important; box-shadow: none !important;
    min-height: 0; height: 30px; width: 30px; padding: 0; display: flex; align-items: center; justify-content: center;}
[class*="st-key-camed_"] button p, [class*="st-key-camdel_"] button p {font-size: 0; line-height: 0; margin: 0;}
[class*="st-key-camed_"] button p::before, [class*="st-key-camdel_"] button p::before {width: 18px; height: 18px;}
[class*="st-key-camed_"] button {color: #1F5FBF !important;} [class*="st-key-camdel_"] button {color: #D92D20 !important;}
[class*="st-key-camed_"] button:hover, [class*="st-key-camdel_"] button:hover {background: rgba(15,157,143,.12) !important;}
.cam .n {padding-right: 68px;}
[class*="st-key-camsel_"]:hover .cam.pick {border-color: var(--teal); background: #F4FBFA;}
.cam .n .cnm {font-weight: 500; color: #3D4A5C; margin-left: 8px; overflow: hidden; text-overflow: ellipsis;}
.cam .n .radio.off {background: #fff; border-color: #B9C4D4; box-shadow: none;}
.cam .n .radio {display: inline-block; width: 18px; height: 18px; border-radius: 50%; border: 2px solid var(--teal); background: var(--teal);
    box-shadow: inset 0 0 0 3px #fff; margin-right: 10px; flex: none;}
.cam .tag, .tag {background: #E3ECFB; color: #1F4F9F; font-weight: 500;}
.infonote {background: #EAF1FD; border: 1px solid #CFE0FA; color: #1F4F9F; border-radius: 10px; padding: 10px 12px; font-size: .9rem; font-weight: 500;
    display: flex; gap: 8px; align-items: center; margin: 6px 0 12px;}
.infonote svg {flex: none;}
.st-key-btn_addcam button {white-space: nowrap; padding-left: 8px; padding-right: 8px;}
.st-key-btn_addcam button p {overflow: visible; text-overflow: clip; font-size: .86rem;}
.card-title {white-space: nowrap;}
.phcam {background: linear-gradient(180deg, #0E3A6C, #0A2C55);}
.swbar .av {background: #DDEFF9; color: var(--navy);}
.card-title small {color: var(--navy-2);}
</style>
""" + "<style>" + _icons + "</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar (navigation + use cases)
if st.query_params.get("signout"):              # the "Sign out" link: a fresh page load has no login any more
    st.session_state["auth"] = False
    st.query_params.clear()
    st.rerun()

with st.sidebar:
    if HAS_KEY:
        st.button("Keep menu open" if SIDE_COMPACT else "Auto-hide menu", key="side_toggle", on_click=_toggle_side)
    cur = st.session_state.get("page", "dash")
    st.button("Dashboard View" if HAS_KEY else ic("dashboard", "Dashboard View"), key="nav_dash", on_click=_go_main,
              type="primary" if cur == "dash" else "secondary", **WIDE)
    st.markdown('<div class="side-label">Analytics</div>', unsafe_allow_html=True)
    for n in LABELS:
        ok, note = avail[n]
        st.checkbox(short(n) if HAS_KEY else ic(UC_ICON.get(n, "analytics"), short(n)), key=f"uc_{n}", disabled=not ok)
    selected = [n for n in LABELS if st.session_state.get(f"uc_{n}") and avail[n][0]]
    st.markdown('<div class="side-sep"></div>', unsafe_allow_html=True)
    st.button("System logs" if HAS_KEY else ic("description", "System logs"), key="nav_logs", on_click=_goto, args=("logs",),
              type="primary" if cur == "logs" else "secondary", **WIDE)
    if not DASH_PASS:
        st.caption("No dashboard password set (DASHBOARD_PASSWORD in .env).")
    if FOOTER:
        st.markdown(f'<div class="footer">{esc(FOOTER)}</div>', unsafe_allow_html=True)


# collapse button (always) and the icon-only rail (when collapsed)
if HAS_KEY:
    _rail = """
[data-testid="stSidebar"] .st-key-side_toggle {display: flex; justify-content: flex-end;}
[data-testid="stSidebar"] .st-key-side_toggle button {width: 34px !important; min-height: 0; height: 32px; padding: 0 !important;
    justify-content: center !important; background: var(--navy-3) !important; border: none !important; border-radius: 8px;}
[data-testid="stSidebar"] .st-key-side_toggle button:hover {background: var(--navy-2) !important;}
[data-testid="stSidebar"] .st-key-side_toggle button p {font-size: 0 !important; gap: 0 !important; justify-content: center;}
"""
    if SIDE_COMPACT:
        # auto mode: a thin icon strip; the menu opens OVER the page while the mouse is on it (the page does not shift)
        _N = '[data-testid="stSidebar"]:not(:hover)'
        _rail += """
[data-testid="stSidebar"] {width: 76px !important; min-width: 76px !important; max-width: 76px !important; overflow: visible !important;}
[data-testid="stSidebar"] > div:first-child {overflow: visible !important;}
[data-testid="stSidebarContent"] {position: absolute !important; left: 0; top: 0; height: 100%; width: 76px !important; overflow: hidden !important;
    background: var(--navy); z-index: 1000001; transition: width .18s ease, box-shadow .18s ease; padding-left: 8px !important; padding-right: 8px !important;}
[data-testid="stSidebar"]:hover [data-testid="stSidebarContent"] {width: 250px !important; box-shadow: 6px 0 24px rgba(8, 24, 48, .35);}
[data-testid="stSidebarUserContent"] {padding-left: 0 !important; padding-right: 0 !important;}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] button {white-space: nowrap;}
[data-testid="stSidebar"] .st-key-side_toggle {justify-content: flex-end;}
%(N)s .st-key-side_toggle {justify-content: center;}
%(N)s [class*="st-key-uc_"], %(N)s [data-testid="stCheckbox"], %(N)s [data-testid="stCheckbox"] label {width: 100% !important;}
%(N)s [data-testid="stCheckbox"] label > div:first-of-type {display: none !important;}
%(N)s [data-testid="stCheckbox"] label {justify-content: center; padding: 8px 0;}
%(N)s [data-testid="stCheckbox"] [data-testid="stWidgetLabel"] {flex: none; width: auto;}
%(N)s [data-testid="stCheckbox"] label p, %(N)s .stButton button p {font-size: 0 !important; gap: 0 !important; justify-content: center;}
%(N)s .stButton button {justify-content: center !important; padding: 10px 0 !important;}
%(N)s .stButton button > div {justify-content: center !important;}
%(N)s [data-testid="stCheckbox"] label[data-selected="true"] {background: rgba(15,157,143,.30); box-shadow: inset 3px 0 0 var(--teal);}
%(N)s .side-label, %(N)s [data-testid="stCaptionContainer"], %(N)s .footer {display: none !important;}
""".replace("%(N)s", _N)
    st.markdown("<style>" + _rail + "</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------- top bar + KPIs
def _logo_html():
    if LOGO and (ROOT / LOGO).exists():
        import base64
        try:
            b = base64.b64encode((ROOT / LOGO).read_bytes()).decode()
            ext = (ROOT / LOGO).suffix.lstrip(".").lower() or "png"
            return f'<img src="data:image/{"svg+xml" if ext == "svg" else ext};base64,{b}">'
        except OSError:
            pass
    m = re.fullmatch(r"([A-Z][a-z0-9]+)([A-Z][A-Za-z0-9]+)", TITLE)      # SceneWyse -> Scene + Wyse in two colours
    if m:
        return f'{esc(m.group(1))}<span class="w2">{esc(m.group(2))}</span>'
    return esc(TITLE)


@fragment(3)
def topbar():
    watchdog()
    c = ctl()
    stats = read_stats()
    now = time.time()
    live_n = sum(1 for s in stats.values() if now - s["timestamp"] < FRESH_SECONDS)
    if running():
        sysc = chip("System running", "ok")
    elif c["wanted"]:
        sysc = chip("System restarting", "warn")
    elif live_n:
        sysc = chip("Running elsewhere", "warn")
    else:
        sysc = chip("System stopped", "idle")
    who = (DASH_USER or "?")[:2].upper() if DASH_PASS else "--"
    out = ('<a class="out" href="?signout=1" target="_self">Sign out</a>' if DASH_PASS else "")
    st.markdown(
        f'<div class="swbar"><span class="logo">{_logo_html()}</span><span class="sep"></span>'
        f'<span class="prod">{esc(PRODUCT)}</span><span class="sp"></span>{sysc}<span class="sep"></span>'
        f'<span class="date">{datetime.now():%d %b %Y &middot; %H:%M}</span><span class="sep"></span>'
        f'<span class="av">{esc(who)}</span>{out}</div>', unsafe_allow_html=True)


@fragment(3)
def kpis():
    stats = read_stats()
    now = time.time()
    live_n = sum(1 for s in stats.values() if now - s["timestamp"] < FRESH_SECONDS)
    df = read_events()
    td = df[df["timestamp"].dt.date == datetime.now().date()] if len(df) else df
    last = df["timestamp"].max() if len(df) else None
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kp("layers", sum(1 for n in LABELS if avail[n][0]), "Use cases available", f"{live_n} reporting now"),
                unsafe_allow_html=True)
    k2.markdown(kp("alert", len(td), "Alerts today", "since midnight", "#D92D20" if len(td) else "#167A3B",
                   "#FDE8E8" if len(td) else "#E6F6EC"), unsafe_allow_html=True)
    k3.markdown(kp("bell", len(df), "Total alerts", "all time", "#1F5FBF", "#E8F0FD"), unsafe_allow_html=True)
    if last is not None:
        days = (datetime.now().date() - last.date()).days
        big = ago(last) if days < 1 else ("yesterday" if days == 1 else f"{days} days ago")
        sub = last.strftime("%d %b %Y, %H:%M")
    else:
        big, sub = "None yet", ""
    k4.markdown(kp("clock", big, "Last alert", sub, "#0F9D8F", "#E3F5F2"), unsafe_allow_html=True)


# ---------------------------------------------------------------- System logs (page and tab share this)
def cam_full():
    """'Main entrance (CAM 01)' or just 'CAM 01'"""
    c = next((x for x in load_cams() if cam_disp(x["id"]) == CAM_LABEL), {})
    return f"{c['name']} ({CAM_LABEL})" if c.get("name") and c["name"].strip().lower() != CAM_LABEL.lower() else CAM_LABEL


def _edit_active():
    c = next((x for x in load_cams() if cam_disp(x["id"]) == CAM_LABEL), None)
    st.session_state.update(cam_edit=True, cam_target=c["id"] if c else None, cam_msg=(None, ""))
    if st.session_state.get("page") == "logs":
        st.session_state["page"] = "dash"
    else:
        st.session_state["_goto_live"] = True


def _toggle_pause():
    st.session_state["logs_paused"] = not st.session_state.get("logs_paused")


_LV = {"INFO": ("info", "#E3F5F2", "#0B7A6B"), "WARN": ("alert", "#FFF1E0", "#B54708"), "ERROR": ("err", "#FDE8E8", "#B42318")}


def logs_view():
    paused = bool(st.session_state.get("logs_paused")) or not st.session_state.get("log_auto", True)
    every = None if paused else 3
    txt, dot = cam_status(read_stats(), time.time())

    @fragment(every)
    def cards():
        stats = read_stats()
        now = time.time()
        t, d = cam_status(stats, now)
        nm = active_names(stats, now)
        rd = [n for n in nm if n in stats and now - stats[n]["timestamp"] < FRESH_SECONDS]
        a, b, c = st.columns(3)
        col = {"ok": ("#167A3B", "#E6F6EC"), "wait": ("#B54708", "#FFF1E0"), "bad": ("#B42318", "#FDE8E8")}.get(
            d, ("#5B6778", "#EEF1F6"))
        ac = next((x for x in load_cams() if cam_disp(x["id"]) == CAM_LABEL), {})
        title = ac.get("name") or CAM_LABEL
        a.markdown(f'<div class="scard" style="--c:#1F5FBF;--bg:#E8F0FD"><div class="ic">{svg("cam")}</div><div style="flex:1">'
                   f'<div class="k">Camera</div><div class="v">{esc(title)}</div>'
                   f'<div class="d">{esc(shown or "not set")}</div></div><span class="camchip">{esc(CAM_LABEL)}</span></div>',
                   unsafe_allow_html=True)
        b.markdown(f'<div class="scard" style="--c:{col[0]};--bg:{col[1]}"><div class="ic">{svg("wifi")}</div><div>'
                   f'<div class="k">Connection</div><div class="v">{esc(t)}</div>'
                   f'<div class="d">{"Pictures are arriving" if d == "ok" else "No frames received" if d != "off" else "Not running"}'
                   f'</div></div></div>', unsafe_allow_html=True)
        c.markdown(f'<div class="scard" style="--c:#167A3B;--bg:#E6F6EC"><div class="ic">{svg("layers")}</div><div>'
                   f'<div class="k">Analytics services</div><div class="v">{len(rd)} ready</div>'
                   f'<div class="d">{esc(", ".join(short(n) for n in rd) or "none yet")}</div></div></div>',
                   unsafe_allow_html=True)

    cards()
    st.write("")
    raw = log_tail(1500)
    allrows = parse_log(raw)
    dates = sorted({r["When"].date() for r in allrows if r["When"]}, reverse=True)
    cam_opts = ["All cameras"] + sorted({*(r["Camera"] for r in allrows if r["Camera"]), *(cam_disp(c["id"]) for c in load_cams())})
    with st.container(border=True):
        f1, f2, f3, f4, f5, f6, f7 = st.columns([1.2, 0.95, 1.0, 1.25, 1.25, 1.4, 1.45])
        f1.selectbox("Camera", cam_opts, key="log_cam")
        f2.selectbox("Severity", ["All levels", "Info", "Warning", "Error"], key="log_sev")
        f3.selectbox("Date", ["All dates"] + [f"{x:%d %b %Y}" for x in dates], key="log_date")
        f4.selectbox("Time range", ["Last hour", "Last 6 hours", "Last 24 hours", "All time"], index=2, key="log_range")
        f5.text_input("Search", placeholder="Search logs...", key="log_q")
        f6.write("")
        f6.button("Resume live logs" if st.session_state.get("logs_paused") else "Pause live logs", key="btn_pause",
                  on_click=_toggle_pause, **({"icon": ":material/play_arrow:" if st.session_state.get("logs_paused") else ":material/pause:"} if HAS_ICONS else {}), **WIDE)
        f7.write("")
        f7.download_button("Download logs", "\n".join(raw).encode("utf-8"), file_name="system_log.txt",
                           mime="text/plain", type="primary", **({"icon": ":material/download:"} if HAS_ICONS else {}), **WIDE)

    @fragment(every)
    def table():
        rows = parse_log(log_tail(1500))
        sev = st.session_state.get("log_sev", "All levels")
        q = (st.session_state.get("log_q") or "").lower().strip()
        want = {"Info": "INFO", "Warning": "WARN", "Error": "ERROR"}.get(sev)
        cam = st.session_state.get("log_cam", "All cameras")
        dpick = st.session_state.get("log_date", "All dates")
        rng = {"Last hour": 1, "Last 6 hours": 6, "Last 24 hours": 24}.get(st.session_state.get("log_range", "Last 24 hours"))
        cutoff = datetime.now() - timedelta(hours=rng) if rng else None
        keep = []
        for r in rows:
            if want and r["Level"] != want:
                continue
            if q and q not in (r["Message"] + " " + r["Source"]).lower():
                continue
            if cam != "All cameras" and r["Camera"] and r["Camera"] != cam:
                continue
            if dpick != "All dates" and (not r["When"] or f"{r['When']:%d %b %Y}" != dpick):
                continue
            if cutoff and r["When"] and r["When"] < cutoff:
                continue
            keep.append(r)
        head_date = next((r["When"] for r in keep if r["When"]), None)
        with st.container(border=True):
            if keep:
                body = ""
                for r in keep[:300]:
                    ic_, bg, fg = _LV[r["Level"]]
                    body += (f'<tr><td class="tm">{r["When"]:%H:%M:%S}</td>' if r["When"] else '<tr><td class="tm">&mdash;</td>')
                    body += (f'<td><span class="lvb" style="background:{bg};color:{fg}">{svg(ic_, 15)}{r["Level"]}</span></td>'
                             f'<td>{esc(r["Source"])}</td><td>{esc(r["Message"])}</td></tr>')
                st.markdown(f'<div class="logwrap"><table class="logt"><thead><tr><th>Timestamp{f" ({head_date:%d %b %Y})" if head_date else ""} &darr;</th>'
                            f'<th>Level</th><th>Source</th><th>Message</th></tr></thead><tbody>{body}</tbody></table></div>',
                            unsafe_allow_html=True)
            else:
                st.caption("No log lines match.")
            fa, fb = st.columns([3, 1.4])
            fa.caption(f"{len(keep)} events shown" + ("  -  live logs paused" if st.session_state.get("logs_paused") else ""))
            auto = fb.checkbox("Auto-scroll to latest events", value=True, key="log_auto")
            if auto != st.session_state.get("_log_auto_seen", True):
                st.session_state["_log_auto_seen"] = auto
                st.rerun()

    table()
    with st.container(border=True):
        k1, k2, k3, k4 = st.columns([0.55, 4, 1.35, 1.35])
        k1.markdown(f'<div class="lbicon">{svg("alert", 26)}</div>', unsafe_allow_html=True)
        colr = {"ok": "#167A3B", "wait": "#B54708", "bad": "#B42318"}.get(dot, "#5B6778")
        hint = ("No frames received from the camera. Check the camera address and credentials, then test the connection."
                if dot in ("wait", "bad") else "Pictures are arriving from the camera." if dot == "ok"
                else "The analytics are not running. Press Start on the Live view.")
        k2.markdown(f'<div style="font-weight:700;color:#0F1B2D">Selected camera: {esc(cam_full())}</div>'
                    f'<div style="font-weight:700;color:{colr}">Status: {esc(txt)}</div>'
                    f'<div style="color:#5B6778;font-size:.84rem">{esc(hint)}</div>', unsafe_allow_html=True)
        if k3.button("Test connection", type="primary", disabled=not url, key="btn_logtest",
                     **({"icon": ":material/play_arrow:"} if HAS_ICONS else {}), **WIDE):
            with st.spinner("Connecting ..."):
                st.session_state["log_test"] = test_rtsp(url)
        if k4.button("Edit camera", key="btn_logedit", **({"icon": ":material/edit:"} if HAS_ICONS else {}), **WIDE):
            _edit_active()
            st.rerun()
        res = st.session_state.get("log_test")
        if res:
            (st.success if res[0] else st.error)(res[1])


if st.session_state.get("page") == "logs":
    topbar()
    h1, h2 = st.columns([1.3, 4])
    h1.markdown('<div class="card-title" style="font-size:1.6rem;white-space:nowrap">System logs</div>', unsafe_allow_html=True)
    h2.button("← Back to Live view", on_click=_goto, args=("dash",))
    st.caption("Camera connectivity and analytics service events.")
    logs_view()
    st.stop()

topbar()
st.markdown('<div style="font-size:1.55rem;font-weight:700;color:#0F1B2D;letter-spacing:-.2px;margin:-3.3rem 0 .3rem">'
            'Welcome back</div>', unsafe_allow_html=True)
kpis()
st.write("")
tab_live, tab_alerts, tab_faces, tab_system = st.tabs(["Live view", "Alert history", "Faces", "System"])


# ---------------------------------------------------------------- Live
def _apply_and_go(names):
    start(names)


with tab_live:
    @fragment(2)
    def banner():
        stats = read_stats()
        now = time.time()
        c = ctl()
        if running() and not any(now - s["timestamp"] < FRESH_SECONDS for s in stats.values()) and now - c["started"] > 15:
            b1, b2 = st.columns([8, 1])
            b1.markdown(f'<div class="warnbar">{svg("alert", 20)} {esc(CAM_LABEL)} is waiting for frames. '
                        'Check the camera connection or open System logs.</div>', unsafe_allow_html=True)
            if b2.button("View logs", key="banner_logs"):
                st.session_state["page"] = "logs"
                st.rerun()

    banner()
    c_main, c_feed, c_cam = st.columns([3.0, 1.3, 1.25])

    # ---- camera view
    with c_main:
        @fragment(2)
        def camera_view():
            stats = read_stats()
            now = time.time()
            names = active_names(stats, now)
            skip = skipped()
            fresh = [n for n in names if n in stats and now - stats[n]["timestamp"] < FRESH_SECONDS]
            with st.container(border=True):
                h1, h2 = st.columns([1.25, 1.35])
                h1.markdown(f'<div class="card-title">Camera view<small>{esc(CAM_LABEL)}'
                            f'{" &middot; " + str(len(names)) + " analytics" if names else ""}</small></div>',
                            unsafe_allow_html=True)
                mode = h2.radio("Layout", ["Single view", "Separate tiles"], horizontal=True, key="layout",
                                label_visibility="collapsed",
                                help="Single view: one picture with every selected use case drawn on it. "
                                     "Separate tiles: one picture per use case.")
                if not names:
                    st.markdown(placeholder("Nothing is running", "Tick the analytics in the sidebar, then press Start.",
                                            380, shown), unsafe_allow_html=True)
                    st.markdown(selcam(), unsafe_allow_html=True)
                    return
                if mode == "Single view" or len(names) == 1:
                    # live video stream when available; otherwise the still picture the runner writes several times a second
                    live_img = OUT / "frames" / "_combined.jpg"
                    cands = [live_img]
                    if len(names) == 1:
                        cands.append(OUT / "frames" / f"{names[0]}.jpg")
                    surl = stream_url()
                    use_stream, img_bytes = False, None
                    for cand in cands:
                        try:
                            if cand.exists() and now - cand.stat().st_mtime < FRESH_SECONDS:
                                if surl and cand == live_img:
                                    use_stream = True
                                else:
                                    img_bytes = cand.read_bytes()
                                break
                        except OSError:
                            continue          # file is being replaced right now; the next refresh gets it
                    ok_img = use_stream or img_bytes is not None
                    if ok_img:
                        st.markdown(camwrap_stream(surl, shown) if use_stream else camwrap(img_bytes, shown),
                                    unsafe_allow_html=True)
                        st.markdown(selcam(), unsafe_allow_html=True)
                    else:
                        why = "; ".join(f"{short(n)}: {skip[n]}" for n in names if n in skip)
                        st.markdown(placeholder("Waiting for frames" if not why else "Could not start",
                                                why or "Your stream will appear here once connected. "
                                                       "The first start loads the models and takes a moment.", 380, shown),
                                    unsafe_allow_html=True)
                        st.markdown(selcam(), unsafe_allow_html=True)
                    for n in names:
                        s = stats.get(n)
                        if n in skip:
                            chp, body = chip("Could not start", "bad"), esc(skip[n])
                        elif s is None:
                            chp, body = chip("Starting", "idle"), ""
                        else:
                            age = now - s["timestamp"]
                            chp = chip("Live", "ok") if age < FRESH_SECONDS else chip(f"Stale {int(age)} s", "warn")
                            counts = s.get("counts") or {}
                            body = "".join(f'<span class="count">{esc(pretty(k))}<b>{esc(v)}</b></span>'
                                           for k, v in counts.items()) or '<span class="quiet">Nothing detected</span>'
                            if s.get("events"):
                                body += '<div class="alert-banner">ALERT: ' + esc(", ".join(
                                    f"{pretty(k)} ({v})" for k, v in s["events"].items())) + '</div>'
                        st.markdown(f'<div class="tile-head"><span class="tile-title">{esc(short(n))}</span>{chp}</div>'
                                    f'<div class="counts">{body}</div>', unsafe_allow_html=True)
                else:
                    ncols = 2 if len(names) <= 4 else 3
                    cols = st.columns(ncols)
                    for i, n in enumerate(names):
                        with cols[i % ncols]:
                            s = stats.get(n)
                            if n in skip:
                                head = chip("Could not start", "bad")
                            elif s is None:
                                head = chip("Starting", "idle")
                            else:
                                age = now - s["timestamp"]
                                head = chip("Live", "ok") if age < FRESH_SECONDS else chip(f"Stale {int(age)} s", "warn")
                            st.markdown(f'<div class="tile-head"><span class="tile-title">{esc(short(n))}</span>{head}</div>',
                                        unsafe_allow_html=True)
                            img = OUT / "frames" / f"{n}.jpg"
                            if n in skip:
                                st.error(skip[n])
                            elif s is not None and img.exists():
                                try:
                                    st.image(img.read_bytes(), **WIDE)
                                except Exception:
                                    pass
                                counts = s.get("counts") or {}
                                st.markdown('<div class="counts">' + ("".join(
                                    f'<span class="count">{esc(pretty(k))}<b>{esc(v)}</b></span>'
                                    for k, v in counts.items()) or '<span class="quiet">Nothing detected right now</span>')
                                    + '</div>', unsafe_allow_html=True)
                                if s.get("events"):
                                    st.markdown('<div class="alert-banner">ALERT: ' + esc(", ".join(
                                        f"{pretty(k)} ({v})" for k, v in s["events"].items())) + '</div>',
                                        unsafe_allow_html=True)
                            else:
                                st.markdown(placeholder("Waiting for frames", "Loading the model ...", 200, shown),
                                            unsafe_allow_html=True)
        camera_view()

    # ---- recent alerts
    with c_feed:
        @fragment(3)
        def feed():
            with st.container(border=True):
                fh1, fh2 = st.columns([1.2, 1])
                _ac = next((c for c in load_cams() if cam_disp(c["id"]) == CAM_LABEL), {})
                fh1.markdown('<div class="card-title">Recent alerts</div>'
                             f'<div class="subcam">{esc(CAM_LABEL)}{" &middot; " + esc(_ac["name"]) if _ac.get("name") else ""}</div>',
                             unsafe_allow_html=True)
                if fh2.button("View all →", key="viewall", **WIDE):
                    import streamlit.components.v1 as _cmp          # Streamlit cannot switch tabs itself: click the tab
                    _cmp.html("<script>const d=window.parent.document;for(const b of d.querySelectorAll('[role=tab]'))"
                              "{if(b.innerText.trim()==='Alert history'){b.click();break;}}</script>", height=0)
                ev = read_events()
                if not len(ev):
                    st.markdown('<div class="quiet">No alerts yet.</div>', unsafe_allow_html=True)
                for r in ev.sort_values("timestamp", ascending=False).head(4).itertuples():
                    uri = snap_uri(r.snapshot, 480)       # a larger picture inside the thumbnail, shown on hover
                    pic = f'<span class="th"><img src="{uri}"></span>' if uri else '<div class="ph"></div>'
                    pill, kind, why = alert_reason(r.event, r.count)
                    st.markdown(f'<div class="feed-item">{pic}<div><div class="t">{esc(short(r.detector))}</div>'
                                f'<span class="pill {kind}">{esc(pill)}</span>'
                                f'<div class="why">{esc(why)}</div>'
                                f'<div class="w">{r.timestamp:%d %b %Y &middot; %H:%M}</div></div></div>',
                                unsafe_allow_html=True)
        feed()

    # ---- cameras + control
    with c_cam:
        with st.container(border=True):
            ch1, ch2 = st.columns([0.95, 1.15])
            ch1.markdown(f'<div class="card-title">Cameras<small>({len(load_cams())})</small></div>',
                         unsafe_allow_html=True)
            if not test_src:
                ch2.button(("Close" if st.session_state.get("cam_edit") else "+ Add camera"),
                           key="btn_addcam", type="primary", on_click=_toggle_cam_edit, **WIDE)

            @fragment(3)
            def cam_item():
                stats = read_stats()
                now = time.time()
                t, d = cam_status(stats, now)
                names = active_names(stats, now) or selected
                tags = "".join(f'<span class="tag">{esc(short(n))}</span>' for n in names)
                try:
                    evc = read_events()["camera"].map(cam_disp).value_counts().to_dict()
                except Exception:
                    evc = {}
                cams = load_cams()
                if len(cams) > 1:
                    q = st.text_input("Find camera", placeholder="Find camera ...", key="cam_find",
                                      label_visibility="collapsed").strip().lower()
                    if q:
                        cams = [c for c in cams if q in (c["id"] + " " + str(c.get("name", "")) + " " + str(c.get("host", ""))).lower()]
                if not cams:
                    st.markdown('<div class="cam"><div class="n"><span class="radio off"></span>No camera yet</div>'
                                '<div class="st">Press + Add camera to add the first one.</div></div>', unsafe_allow_html=True)
                for c in cams:
                    cid = cam_disp(c["id"])
                    ck = c["id"].replace(" ", "_")
                    ttl = esc(cid) + (f' <span class="cnm">{esc(c["name"])}</span>' if c.get("name") and c["name"].strip().lower() != cid.lower() else "")
                    n2 = evc.get(cid, 0)
                    cnt = f'<div class="st">{n2} alert{"" if n2 == 1 else "s"} logged</div>'
                    active = cid == CAM_LABEL
                    with kcontainer(("camact_" if active else "camsel_") + ck):    # inactive: the whole card is the click area
                        if active:
                            st.markdown(f'<div class="cam sel"><div class="n"><span class="radio"></span>{ttl}</div>'
                                        f'<div class="st"><span class="dot {d}"></span>{esc(t)}</div>'
                                        f'<div class="st">{addr_line(shown or "no address set")}</div>{cnt}{tags}</div>',
                                        unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="cam pick"><div class="n"><span class="radio off"></span>{ttl}</div>'
                                        f'<div class="st">{addr_line("Webcam " + str(c.get("port", 0)) if c.get("host") == "webcam" else ("Browser webcam " + str(c.get("host", ""))[8:] if str(c.get("host", "")).startswith("browser") else str(c.get("host", "")) + ":" + str(c.get("port", 554))))}</div>'
                                        f'{cnt}</div>', unsafe_allow_html=True)
                            if st.button("Select " + c["id"], key=f"use_{ck}", disabled=not c.get("url"),
                                         help=None if c.get("url") else "Edit this camera and save it to select it"):
                                _use_cam(c["id"])
                                st.rerun()
                        if st.button("Edit", key=f"camed_{ck}"):
                            _edit_cam(c["id"])
                            st.rerun()
                        st.button("Remove", key=f"camdel_{ck}", on_click=_ask_del, args=(c["id"],))
                dcid = st.session_state.get("cam_del")
                if dcid:
                    st.warning(f"Remove {dcid} from the camera list? Its alerts stay in Alert history.")
                    y1, y2 = st.columns(2)
                    if y1.button("Remove", key="cam_del_yes", type="primary", **WIDE):
                        delete_cam(dcid)
                        st.session_state["cam_del"] = None
                        st.session_state["cam_msg"] = ("success", f"{dcid} removed.")
                        st.rerun()
                    if y2.button("Cancel", key="cam_del_no", **WIDE):
                        st.session_state["cam_del"] = None
                        st.rerun()
            cam_item()

            if not test_src and st.session_state.get("cam_edit"):
                if True:
                    if not DASH_PASS:
                        st.caption("Set DASHBOARD_PASSWORD in .env to allow changing the camera from here.")
                    elif running():
                        st.caption("Stop the analytics first (Stop button below), then change the camera.")
                    else:
                        tid = st.session_state.get("cam_target")
                        tgt = next((x for x in load_cams() if x["id"] == tid), None) if tid else None
                        turl = (tgt or {}).get("url") or ""
                        h0, p0, u0, w0, path0 = rtsp_parts(turl if not (turl.isdigit() or is_browser_url(turl)) else "")
                        if tgt and not turl:
                            h0, p0 = tgt.get("host", ""), int(tgt.get("port", 554))
                        n0 = (tgt or {}).get("name", "")
                        st.markdown(f"**Edit {tid}**" if tgt else "**Add a camera**")
                        if tgt:
                            is_web = tgt.get("host") == "webcam"
                            is_br = str(tgt.get("host", "")).startswith("browser")
                        else:
                            kind_pick = st.radio("Camera type", ["IP camera (RTSP)", "Webcam (this PC)", "Browser webcam (other laptop)"],
                                                 horizontal=True, key="cam_kind", label_visibility="collapsed")
                            is_web = kind_pick.startswith("Webcam")
                            is_br = kind_pick.startswith("Browser")
                        fk = (tid or "new") + ("b" if is_br else ("w" if is_web else "r"))
                        with st.form("camera_form_" + fk):
                            f_name = st.text_input("Camera name (optional)", value=n0, placeholder="e.g. Main entrance", key="fn_" + fk)
                            if is_br:
                                slot = slot_of_url(turl) if tgt else next_browser_slot()
                                lnk = browser_link(slot)
                                st.caption(f"Webcam slot {slot}. On that laptop open this link in Chrome, press Start camera, "
                                           "click Allow and keep the tab open. Each laptop needs its own camera entry and link:")
                                st.code(lnk or "Start the dashboard with  streamlit run serve.py  to get the link.", language=None)
                                st.caption("Open this dashboard through your ngrok address (not localhost) so the link shows that address.")
                            elif is_web:
                                f_idx = st.number_input("Webcam number", 0, 9, int(tgt["port"]) if tgt else 0, step=1, key="fi_" + fk,
                                                        help="0 = the built-in or first USB webcam, 1 = the second one")
                            else:
                                f_host = st.text_input("Camera IP address", value=h0, placeholder="192.168.1.50", key="fh_" + fk)
                                c1, c2 = st.columns(2)
                                f_port = c1.number_input("Port", 1, 65535, int(p0), step=1, key="fp_" + fk)
                                f_path = c2.text_input("Stream", value=path0, key="fs_" + fk, help="Tapo: stream1 (HD) or stream2 (light)")
                                f_user = st.text_input("User name", value=u0, key="fu_" + fk)
                                f_pw = st.text_input("Password", type="password", key="fw_" + fk,
                                                     placeholder="leave empty to keep the saved one" if w0 else "")
                            b1, b2 = st.columns(2)
                            do_test = b1.form_submit_button("Test", **WIDE)
                            do_save = b2.form_submit_button("Save", type="primary", **WIDE)
                        if do_test or do_save:
                            new_url = None
                            if is_br:
                                new_url = "browser:" + slot
                            elif is_web:
                                new_url = str(int(f_idx))
                            elif not f_host.strip() or not f_path.strip():
                                st.session_state["cam_msg"] = ("error", "Enter the camera IP address and the stream name.")
                            else:
                                new_url = build_rtsp(f_host, f_port, f_user, f_pw or (w0 if f_user == u0 else ""), f_path)
                            if new_url is not None:
                                if do_test:
                                    with st.spinner("Connecting ..."):
                                        ok, msg = test_rtsp(new_url)
                                    st.session_state["cam_msg"] = ("success" if ok else "error", msg)
                                elif tgt:
                                    update_cam(tid, new_url, f_name.strip() or None)
                                    save_rtsp(new_url)
                                    st.session_state["cam_msg"] = ("success", f"{tid} updated and selected. Press Start to use it.")
                                    st.session_state["cam_edit"], st.session_state["cam_target"] = False, None
                                    st.rerun()
                                else:
                                    save_rtsp(new_url)
                                    entry, is_new = register_cam(new_url, f_name.strip() or None)
                                    st.session_state["cam_msg"] = ("success", (f"{entry['id']} added to the camera log and selected. "
                                                                               if is_new else f"{entry['id']} was already in the log - updated and selected. ") + "Press Start to use this camera.")
                                    st.session_state["cam_edit"] = False
                                    st.rerun()
            if not test_src:
                kind, msg = st.session_state.get("cam_msg", (None, ""))
                if kind:
                    getattr(st, kind)(msg)

        st.markdown(f'<div class="infonote">{svg("info", 18)}<span>Select a camera to update the feed and alerts.</span></div>',
                    unsafe_allow_html=True)
        can_start = bool(selected) and (bool(url) or bool(test_src))
        on = running() or ctl()["wanted"]
        b1, b2 = st.columns(2)
        if on:
            changed = sorted(selected) != sorted(ctl()["names"])
            if b1.button("Apply changes" if changed else "Restart", key="btn_restart", disabled=not can_start,
                         type="primary" if changed else "secondary", **WIDE):
                start(selected)
                st.rerun()
        elif b1.button("Start", key="btn_start", type="primary", disabled=not can_start, **WIDE):
            start(selected)
            st.rerun()
        if b2.button("Stop", key="btn_stop", disabled=not (running() or ctl()["wanted"]), **WIDE):
            stop()
            st.rerun()
        if not selected:
            st.caption("Tick at least one analytic in the sidebar.")
        elif not (url or test_src):
            st.caption("Add the camera address first (Edit camera).")

    # ---- detection workspace
    with st.container(border=True):
        st.markdown('<div class="card-title">Detection workspace</div>', unsafe_allow_html=True)
        pk = kcontainer("presets")
        with pk:
            pcols = st.columns(len(PRESET_ORDER))
            for col, pname in zip(pcols, PRESET_ORDER):
                want = [n for n in PRESETS[pname] if avail[n][0]]
                active = sorted(selected) == sorted(want) and (bool(want) or not selected)
                col.button(pname.title() if HAS_KEY else ic({"All": "apps", "Safety kit": "health_and_safety", "Office": "business_center",
                                                     "Dress code": "checkroom", "Face recognition": "face", "None": "block"}[pname], pname),
                           key=f"preset_{pname}", on_click=_apply_preset, args=(PRESETS[pname],),
                           type="primary" if active else "secondary", **WIDE)
        if len(selected) > 4:
            st.caption("Many analytics share one CPU, so each refreshes less often.")

        @fragment(2)
        def workspace():
            stats = read_stats()
            now = time.time()
            running_names = active_names(stats, now)
            skip = skipped()
            rows = []
            for n in LABELS:
                if not avail[n][0]:
                    txt, cls = "Unavailable", ""
                elif n in running_names:
                    s = stats.get(n)
                    if n in skip:
                        txt, cls = "Could not start", "bad"
                    elif s is None:
                        txt, cls = "Starting", "wait"
                    elif now - s["timestamp"] < FRESH_SECONDS:
                        txt, cls = "Running", "ok"
                    else:
                        txt, cls = "Stale", "wait"
                else:
                    txt, cls = ("Selected", "sel") if st.session_state.get(f"uc_{n}") else ("Off", "")
                sub = f"<small>{esc(UC_SUB[n])}</small>" if n in UC_SUB else (
                    f"<small>{esc(avail[n][1])}</small>" if not avail[n][0] else "")
                rows.append(f'<div class="wsrow"><span class="wi">{_wsicon(n)}</span><span class="nm">{esc(short(n))}{sub}</span>'
                            f'<span class="stt"><span class="dot {cls}"></span>{txt}</span></div>')
            st.markdown(f'<div class="wsgrid">{"".join(rows)}</div>', unsafe_allow_html=True)
            live_n = sum(1 for n in running_names if n in stats and now - stats[n]["timestamp"] < FRESH_SECONDS)
            if running_names:
                msg = f"{live_n} of {len(running_names)} services running" + ("" if live_n else " &middot; Waiting for camera.")
            else:
                msg = "Nothing running. Choose analytics and press Start."
            st.markdown(f'<div class="wsfoot"><span class="dot {"ok" if live_n else "wait" if running_names else ""}"></span>{msg}</div>',
                        unsafe_allow_html=True)
        workspace()


# ---------------------------------------------------------------- Alerts
def _mark_reviewed(eid):
    mark_reviewed(eid)


with tab_alerts:
    @fragment(3)
    def alerts():
        df = read_events()
        if not len(df):
            st.markdown('<div class="empty"><h3>No alerts yet</h3>'
                        'Alerts appear here as soon as a detector raises one.</div>', unsafe_allow_html=True)
            return
        today = datetime.now().date()
        st.markdown('<div class="card-title" style="font-size:1.5rem">Alert history</div>'
                    '<div style="color:#5B6778;margin-bottom:10px">Browse and review camera events.</div>',
                    unsafe_allow_html=True)

        with st.container(border=True):
            f1, f2, f3, f4, f5 = st.columns([1.2, 1.25, 1.75, 1.45, 0.95])
            df = df.assign(camera=df["camera"].map(cam_disp))
            cams = sorted({*df["camera"].unique(), *(c["id"] for c in load_cams())})
            cam_pick = f1.selectbox("Camera", ["All cameras"] + cams)
            pick = f2.multiselect("Analytic", sorted(df["detector"].unique()), format_func=short, placeholder="All analytics")
            lo, hi = df["timestamp"].min().date(), df["timestamp"].max().date()
            quick = st.session_state.get("alert_quick")        # a quick-range button stores the range
            start_d, end_d = (quick if quick else (lo, hi))
            rng = f3.date_input("Date range", value=(start_d, end_d), min_value=min(lo, start_d),
                                max_value=max(hi, end_d, today))
            search = f4.text_input("Search", placeholder="Search alerts").strip().lower()

            def _set_range(kind):
                t = datetime.now().date()
                if kind == "today":
                    st.session_state["alert_quick"] = (t, t)
                elif kind == "yesterday":
                    y = t - timedelta(days=1)
                    st.session_state["alert_quick"] = (y, y)
                elif kind == "week":
                    st.session_state["alert_quick"] = (t - timedelta(days=6), t)
                elif kind == "month":
                    st.session_state["alert_quick"] = (t.replace(day=1), t)
                else:
                    st.session_state["alert_quick"] = None

            view = df
            if cam_pick != "All cameras":
                view = view[view["camera"] == cam_pick]
            if pick:
                view = view[view["detector"].isin(pick)]
            if isinstance(rng, (tuple, list)) and len(rng) == 2:
                view = view[(view["timestamp"].dt.date >= rng[0]) & (view["timestamp"].dt.date <= rng[1])]
            if search:
                hay = (view["detector"].map(short) + " " + view["event"].map(pretty) + " " + view["camera"].astype(str)).str.lower()
                view = view[hay.str.contains(search, regex=False)]
            view = view.sort_values("timestamp", ascending=False).reset_index(drop=True)
            f5.write("")
            f5.download_button("Export CSV", view.to_csv(index=False).encode("utf-8"), file_name=f"alerts_{today}.csv",
                               mime="text/csv", **WIDE)

            q = st.columns([1, 1, 1.3, 1.3, 1.3, 3])
            for col, (txt, kind) in zip(q, [("Today", "today"), ("Yesterday", "yesterday"), ("Last 7 days", "week"),
                                            ("This month", "month"), ("All dates", "all")]):
                col.button(txt, key=f"q_{kind}", on_click=_set_range, args=(kind,), **WIDE)

        left, right = st.columns([2.15, 1])
        with left.container(border=True):
            rtxt = (f"{rng[0]:%d %b %Y}" if rng[0] == rng[1] else f"{rng[0]:%d %b} - {rng[1]:%d %b %Y}") \
                if isinstance(rng, (tuple, list)) and len(rng) == 2 else "all dates"
            st.markdown(f'<div class="card-title">{len(view)} alerts<small style="color:#5B6778">&middot; '
                        f'{esc(cam_pick if cam_pick == "All cameras" else cam_disp(cam_pick))} &middot; {esc(rtxt)}</small></div>',
                        unsafe_allow_html=True)
            table = view.head(200).assign(camera=lambda d: d["camera"].map(cam_disp)).rename(columns={"timestamp": "Timestamp", "camera": "Camera"})
            table["Analytic"] = table["detector"].map(short)
            table["Result"] = table["event"].map(pretty)
            table["Status"] = table["status"].map(lambda s: "Reviewed" if s == "reviewed" else "Unreviewed")
            snaps = table["snapshot"].tolist()
            # the thumbnail keeps a larger picture inside it: Streamlit's double-click viewer shows it at its own size
            table["Preview"] = ([snap_uri(x, 560) for x in snaps[:40]] + [snap_uri(x, 240) for x in snaps[40:100]]
                                + [""] * max(0, len(snaps) - 100))
            table = table[["Preview", "Timestamp", "Camera", "Analytic", "Result", "Status"]]
            cfgc = {"Timestamp": st.column_config.DatetimeColumn("Timestamp", format="DD MMM YYYY, HH:mm:ss"),
                    "Preview": st.column_config.ImageColumn("Preview", width="small")}
            try:
                sel = st.dataframe(table, column_config=cfgc, hide_index=True, on_select="rerun",
                                   selection_mode="single-row", height=min(440, 38 * len(table) + 42), **WIDE)
                rows = sel.selection.rows
            except TypeError:                     # very old Streamlit: no row selection
                st.dataframe(table, column_config=cfgc, hide_index=True, height=min(440, 38 * len(table) + 42), **WIDE)
                rows = []
            st.caption(f"Showing the latest {len(table)} of {len(view)}. Tick a row to see it on the right; "
                       "double-click a preview to enlarge it.")

            # a tick picks that alert; the pick is remembered by its row id, so new alerts at the top do not swap it
            if list(rows) != st.session_state.get("alert_rows_prev", []):
                st.session_state["alert_rows_prev"] = list(rows)
                st.session_state["alert_pick"] = int(view.iloc[rows[0]]["id"]) if rows and rows[0] < len(view) else None

        with right.container(border=True):
            st.markdown('<div class="card-title">Alert details</div>', unsafe_allow_html=True)
            pid = st.session_state.get("alert_pick")
            row = None
            if pid is not None and len(view) and (view["id"] == pid).any():
                row = view[view["id"] == pid].iloc[0]
            elif len(view):
                row = view.iloc[0]                       # nothing ticked: show the newest
            if row is None:
                st.caption("No alert to show.")
            else:
                st.markdown(f'<div style="font-weight:700;color:#0F1B2D;margin-top:6px">{esc(short(row["detector"]))}</div>'
                            f'<div style="color:#5B6778;margin-bottom:8px">{esc(pretty(row["event"]))}</div>',
                            unsafe_allow_html=True)
                snap = OUT / "snapshots" / str(row["snapshot"])
                if snap.exists():
                    st.image(str(snap), **WIDE)
                else:
                    st.caption("The snapshot file is missing.")
                reviewed = row["status"] == "reviewed"
                st.markdown(
                    f'<div class="kv"><span>Timestamp</span><span>{row["timestamp"]:%d %b %Y &middot; %H:%M:%S}</span></div>'
                    f'<div class="kv"><span>Camera</span><span>{esc(cam_disp(row["camera"]))}</span></div>'
                    f'<div class="kv"><span>Count</span><span>{esc(row["count"])}</span></div>'
                    f'<div class="kv"><span>Status</span><span>{chip("Reviewed", "ok") if reviewed else chip("Unreviewed", "idle")}'
                    f'</span></div>', unsafe_allow_html=True)
                st.button("Mark reviewed", key="mark_rev", type="primary", disabled=reviewed or not HAS_REVIEW,
                          on_click=_mark_reviewed, args=(int(row["id"]),), **WIDE)

        if len(view):
            with st.expander("Charts"):
                g1, g2 = st.columns(2)
                with g1:
                    st.markdown("**Alerts by use case**")
                    st.bar_chart(view.groupby("detector").size().rename(index=short), height=220)
                with g2:
                    span = (view["timestamp"].max() - view["timestamp"].min()).total_seconds()
                    by_hour = span < 2 * 86400
                    st.markdown("**Alerts per hour**" if by_hour else "**Alerts per day**")
                    key = view["timestamp"].dt.floor("h" if by_hour else "D")
                    st.bar_chart(view.groupby(key).size(), height=220)

        with st.expander("PDF report"):
            e2, e3 = st.columns([1, 2])
            if build_pdf is None:
                e2.caption("PDF export needs:  pip install reportlab")
            else:
                with_pics = e3.checkbox("Include snapshots in the PDF", value=True)
                if e2.button("Create PDF report", **WIDE):
                    try:
                        st.session_state["pdf"] = build_pdf(
                            view, TITLE,
                            (f"{rng[0]:%d %b %Y}" if rng[0] == rng[1] else f"{rng[0]:%d %b %Y} - {rng[1]:%d %b %Y}")
                            if isinstance(rng, (tuple, list)) and len(rng) == 2 else
                            (f"{view['timestamp'].min():%d %b %Y} - {view['timestamp'].max():%d %b %Y}"
                             if len(view) else "-"),
                            OUT / "snapshots", LABELS, with_pics)
                    except Exception as ex:
                        st.session_state["pdf"] = None
                        st.error(f"Could not create the PDF: {ex}")
                if st.session_state.get("pdf"):
                    e2.download_button("Download PDF", st.session_state["pdf"],
                                       file_name=f"alert_report_{today}.pdf", mime="application/pdf", **WIDE)

    alerts()


# ---------------------------------------------------------------- Faces
FACE_DIR = OUT / "face_enroll"


def face_status():
    try:
        return json.loads((FACE_DIR / "status.json").read_text())
    except (OSError, ValueError):
        return None


def face_capture_active():
    return (FACE_DIR / "request.json").exists()


def _face_start():
    name = st.session_state.get("face_name", "").strip()
    st.session_state["face_err"] = ""
    if not valid_name(name):
        st.session_state["face_err"] = ("Enter a name with letters, numbers, spaces or . _ ' - "
                                        "(at most 40 characters; 'Unknown' is not allowed).")
        return
    FACE_DIR.mkdir(parents=True, exist_ok=True)
    (FACE_DIR / "status.json").unlink(missing_ok=True)
    tmp = FACE_DIR / "request.json.tmp"
    tmp.write_text(json.dumps({"id": f"{time.time():.3f}", "name": name, "target": 20, "ts": time.time()}))
    os.replace(tmp, FACE_DIR / "request.json")
    st.session_state["face_seen"] = "collecting"


def _face_cancel():
    (FACE_DIR / "request.json").unlink(missing_ok=True)


with tab_faces:
    if FaceDB is None:
        st.error("The face database module (common/face_db.py) is missing.")
    elif not DASH_PASS:
        st.info("Face data is personal. Set DASHBOARD_PASSWORD in .env (then restart the dashboard) to add or remove people here.")
    else:
        db = FaceDB()
        face_on = running() and "face_recognition" in ctl()["names"]
        st.markdown("**Add a person.** The person stands in front of the camera and turns their head slowly. "
                    "About 20 face pictures are collected and saved as numbers (face embeddings). "
                    "After that the person is recognised by name on the Live view; faces that match nobody show as Unknown "
                    "and raise an alert.")
        if not face_on:
            why = skipped().get("face_recognition")
            st.warning("Face recognition is not running. Tick **Face recognition** in the sidebar and press **Start**, "
                       "then come back here. The first start downloads two small face models (about 40 MB)."
                       + (f" Last start problem: {why}" if why else ""))
        cl, cr = st.columns([1.7, 1])
        with cr:
            with st.container(border=True):
                st.text_input("Name", key="face_name", placeholder="e.g. Priya", disabled=not face_on or face_capture_active())
                b1, b2 = st.columns(2)
                b1.button("Start capture", on_click=_face_start, type="primary",
                          disabled=not face_on or face_capture_active(), **WIDE)
                b2.button("Cancel", on_click=_face_cancel, disabled=not face_capture_active(), **WIDE)
                if st.session_state.get("face_err"):
                    st.error(st.session_state["face_err"])
                st.caption("Tips: good light on the face, look straight at the camera first, then turn slightly left, "
                           "right, up and down. Capturing again under the same name adds more pictures to that person.")

        @fragment(1 if face_on else None)
        def face_live():
            stt = face_status()
            state = (stt or {}).get("state")
            img = OUT / "frames" / "face_recognition.jpg"
            if face_on and img.exists() and time.time() - img.stat().st_mtime < 30:
                st.image(img.read_bytes(), **WIDE)
            elif face_on:
                st.caption("Waiting for the first picture from the camera ...")
            if stt and face_on:
                if state == "collecting":
                    st.progress(min(1.0, stt["count"] / max(1, stt["target"])),
                                text=f"Adding {stt['name']}: {stt['count']} of {stt['target']} pictures")
                    st.info(stt.get("message") or "")
                elif state == "done":
                    st.success(f"{stt['name']}: {stt['message']} They are recognised from now on.")
                elif state == "failed":
                    st.error(stt["message"])
                elif state == "cancelled":
                    st.info("Capture cancelled.")
            # the capture just ended: redraw the whole tab so the list of people below updates
            if st.session_state.get("face_seen") == "collecting" and state in ("done", "failed", "cancelled"):
                st.session_state["face_seen"] = state
                st.rerun()

        with cl:
            face_live()

        st.markdown("#### People who are recognised")
        people = db.people()
        if not people:
            st.caption("Nobody added yet.")
        for r0 in range(0, len(people), 4):
            cols = st.columns(4)
            for col, p in zip(cols, people[r0:r0 + 4]):
                with col.container(border=True):
                    if p["thumb"]:
                        st.image(p["thumb"].read_bytes(), width=110)
                    st.markdown(f"**{esc(p['name'])}**")
                    st.caption(f"{p['samples']} pictures")
                    key = f"face_del_{p['name']}"
                    if st.session_state.get(key):
                        st.warning("Delete this person?")
                        d1, d2 = st.columns(2)
                        if d1.button("Yes", key=key + "_y", **WIDE):
                            db.delete(p["name"])
                            st.session_state.pop(key, None)
                            st.rerun()
                        if d2.button("No", key=key + "_n", **WIDE):
                            st.session_state.pop(key, None)
                            st.rerun()
                    elif st.button("Delete", key=key + "_b", **WIDE):
                        st.session_state[key] = True
                        st.rerun()


# ---------------------------------------------------------------- System
with tab_system:
    logs_view()


# ---------------------------------------------------------------- jump to the Live view tab when asked
if st.session_state.pop("_goto_live", False):          # Streamlit cannot switch tabs itself: click the tab
    import streamlit.components.v1 as _cmp
    _cmp.html("<script>const d=window.parent.document;for(const b of d.querySelectorAll('[role=tab]'))"
              "{if(b.innerText.trim()==='Live view'){b.click();break;}}</script>", height=0)
