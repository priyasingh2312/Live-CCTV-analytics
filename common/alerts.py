"""Telegram and email alerts. Credentials are read from environment / .env.

Telegram: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
Email   : SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_TO
Alerts are sent in a background thread so video processing is never blocked.
E-mails are combined: alerts that arrive within alerts.email.batch_seconds (default 30) go out as ONE e-mail.
"""
import html
import mimetypes
import os
import re
import smtplib
import threading
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path


class AlertManager:
    def __init__(self, alert_cfg):
        alert_cfg = alert_cfg or {}
        self.telegram_on = bool((alert_cfg.get("telegram") or {}).get("enabled"))
        email_cfg = alert_cfg.get("email") or {}
        self.email_on = bool(email_cfg.get("enabled"))
        try:
            self.window = max(0.0, float(email_cfg.get("batch_seconds", 30)))
        except (TypeError, ValueError):
            self.window = 30.0
        self._buf, self._timer, self._lock = [], None, threading.Lock()

    def send(self, text, image_path=None):
        if not (self.telegram_on or self.email_on):
            return
        threading.Thread(target=self._send, args=(text, image_path), daemon=True).start()

    def _send(self, text, image_path):
        if self.telegram_on:
            try:
                self._telegram(text, image_path)
            except Exception as e:  # never crash the video loop
                print(f"[alert] telegram failed: {e}")
        if self.email_on:
            self._queue_email(text, image_path)

    # ---- one e-mail for all alerts of a short time window
    def _queue_email(self, text, image_path):
        if self.window <= 0:
            self._send_email([(text, image_path)])
            return
        with self._lock:
            self._buf.append((text, image_path))
            if self._timer is None:
                self._timer = threading.Timer(self.window, self._flush)
                self._timer.daemon = True
                self._timer.start()

    def _flush(self):
        with self._lock:
            items, self._buf, self._timer = self._buf, [], None
        if items:
            self._send_email(items)

    def _send_email(self, items):
        try:
            self._email(items)
        except Exception as e:
            print(f"[alert] email failed: {e}")

    @staticmethod
    def _telegram(text, image_path):
        import requests

        token = os.environ["TELEGRAM_BOT_TOKEN"]
        chat_id = os.environ["TELEGRAM_CHAT_ID"]
        base = f"https://api.telegram.org/bot{token}"
        if image_path:
            with open(image_path, "rb") as f:
                r = requests.post(f"{base}/sendPhoto", data={"chat_id": chat_id, "caption": text},
                                  files={"photo": f}, timeout=20)
        else:
            r = requests.post(f"{base}/sendMessage", data={"chat_id": chat_id, "text": text}, timeout=20)
        r.raise_for_status()

    @staticmethod
    def _email(items):
        infos = [describe_alert(t) for t, _ in items[:MAX_IN_EMAIL]]
        more = max(0, len(items) - MAX_IN_EMAIL)
        brand, cam = infos[0]["brand"], infos[0]["camera"]
        msg = EmailMessage()
        if len(items) == 1:
            msg["Subject"] = f"{brand} alert: {infos[0]['headline']} ({cam})"
        else:
            msg["Subject"] = f"{brand}: {len(items)} alerts on {cam}"
        msg["From"] = f"{brand} Alerts <{os.environ['SMTP_USER']}>"
        msg["To"] = os.environ["ALERT_EMAIL_TO"]
        plain = infos[0]["plain"] if len(items) == 1 else (
            f"{brand} - {len(items)} alerts on {cam}\n\n" + "\n".join(
                f"- {i['when']}  {i['headline']}  ({i['det']})" for i in infos)
            + (f"\n... and {more} more\n" if more else "\n")
            + "\nSnapshots are attached. Open the dashboard (Alert history) to review the alerts.\n")
        msg.set_content(plain)
        images = []
        for _, path in items[:MAX_IN_EMAIL]:
            data = None
            if path:
                try:
                    ctype = mimetypes.guess_type(str(path))[0] or "image/jpeg"
                    with open(path, "rb") as f:
                        data = (f.read(), ctype.split("/", 1))
                except OSError:
                    data = None
            images.append(data)
        msg.add_alternative(_html_many(brand, cam, infos, images, more), subtype="html")
        html_part = msg.get_payload()[1]
        for k, data in enumerate(images):
            if data:
                html_part.add_related(data[0], maintype=data[1][0], subtype=data[1][1], cid=f"<snap{k}>")
        host = os.environ["SMTP_HOST"]
        port = int(os.environ.get("SMTP_PORT", "587"))
        with smtplib.SMTP(host, port, timeout=20) as s:
            s.starttls()
            s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
            s.send_message(msg)


MAX_IN_EMAIL = 10


# ---------------------------------------------------------------- e-mail wording
_LINE = re.compile(r"^\[(?P<det>[^\]]+)\]\s+(?P<ev>.+?)\s+\(count=(?P<n>\d+)\)\s+at\s+(?P<ts>.+)$")


def _brand():
    try:
        import yaml
        cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text(encoding="utf-8")) or {}
        return (cfg.get("dashboard") or {}).get("title") or "Video analytics"
    except Exception:
        return "Video analytics"


def _cam():
    raw = os.environ.get("CAMERA_NAME", "").strip()
    m = re.fullmatch(r"(?i)cam(?:era)?[-_ ]*(\d+)", raw)
    return f"CAM {int(m.group(1)):02d}" if m else (raw or "Camera")


def _reason(event, n):
    low = event.lower()
    if low == "unknown":
        return "Unknown person", "A face was seen that is not in the saved faces."
    if low.startswith("no_") or low.startswith("no "):
        item = low[3:].replace("_", " ").strip()
        return f"No {item} detected", f"No {item} was detected" + (f" on {n} people." if n > 1 else ".")
    if " above " in low:
        lab, lim = low.split(" above ", 1)
        return "Threshold exceeded", f"{lab.replace('_', ' ').capitalize()} count {n} is above the limit of {lim}."
    if " below " in low:
        lab, lim = low.split(" below ", 1)
        return "Below minimum", f"{lab.replace('_', ' ').capitalize()} count is below the minimum of {lim}."
    if low == "beard":
        return "Beard visible", "An unshaved face was seen."
    return event.replace("_", " ").capitalize(), event.replace("_", " ").capitalize() + (f" ({n})." if n > 1 else ".")


def describe_alert(text):
    brand, cam = _brand(), _cam()
    m = _LINE.match(text.strip())
    if not m:
        plain = f"{brand} alert\nCamera: {cam}\n\n{text}\n"
        return {"brand": brand, "camera": cam, "headline": text[:60], "plain": plain,
                "det": "-", "when": "-", "why": text}
    det = m["det"].replace("_detection", "").replace("_", " ").strip().capitalize()
    n = int(m["n"])
    headline, why = _reason(m["ev"], n)
    try:
        when = datetime.strptime(m["ts"].strip(), "%Y-%m-%d %H:%M:%S").strftime("%d %b %Y, %H:%M:%S")
    except ValueError:
        when = m["ts"].strip()
    plain = (f"{brand} - alert\n\n{headline}\n\nCamera:    {cam}\nAnalytic: {det}\nTime:      {when}\n"
             f"Details:   {why}\n\nA snapshot is attached. Open the dashboard (Alert history) to review this alert.\n")
    return {"brand": brand, "camera": cam, "headline": headline, "plain": plain,
            "det": det, "when": when, "why": why}


def _html_many(brand, cam, infos, images, more=0):
    def esc(x):
        return html.escape(str(x))
    m = re.fullmatch(r"([A-Z][a-z0-9]+)([A-Z][A-Za-z0-9]+)", brand)
    name = (f'{esc(m.group(1))}<span style="color:#7FB2FF">{esc(m.group(2))}</span>' if m else esc(brand))
    n = len(infos) + more
    if n == 1:
        title, sub = infos[0]["headline"], f"Detected on {cam}"
    else:
        title, sub = f"{n} alerts on {cam}", "Alerts that arrived close together"
    cards = ""
    for k, i in enumerate(infos):
        rows = "".join(
            f'<tr><td style="padding:6px 0;color:#5B6778;width:90px;font-size:13px">{kk}</td>'
            f'<td style="padding:6px 0;color:#0F1B2D;font-size:13px;font-weight:600">{esc(v)}</td></tr>'
            for kk, v in (("Analytic", i["det"]), ("Time", i["when"]), ("Details", i["why"])) if v and v != "-")
        pic = (f'<img src="cid:snap{k}" alt="Snapshot" style="display:block;width:100%;max-width:520px;border-radius:8px;'
               'border:1px solid #E3E8F0;margin-top:12px">') if images[k] else ""
        head = ("" if n == 1 else
                f'<div style="font-size:16px;font-weight:700;color:#0F1B2D;margin-bottom:4px">{esc(i["headline"])}</div>')
        box = ("" if n == 1 else "border:1px solid #E3E8F0;border-radius:10px;padding:14px 16px;")
        cards += (f'<div style="{box}margin:0 0 14px 0">{head}<table role="presentation" width="100%" cellpadding="0" '
                  f'cellspacing="0">{rows}</table>{pic}</div>')
    if more:
        cards += f'<div style="color:#5B6778;font-size:13px;margin-bottom:8px">... and {more} more. Open Alert history to see all of them.</div>'
    return f"""<!doctype html><html><body style="margin:0;background:#F3F7FB;font-family:Segoe UI,Arial,sans-serif">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#F3F7FB;padding:24px 12px"><tr><td align="center">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;background:#fff;border-radius:12px;overflow:hidden;border:1px solid #E3E8F0">
<tr><td style="background:#0B3A6E;padding:18px 28px;color:#fff;font-size:22px;font-weight:800;letter-spacing:-.2px">{name}</td></tr>
<tr><td style="padding:26px 28px 8px 28px">
<div style="display:inline-block;background:#FDE8E8;color:#B42318;font-size:12px;font-weight:700;padding:4px 12px;border-radius:999px;letter-spacing:.4px">{"ALERT" if n == 1 else "ALERTS"}</div>
<h1 style="margin:12px 0 4px 0;font-size:22px;color:#0F1B2D">{esc(title)}</h1>
<div style="color:#5B6778;font-size:14px">{esc(sub)}</div></td></tr>
<tr><td style="padding:14px 28px 4px 28px"><div style="border-top:1px solid #EDF0F5;padding-top:14px">{cards}</div></td></tr>
<tr><td style="padding:18px 28px 26px 28px;color:#7A8696;font-size:12px;line-height:1.5;border-top:1px solid #EDF0F5">
This is an automated message from {esc(brand)}. Open the dashboard and go to <b>Alert history</b> to review the alerts and mark them as reviewed.</td></tr>
</table></td></tr></table></body></html>"""
