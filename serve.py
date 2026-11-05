"""Start the dashboard with an extra page that lets ANOTHER laptop's webcam be used as a camera.

    streamlit run serve.py                      (use this instead of  streamlit run dashboard\\app.py)

On the other laptop open   https://<your-ngrok-address>/browser-camera?key=<key>
The key is created on first start and saved in the file  .browser_cam_key  (shown in the dashboard when you
choose "Browser webcam"). Without the key nobody can send pictures, even though the ngrok address is public.
Each webcam has its own slot (&cam=web1, &cam=web2 ...); its pictures are saved to outputs\\browser_cam\\<slot>.jpg and read by the analytics like any other camera.
"""
import os
import pathlib
import secrets

import streamlit as st
from starlette.responses import HTMLResponse, PlainTextResponse, Response
from starlette.routing import Route

ROOT = pathlib.Path(__file__).resolve().parent
KEY_FILE = ROOT / ".browser_cam_key"
FRAME_DIR = ROOT / "outputs" / "browser_cam"
MAX_BYTES = 3_000_000


def get_key():
    if not KEY_FILE.exists():
        KEY_FILE.write_text(secrets.token_urlsafe(12), encoding="utf-8")
    return KEY_FILE.read_text(encoding="utf-8").strip()


def slot_of(request):
    """Which webcam this is (web1, web2 ...): letters, digits, - and _ only, so it is safe as a file name."""
    s = "".join(ch for ch in request.query_params.get("cam", "default").lower() if ch.isalnum() or ch in "-_")[:32]
    return s or "default"


def _ok(request):
    return secrets.compare_digest(request.query_params.get("key", ""), get_key())


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Browser camera</title>
<style>
body{font-family:system-ui,Segoe UI,Arial;margin:0;background:#0b1b2e;color:#e8eef6;display:flex;flex-direction:column;align-items:center;padding:16px}
video{width:100%;max-width:720px;border-radius:12px;background:#000}
button,select{font-size:16px;padding:10px 16px;border-radius:8px;border:0;margin:6px}
button{background:#1f7ae0;color:#fff;cursor:pointer}button.stop{background:#c0392b}
#st{margin:8px;min-height:24px}.s{opacity:.8;font-size:13px;max-width:720px;text-align:center}
</style></head><body>
<h2>Use this laptop's webcam as a camera</h2><div class="s" id="slot"></div>
<video id="v" autoplay playsinline muted></video>
<div><select id="dev"></select><button id="go">Start camera</button></div>
<div id="st">Press Start camera and click Allow when the browser asks.</div>
<div class="s">Keep this tab open and visible while the analytics run (a hidden tab slows down or pauses the camera).
Pictures are sent to the dashboard computer; nothing is stored here.</div>
<script>
const q=new URLSearchParams(location.search),key=q.get('key')||'',cam=q.get('cam')||'default';
const v=document.getElementById('v'),st=document.getElementById('st'),go=document.getElementById('go'),dev=document.getElementById('dev');
let stream=null,timer=null,busy=false,sent=0,fails=0,last=0;
const cv=document.createElement('canvas');
async function fillDevices(){try{const l=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput');
 dev.innerHTML=l.map((d,i)=>`<option value="${d.deviceId}">${d.label||('Camera '+(i+1))}</option>`).join('');}catch(e){}}
async function start(){
 if(!navigator.mediaDevices){st.textContent='This page needs https (use the https ngrok address).';return;}
 try{stream=await navigator.mediaDevices.getUserMedia({video:{deviceId:dev.value?{exact:dev.value}:undefined,width:{ideal:1280},height:{ideal:720}},audio:false});}
 catch(e){st.textContent='Could not open the camera: '+e.message;return;}
 v.srcObject=stream;await fillDevices();go.textContent='Stop camera';go.className='stop';
 try{if(navigator.wakeLock)navigator.wakeLock.request('screen');}catch(e){}
 timer=setInterval(send,120);
}
function stop(){clearInterval(timer);if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;v.srcObject=null;go.textContent='Start camera';go.className='';st.textContent='Stopped.';}
function send(){
 if(busy||!stream||!v.videoWidth)return;busy=true;
 const w=Math.min(1280,v.videoWidth),h=Math.round(w*v.videoHeight/v.videoWidth);cv.width=w;cv.height=h;
 cv.getContext('2d').drawImage(v,0,0,w,h);
 cv.toBlob(async b=>{try{const r=await fetch('/browser-camera/frame?key='+encodeURIComponent(key)+'&cam='+encodeURIComponent(cam),{method:'POST',body:b});
   if(r.ok){sent++;fails=0;const n=Date.now();st.textContent='Sending to the dashboard ... '+sent+' pictures sent'+(last?(' ('+(1000/(n-last)).toFixed(1)+' per second)'):'');last=n;}
   else{fails++;st.textContent=r.status===403?'Wrong or missing key in the address. Copy the full link from the dashboard.':'The dashboard refused a picture ('+r.status+').';}}
  catch(e){fails++;st.textContent='Cannot reach the dashboard. Check the ngrok address and that the dashboard is running.';}
  busy=false;},'image/jpeg',0.7);
}
document.getElementById('slot').textContent='Camera slot: '+cam;
go.onclick=()=>stream?stop():start();fillDevices();
</script></body></html>"""


async def page(request):
    if not _ok(request):
        return PlainTextResponse("Wrong or missing key. Copy the full link from the dashboard (Camera > Browser webcam).", status_code=403)
    return HTMLResponse(PAGE)


async def frame(request):
    if not _ok(request):
        return PlainTextResponse("forbidden", status_code=403)
    data = await request.body()
    if not data or len(data) > MAX_BYTES or data[:2] != b"\xff\xd8":      # must be a JPEG, not too large
        return PlainTextResponse("bad picture", status_code=400)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    out = FRAME_DIR / (slot_of(request) + ".jpg")
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, out)                                             # the reader never sees a half-written file
    return Response(status_code=204)


get_key()
app = st.App(
    "dashboard/app.py",
    routes=[Route("/browser-camera", page), Route("/browser-camera/frame", frame, methods=["POST"])],
)
