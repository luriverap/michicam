import os
import re
import time
import socket
import struct
import asyncio
import urllib.request
import concurrent.futures
import cv2
import numpy as np

# Precomputed G.711 A-law to 16-bit signed linear PCM lookup table
ALAW_TABLE = []
for i in range(256):
    a = i ^ 0x55
    sign = a & 0x80
    exponent = (a & 0x70) >> 4
    mantissa = a & 0x0F
    if exponent == 0:
        sample = (mantissa << 4) + 8
    else:
        sample = ((mantissa << 4) + 0x108) << (exponent - 1)
    if sign == 0:
        sample = -sample
    ALAW_TABLE.append(struct.pack("<h", max(-32768, min(32767, sample))))


def alaw_to_pcm16(alaw_bytes: bytes) -> bytes:
    return b"".join(ALAW_TABLE[b] for b in alaw_bytes)


class V380Camera:
    def __init__(
        self,
        cam_id="v380",
        name="Cámara 2 • V380 Pro",
        dev_id=111085293,
        ip="192.168.80.114",
        username="admin",
        password="",
        on_frame=None,
    ):
        self.cam_id = cam_id
        self.name = name
        self.dev_id = int(dev_id)
        self.ip = ip or "192.168.80.114"
        self.username = username or "admin"
        self.password = password or ""
        self.on_frame = on_frame

        self.connected = False
        self.connecting = False
        self.last_error = None
        self.last_frame: bytes | None = None
        self.last_frame_ts = 0.0
        self.fps = 0.0
        self._frame_counter = 0
        self._fps_ts = time.time()

        self.status_info = {
            "ir": True,
            "lamp": False,
            "resolution": "hd",
            "flip": 0,
            "mode": f"RTSP + ONVIF ({self.ip})",
        }

        self.audio_subscribers: set[asyncio.Queue] = set()
        self.audio_running = False
        self._audio_task: asyncio.Task | None = None
        self._task: asyncio.Task | None = None
        self._stop_flag = False
        self._restart_stream = False

    def update_credentials(self, dev_id=None, ip=None, username=None, password=None):
        if dev_id:
            try:
                self.dev_id = int(str(dev_id).strip())
            except ValueError:
                pass
        if ip is not None and ip.strip():
            self.ip = ip.strip()
        if username is not None:
            self.username = username.strip() or "admin"
        if password is not None:
            self.password = password
        self._restart_stream = True

    def start(self, loop: asyncio.AbstractEventLoop):
        self._stop_flag = False
        if self._task is None or self._task.done():
            self._task = loop.create_task(self._run_loop())

    async def stop(self):
        self._stop_flag = True
        self._restart_stream = True
        self.connected = False

    def _discover_lan_ip(self) -> str:
        """Verify known IP on port 554/8899 or scan 192.168.80.0/24 for ONVIF/RTSP V380 camera."""
        if self.ip:
            for p in (554, 8899):
                try:
                    s = socket.create_connection((self.ip, p), timeout=0.6)
                    s.close()
                    return self.ip
                except Exception:
                    pass

        def _probe_onvif(h):
            ip_c = f"192.168.80.{h}"
            if ip_c == "192.168.80.164":
                return None
            s_c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s_c.settimeout(0.25)
            try:
                if s_c.connect_ex((ip_c, 8899)) == 0 or s_c.connect_ex((ip_c, 554)) == 0:
                    return ip_c
            except Exception:
                pass
            finally:
                s_c.close()
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as ex:
            for res in ex.map(_probe_onvif, range(1, 255)):
                if res:
                    return res

        # Remote WAN fallback if router forwards port 554/8899 to 192.168.80.114
        wan_ip = "190.158.28.202"
        for p in (554, 8899):
            try:
                s = socket.create_connection((wan_ip, p), timeout=1.0)
                s.close()
                return wan_ip
            except Exception:
                pass
        return self.ip

    def _stream_rtsp_sync(self, ip: str):
        ch = "ch00_0" if self.status_info.get("resolution") == "hd" else "ch00_1"
        auth = f"{self.username}:{self.password}@" if self.password else ""
        rtsp_url = f"rtsp://{auth}{ip}:554/live/{ch}"

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            raise RuntimeError(
                f"En espera de red local de casa ({ip}:554). "
                "V380 RTSP requiere que el servidor esté en el Wi-Fi de casa o abrir el puerto 554 en el router."
            )

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.connected = True
        self.connecting = False
        self.last_error = None
        self.status_info["mode"] = f"RTSP + ONVIF ({ip})"
        self._restart_stream = False
        print(f"[V380] Connected & streaming video from {rtsp_url} ({self.dev_id})")

        try:
            bad_reads = 0
            while not self._stop_flag and not self._restart_stream and cap.isOpened():
                ok, frame = cap.read()
                if not ok or frame is None:
                    bad_reads += 1
                    if bad_reads > 8:
                        break
                    time.sleep(0.05)
                    continue
                bad_reads = 0

                flip_mode = self.status_info.get("flip", 0)
                if flip_mode == 1:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif flip_mode == 2:
                    frame = cv2.flip(frame, 1)
                elif flip_mode == 3:
                    frame = cv2.flip(frame, 0)

                ok_jpg, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok_jpg:
                    now = time.time()
                    self.last_frame = buf.tobytes()
                    self.last_frame_ts = now
                    self._frame_counter += 1
                    if now - self._fps_ts >= 2.0:
                        self.fps = round(self._frame_counter / (now - self._fps_ts), 1)
                        self._frame_counter = 0
                        self._fps_ts = now
                    if self.on_frame:
                        try:
                            self.on_frame(self.cam_id, self.name, self.last_frame)
                        except Exception:
                            pass
        finally:
            cap.release()
            self.connected = False

    def ingest_pushed_frame(self, jpeg_bytes: bytes):
        now = time.time()
        self.last_frame = jpeg_bytes
        self.last_frame_ts = now
        self.connected = True
        self.connecting = False
        self.last_error = None
        self._frame_counter += 1
        if now - self._fps_ts >= 2.0:
            self.fps = round(self._frame_counter / (now - self._fps_ts), 1)
            self._frame_counter = 0
            self._fps_ts = now
        if self.on_frame:
            try:
                self.on_frame(self.cam_id, self.name, self.last_frame)
            except Exception:
                pass

    async def _run_loop(self):
        while not self._stop_flag:
            if self.last_frame_ts and (time.time() - self.last_frame_ts) < 8.0:
                self.connected = True
                self.connecting = False
                await asyncio.sleep(2.0)
                continue
            try:
                self.connecting = True
                lan_ip = await asyncio.to_thread(self._discover_lan_ip)
                if lan_ip:
                    self.ip = lan_ip
                await asyncio.to_thread(self._stream_rtsp_sync, self.ip)
            except Exception as e:
                if not (self.last_frame_ts and (time.time() - self.last_frame_ts) < 8.0):
                    self.last_error = str(e)
                    print(f"[V380] Stream notice ({self.ip}): {e}")
            finally:
                if not (self.last_frame_ts and (time.time() - self.last_frame_ts) < 8.0):
                    self.connected = False
                self.connecting = False

            if not self._stop_flag:
                await asyncio.sleep(2.5)

    def _onvif_ptz_move(self, vx: float, vy: float) -> bool:
        soap = f"""<?xml version="1.0" encoding="utf-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema">
  <s:Body>
    <tptz:ContinuousMove>
      <tptz:ProfileToken>PROFILE_000</tptz:ProfileToken>
      <tptz:Velocity>
        <tt:PanTilt x="{vx:.2f}" y="{vy:.2f}"/>
      </tptz:Velocity>
    </tptz:ContinuousMove>
  </s:Body>
</s:Envelope>"""
        try:
            req = urllib.request.Request(
                f"http://{self.ip}:8899/onvif/ptz_service",
                data=soap.encode("utf-8"),
                headers={"Content-Type": "application/soap+xml; charset=utf-8"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3.0) as r:
                return r.status == 200
        except Exception as e:
            print(f"[V380] ONVIF Move error: {e}")
            return False

    def _onvif_ptz_stop(self) -> bool:
        soap = """<?xml version="1.0" encoding="utf-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl">
  <s:Body>
    <tptz:Stop>
      <tptz:ProfileToken>PROFILE_000</tptz:ProfileToken>
      <tptz:PanTilt>true</tptz:PanTilt>
      <tptz:Zoom>true</tptz:Zoom>
    </tptz:Stop>
  </s:Body>
</s:Envelope>"""
        try:
            req = urllib.request.Request(
                f"http://{self.ip}:8899/onvif/ptz_service",
                data=soap.encode("utf-8"),
                headers={"Content-Type": "application/soap+xml; charset=utf-8"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3.0) as r:
                return r.status == 200
        except Exception as e:
            print(f"[V380] ONVIF Stop error: {e}")
            return False

    def _onvif_set_ir(self, enabled: bool) -> bool:
        mode = "OFF" if enabled else "ON"  # In ONVIF IrCutFilter: OFF = night mode (IR LEDs on), ON = day mode
        soap = f"""<?xml version="1.0" encoding="utf-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" xmlns:timg="http://www.onvif.org/ver20/imaging/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema">
  <s:Body>
    <timg:SetImagingSettings>
      <timg:VideoSourceToken>V_SRC_000</timg:VideoSourceToken>
      <timg:ImagingSettings>
        <tt:IrCutFilter>{mode}</tt:IrCutFilter>
      </timg:ImagingSettings>
      <timg:ForcePersistence>true</timg:ForcePersistence>
    </timg:SetImagingSettings>
  </s:Body>
</s:Envelope>"""
        try:
            req = urllib.request.Request(
                f"http://{self.ip}:8899/onvif/image_service",
                data=soap.encode("utf-8"),
                headers={"Content-Type": "application/soap+xml; charset=utf-8"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3.0) as r:
                return r.status == 200
        except Exception:
            return True

    async def ptz(self, action: str):
        if not self.connected:
            return {"ok": False, "error": "Cámara V380 Pro no conectada"}
        clean = action.replace("step_", "")
        if clean == "stop":
            ok = await asyncio.to_thread(self._onvif_ptz_stop)
            return {"ok": ok}

        vel_map = {
            "up": (0.0, 0.6),
            "down": (0.0, -0.6),
            "left": (-0.6, 0.0),
            "right": (0.6, 0.0),
        }
        if clean not in vel_map:
            return {"ok": False, "error": f"Dirección {action} no soportada"}

        vx, vy = vel_map[clean]
        ok = await asyncio.to_thread(self._onvif_ptz_move, vx, vy)
        asyncio.create_task(self._delayed_stop(0.45))
        return {"ok": ok}

    async def _delayed_stop(self, delay: float):
        await asyncio.sleep(delay)
        await asyncio.to_thread(self._onvif_ptz_stop)

    async def control(self, setting: str, value=None):
        if setting == "reconnect":
            self._restart_stream = True
            return {"ok": True, "message": "Reconectando cámara V380 Pro..."}

        if setting == "ir":
            new_val = not self.status_info["ir"] if value is None else bool(value)
            await asyncio.to_thread(self._onvif_set_ir, new_val)
            self.status_info["ir"] = new_val
        elif setting == "lamp":
            new_val = not self.status_info["lamp"] if value is None else bool(value)
            self.status_info["lamp"] = new_val
        elif setting == "flip":
            self.status_info["flip"] = (self.status_info["flip"] + 1) % 4
        elif setting == "resolution":
            res_str = str(value or ("sd" if self.status_info["resolution"] == "hd" else "hd")).lower()
            self.status_info["resolution"] = res_str
            self._restart_stream = True
        return {"ok": True, "status": self.status_info}

    async def ensure_audio(self):
        if not self.audio_running:
            self.audio_running = True
            self._audio_task = asyncio.create_task(asyncio.to_thread(self._rtsp_audio_loop))

    def _rtsp_audio_loop(self):
        """Stream PCMA (G.711 A-law 8kHz) from RTSP track2 and convert to PCM16."""
        try:
            s = socket.create_connection((self.ip, 554), timeout=4.0)
            s.settimeout(4.0)
            setup_req = (
                f"SETUP rtsp://{self.ip}:554/live/ch00_1/track2 RTSP/1.0\r\n"
                f"CSeq: 1\r\n"
                f"Transport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n\r\n"
            )
            s.sendall(setup_req.encode("latin-1"))
            r1 = s.recv(4096).decode("latin-1", errors="ignore")
            m = re.search(r"Session:\s*([^\r\n;]+)", r1)
            sess = m.group(1).strip() if m else ""
            play_req = (
                f"PLAY rtsp://{self.ip}:554/live/ch00_1 RTSP/1.0\r\n"
                f"CSeq: 2\r\n"
                f"Session: {sess}\r\n"
                f"Range: npt=0.000-\r\n\r\n"
            )
            s.sendall(play_req.encode("latin-1"))
            buf = bytearray()
            while self.audio_running and self.audio_subscribers and not self._stop_flag:
                chunk = s.recv(2048)
                if not chunk:
                    break
                buf.extend(chunk)
                while len(buf) >= 4:
                    idx = buf.find(b"$")
                    if idx < 0:
                        buf.clear()
                        break
                    if idx > 0:
                        del buf[:idx]
                    if len(buf) < 4:
                        break
                    ch = buf[1]
                    pkt_len = struct.unpack_from(">H", buf, 2)[0]
                    if len(buf) < 4 + pkt_len:
                        break
                    rtp = bytes(buf[4 : 4 + pkt_len])
                    del buf[: 4 + pkt_len]
                    if ch == 0 and len(rtp) > 12:
                        pt = rtp[1] & 0x7F
                        if pt == 8:  # PCMA 8kHz
                            pcm = alaw_to_pcm16(rtp[12:])
                            dead = []
                            for q in list(self.audio_subscribers):
                                try:
                                    q.put_nowait(pcm)
                                except asyncio.QueueFull:
                                    dead.append(q)
                            for q in dead:
                                self.audio_subscribers.discard(q)
            s.close()
        except Exception as e:
            print(f"[V380] Audio stream ended: {e}")
        finally:
            self.audio_running = False

    def get_summary(self):
        is_conn = self.connected or bool(self.last_frame_ts and (time.time() - self.last_frame_ts) < 15.0)
        return {
            "id": self.cam_id,
            "name": self.name,
            "protocol": "V380 Pro (RTSP + ONVIF Direct)",
            "dev_uid": f"{self.dev_id} (v380_81766371136)",
            "ip": self.ip,
            "connected": is_conn,
            "connecting": False if is_conn else self.connecting,
            "fps": self.fps if is_conn else 0.0,
            "last_frame_age": round(time.time() - self.last_frame_ts, 1) if self.last_frame_ts else None,
            "last_error": self.last_error,
            "status": self.status_info,
            "capabilities": ["ptz", "ir", "lamp", "flip", "resolution", "audio"],
        }
