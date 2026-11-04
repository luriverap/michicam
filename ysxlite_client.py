import os
import sys
import time
import socket
import struct
import asyncio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AIOPPPP_PATH = os.path.join(BASE_DIR, "aiopppp_repo")
if AIOPPPP_PATH not in sys.path:
    sys.path.insert(0, AIOPPPP_PATH)

from aiopppp import Device
from aiopppp.types import DeviceDescriptor, DeviceID, Encryption
from aiopppp.session import make_session, SessionUDPProtocol


class YsxLiteCamera:
    def __init__(self, cam_id="ysxlite", name="Cámara YsxLite (Gatos)", ip="192.168.80.164", dev_uid="PTZC-934480-LPXPD", on_frame=None):
        self.cam_id = cam_id
        self.name = name
        self.ip = ip
        self.dev_uid = dev_uid
        self.on_frame = on_frame

        self.dev: Device | None = None
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
            "mcuver": "",
            "devType": "XR_PTZ",
        }

        self.audio_subscribers: set[asyncio.Queue] = set()
        self.audio_running = False
        self._audio_task: asyncio.Task | None = None
        self._task: asyncio.Task | None = None
        self._stop_flag = False

    def start(self, loop: asyncio.AbstractEventLoop):
        self._stop_flag = False
        if self._task is None or self._task.done():
            self._task = loop.create_task(self._run_loop())

    async def stop(self):
        self._stop_flag = True
        if self.dev:
            try:
                await self.dev.close()
            except Exception:
                pass
        self.connected = False

    async def _discover_ip(self) -> tuple[str | None, socket.socket | None, tuple[str, int] | None]:
        """Try local LAN UDP 32108 first; if away from home, use P2P WAN lookup."""
        def _scan():
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.5)
            try:
                s.sendto(b"\xf1\x30\x00\x00", (self.ip, 32108))
                try:
                    data, addr = s.recvfrom(1024)
                    if len(data) >= 16:
                        s.close()
                        return addr[0], None, None
                except socket.timeout:
                    pass
            except Exception:
                pass
            finally:
                try:
                    s.close()
                except Exception:
                    pass

            # Try P2P WAN lookup for PTZC-934480-LPXPD
            parts = (self.dev_uid or "PTZC-934480-LPXPD").split("-")
            if len(parts) == 3:
                prefix_b = parts[0].encode("ascii")[:4].ljust(4, b"\x00")
                serial_n = int(parts[1])
                suffix_b = parts[2].encode("ascii")[:8].ljust(8, b"\x00")
                did = struct.pack(">4sQ8s", prefix_b, serial_n, suffix_b)

                ws = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                ws.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                ws.bind(("0.0.0.0", 0))
                local_port = ws.getsockname()[1]
                ws.settimeout(0.25)

                hello = b"\xf1\x00\x00\x00"
                sockaddr = struct.pack(
                    ">HH4s8s",
                    2,
                    ((local_port & 0xFF) << 8) | (local_port >> 8),
                    b"\x01\x08\x0a\x0a",
                    b"\x00" * 8,
                )
                p2preq = b"\xf1\x20\x00\x24" + did + sockaddr
                punch = b"\xf1\x41\x00\x14" + did

                for srv in [("162.62.63.154", 32100), ("139.155.68.77", 32100), ("119.45.114.92", 32100)]:
                    try:
                        ws.sendto(hello, srv)
                        ws.sendto(p2preq, srv)
                    except Exception:
                        pass

                targets = set()
                deadline = time.time() + 3.5
                while time.time() < deadline:
                    for t in list(targets):
                        try:
                            ws.sendto(punch, t)
                        except Exception:
                            pass
                    try:
                        data, addr = ws.recvfrom(2048)
                        if len(data) >= 2:
                            typ = data[1]
                            if typ == 0x40 and len(data) >= 12:
                                raw = data[4:20]
                                port = struct.unpack("<H", raw[2:4])[0]
                                ip_str = ".".join(str(b) for b in raw[4:8][::-1])
                                targets.add((ip_str, port))
                                ws.sendto(punch, (ip_str, port))
                            elif typ in (0x41, 0x42):
                                return None, ws, addr
                    except socket.timeout:
                        pass
                ws.close()
            return self.ip, None, None

        return await asyncio.to_thread(_scan)

    def _handle_video_frame(self, frame_data: bytes):
        now = time.time()
        self.last_frame = bytes(frame_data)
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

    async def _audio_pump_loop(self):
        while self.connected and self.dev and self.dev.is_connected and self.audio_running:
            try:
                af = await asyncio.wait_for(self.dev.session.get_audio_frame(), timeout=5.0)
                if af and af.data and self.audio_subscribers:
                    dead = []
                    for q in self.audio_subscribers:
                        try:
                            q.put_nowait(af.data)
                        except asyncio.QueueFull:
                            dead.append(q)
                    for q in dead:
                        self.audio_subscribers.discard(q)
            except asyncio.TimeoutError:
                if not self.audio_subscribers:
                    break
            except Exception:
                break
        self.audio_running = False

    async def _run_loop(self):
        while not self._stop_flag:
            try:
                self.connecting = True
                discovered_ip, wan_sock, wan_addr = await self._discover_ip()
                if wan_sock is not None and wan_addr is not None:
                    print(f"[YsxLite] Connected via P2P WAN to {wan_addr[0]}:{wan_addr[1]} ({self.dev_uid})")
                    wan_sock.setblocking(False)
                    wan_sock.connect(wan_addr)
                    parts = (self.dev_uid or "PTZC-934480-LPXPD").split("-")
                    dev_id = DeviceID(parts[0], parts[1], parts[2])
                    desc = DeviceDescriptor(dev_id, wan_addr[0], wan_addr[1], Encryption.NONE, False)
                    dev = Device(wan_addr[0])
                    dev.enable_reconnect = False
                    dev.descriptor = desc
                    sess = make_session(
                        device=desc,
                        login="",
                        password="",
                        on_device_lost=lambda d: dev.on_device_lost(),
                    )

                    async def _custom_udp(s=wan_sock, session_ref=sess):
                        loop = asyncio.get_running_loop()
                        t, _ = await loop.create_datagram_endpoint(
                            lambda: SessionUDPProtocol(lambda data: session_ref.on_receive(data)),
                            sock=s,
                        )
                        return t

                    sess.create_udp = _custom_udp
                    dev._session = sess
                    sess.start()
                    await asyncio.wait_for(sess.device_is_ready.wait(), timeout=12.0)
                    dev.properties = sess.dev_properties or {}
                else:
                    if discovered_ip and discovered_ip != self.ip:
                        print(f"[YsxLite] Discovered camera at new IP: {discovered_ip}")
                        self.ip = discovered_ip
                    dev = Device(self.ip)
                    dev.enable_reconnect = False
                    await asyncio.wait_for(dev.connect(), timeout=14.0)

                self.dev = dev
                self.connected = True
                self.connecting = False
                self.last_error = None

                st = dev.properties or {}
                if st:
                    self.status_info["ir"] = bool(st.get("icut", 1))
                    self.status_info["lamp"] = bool(st.get("lamp", 0))
                    self.status_info["mcuver"] = str(st.get("mcuver", ""))
                    self.status_info["devType"] = str(st.get("devType", "XR_PTZ"))

                await dev.start_video()
                print(f"[YsxLite] Connected & streaming video from {self.ip} ({self.dev_uid})")

                while not self._stop_flag and dev.is_connected:
                    frame = await asyncio.wait_for(dev.get_video_frame(), timeout=12.0)
                    if frame and frame.data:
                        self._handle_video_frame(frame.data)

            except Exception as e:
                self.last_error = str(e)
                print(f"[YsxLite] Connection/stream cycle notice ({self.ip}): {e}")
            finally:
                self.connected = False
                self.connecting = False
                self.audio_running = False
                if self.dev:
                    try:
                        await self.dev.close()
                    except Exception:
                        pass
                    self.dev = None

            if not self._stop_flag:
                await asyncio.sleep(3.0)

    async def ensure_audio(self):
        if self.dev and self.connected and not self.audio_running:
            try:
                await self.dev.session.start_audio()
                self.audio_running = True
                self._audio_task = asyncio.create_task(self._audio_pump_loop())
            except Exception as e:
                print(f"[YsxLite] start_audio error: {e}")

    async def ptz(self, action: str):
        if not self.dev or not self.connected:
            return {"ok": False, "error": "Cámara YsxLite no conectada"}
        # Invert directions for Camera 1 hardware orientation (up<->down, left<->right)
        invert_map = {
            "up": "down",
            "down": "up",
            "left": "right",
            "right": "left",
        }
        try:
            sess = self.dev.session
            if action == "stop":
                await sess.rotate_stop()
            elif action.startswith("step_"):
                d = invert_map.get(action.replace("step_", ""))
                if d:
                    await sess.step_rotate(d)
            elif action in invert_map:
                await sess.rotate_start(invert_map[action])
                asyncio.create_task(self._delayed_stop(0.45))
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def _delayed_stop(self, delay: float):
        await asyncio.sleep(delay)
        if self.dev and self.connected:
            try:
                await self.dev.session.rotate_stop()
            except Exception:
                pass

    async def control(self, setting: str, value=None):
        if setting == "reconnect":
            if self.dev:
                try:
                    await self.dev.close()
                except Exception:
                    pass
            return {"ok": True, "message": "Reconectando cámara YsxLite..."}

        if not self.dev or not self.connected:
            return {"ok": False, "error": "Cámara YsxLite no conectada"}

        try:
            sess = self.dev.session
            if setting == "ir":
                new_val = not self.status_info["ir"] if value is None else bool(value)
                await sess.toggle_ir(new_val)
                self.status_info["ir"] = new_val
            elif setting == "lamp":
                new_val = not self.status_info["lamp"] if value is None else bool(value)
                await sess.toggle_whitelight(new_val)
                self.status_info["lamp"] = new_val
            elif setting == "flip":
                rot_modes = ["normal", "v", "h", "hv"]
                new_flip = (self.status_info["flip"] + 1) % 4 if value is None else int(value) % 4
                await sess.set_rotate(rot_modes[new_flip])
                self.status_info["flip"] = new_flip
            elif setting == "resolution":
                res_str = str(value or ("sd" if self.status_info["resolution"] == "hd" else "hd")).lower()
                await sess.set_resolution(res_str)
                self.status_info["resolution"] = res_str
            return {"ok": True, "status": self.status_info}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_summary(self):
        return {
            "id": self.cam_id,
            "name": self.name,
            "protocol": "YsxLite / PPPP Direct LAN",
            "dev_uid": self.dev_uid,
            "ip": self.ip,
            "connected": self.connected,
            "connecting": self.connecting,
            "fps": self.fps if self.connected else 0.0,
            "last_frame_age": round(time.time() - self.last_frame_ts, 1) if self.last_frame_ts else None,
            "last_error": self.last_error,
            "status": self.status_info,
            "capabilities": ["ptz", "ir", "lamp", "flip", "resolution", "audio"],
        }
