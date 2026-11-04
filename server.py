import os
import json
import time
import asyncio
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

from ysxlite_client import YsxLiteCamera
from v380_client import V380Camera
from cat_detector import CatDetector, CAPTURES_DIR
from gdrive_sync import GoogleDriveManager, SAVED_DIR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

DEFAULT_CONFIG = {
    "ysxlite_ip": "192.168.80.164",
    "ysxlite_uid": "PTZC-934480-LPXPD",
    "v380_dev_id": "111085293",
    "v380_ip": "192.168.80.114",
    "v380_username": "admin",
    "v380_password": "",
    "ai_enabled": True,
    "motion_enabled": False,
    "auto_capture_cat": True,
    "auto_capture_motion": True,
    "cat_confidence": 0.32,
    "motion_threshold": 2.2,
    "capture_cooldown": 25,
    "ntfy_enabled": True,
    "ntfy_topic": "michicam-alertas-934480",
    "telegram_enabled": False,
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    "gdrive_folder_name": "MichiCam_Cámaras",
    "gdrive_local_mirror": "",
    "public_tunnel_url": "",
    "cloud_server_url": "",
}


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg


def save_config(cfg: dict):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Server] Error saving config: {e}")


def make_placeholder_jpeg(title: str, subtitle: str) -> bytes:
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (22, 16, 12)
    cv2.putText(img, title, (40, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (235, 210, 255), 2)
    cv2.putText(img, subtitle, (40, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 175, 200), 1)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else b""


def ensure_app_icon():
    icon_path = os.path.join(STATIC_DIR, "icon.png")
    if os.path.exists(icon_path):
        return
    img = np.zeros((192, 192, 3), dtype=np.uint8)
    for y in range(192):
        for x in range(192):
            img[y, x] = (int(240 - x * 0.4), int(90 + y * 0.2), int(140 + x * 0.5))
    cv2.circle(img, (96, 105), 48, (255, 255, 255), -1)
    pts_l = np.array([[52, 85], [62, 42], [90, 68]], np.int32)
    pts_r = np.array([[140, 85], [130, 42], [102, 68]], np.int32)
    cv2.fillPoly(img, [pts_l, pts_r], (255, 255, 255))
    cv2.circle(img, (78, 100), 7, (35, 20, 45), -1)
    cv2.circle(img, (114, 100), 7, (35, 20, 45), -1)
    cv2.circle(img, (96, 114), 5, (180, 105, 255), -1)
    cv2.imwrite(icon_path, img)


class MichiCamApp:
    def __init__(self):
        ensure_app_icon()
        self.config = load_config()
        save_config(self.config)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.gdrive = GoogleDriveManager(lambda: self.config)
        self.detector = CatDetector(lambda: self.config, gdrive_manager=self.gdrive)

        self.ysxlite = YsxLiteCamera(
            cam_id="ysxlite",
            name="Cámara 1",
            ip=self.config["ysxlite_ip"],
            dev_uid=self.config["ysxlite_uid"],
            on_frame=self._on_camera_frame,
        )
        self.v380 = V380Camera(
            cam_id="v380",
            name="Cámara 2",
            dev_id=self.config["v380_dev_id"],
            ip=self.config["v380_ip"],
            username=self.config["v380_username"],
            password=self.config["v380_password"],
            on_frame=self._on_camera_frame,
        )
        self.cameras = {
            "ysxlite": self.ysxlite,
            "v380": self.v380,
        }
        self._placeholders = {
            "ysxlite": make_placeholder_jpeg("Conectando Camara 1...", "192.168.80.164"),
            "v380": make_placeholder_jpeg("Conectando Camara 2...", "192.168.80.114"),
        }

    def _on_camera_frame(self, cam_id: str, cam_name: str, jpeg_bytes: bytes):
        self.detector.on_raw_frame(cam_id, jpeg_bytes)
        now = time.time()
        if now - self.detector.last_analysis_ts.get(cam_id, 0) >= 0.65:
            self.executor.submit(self.detector.process_frame, cam_id, cam_name, jpeg_bytes)

    async def start(self):
        loop = asyncio.get_running_loop()
        self.ysxlite.start(loop)
        self.v380.start(loop)
        if not os.environ.get("RENDER"):
            loop.create_task(self._cloud_relay_loop())

    async def _cloud_relay_loop(self):
        import urllib.request
        last_pushed = {"ysxlite": 0.0, "v380": 0.0}
        while True:
            try:
                cloud_url = (self.config.get("cloud_server_url") or "").strip().rstrip("/")
                if not cloud_url:
                    await asyncio.sleep(1.5)
                    continue
                pushed_any = False
                for cid, cam_obj in (("ysxlite", self.ysxlite), ("v380", self.v380)):
                    if cam_obj.connected and cam_obj.last_frame and cam_obj.last_frame_ts != last_pushed[cid]:
                        last_pushed[cid] = cam_obj.last_frame_ts
                        jpg = cam_obj.last_frame
                        def _push(c=cid, data=jpg):
                            req = urllib.request.Request(
                                f"{cloud_url}/api/camera/{c}/push_frame",
                                data=data,
                                headers={"Content-Type": "image/jpeg"},
                                method="POST",
                            )
                            with urllib.request.urlopen(req, timeout=3.0) as _:
                                pass
                        await asyncio.to_thread(_push)
                        pushed_any = True
                await asyncio.sleep(0.2 if pushed_any else 0.8)
            except Exception:
                await asyncio.sleep(2.0)

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            req_line = await asyncio.wait_for(reader.readline(), timeout=10.0)
            if not req_line:
                writer.close()
                return
            parts = req_line.decode("latin-1", errors="ignore").strip().split(" ")
            if len(parts) < 2:
                writer.close()
                return
            method, raw_path = parts[0].upper(), parts[1]
            parsed = urllib.parse.urlparse(raw_path)
            path = urllib.parse.unquote(parsed.path)
            query = urllib.parse.parse_qs(parsed.query)

            headers = {}
            while True:
                hline = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if not hline or hline in (b"\r\n", b"\n"):
                    break
                hstr = hline.decode("latin-1", errors="ignore").strip()
                if ":" in hstr:
                    k, v = hstr.split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            body = b""
            clen = int(headers.get("content-length", "0"))
            if clen > 0:
                body = await asyncio.wait_for(reader.readexactly(clen), timeout=10.0)

            await self.route(method, path, query, body, writer)
        except Exception:
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _send_json(self, writer: asyncio.StreamWriter, data: dict, status="200 OK"):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        hdr = (
            f"HTTP/1.1 {status}\r\n"
            f"Content-Type: application/json; charset=utf-8\r\n"
            f"Content-Length: {len(payload)}\r\n"
            f"Cache-Control: no-cache\r\n"
            f"Connection: close\r\n\r\n"
        ).encode("latin-1")
        writer.write(hdr + payload)
        await writer.drain()

    async def _send_file(self, writer: asyncio.StreamWriter, filepath: str, content_type: str):
        if not os.path.isfile(filepath):
            await self._send_json(writer, {"error": "Not found"}, "404 Not Found")
            return
        with open(filepath, "rb") as f:
            data = f.read()
        hdr = (
            f"HTTP/1.1 200 OK\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(data)}\r\n"
            f"Accept-Ranges: bytes\r\n"
            f"Cache-Control: no-cache\r\n"
            f"Connection: close\r\n\r\n"
        ).encode("latin-1")
        writer.write(hdr + data)
        await writer.drain()

    async def route(self, method: str, path: str, query: dict, body: bytes, writer: asyncio.StreamWriter):
        if path in ("/", "/index.html"):
            await self._send_file(writer, os.path.join(STATIC_DIR, "index.html"), "text/html; charset=utf-8")
            return
        if path == "/manifest.json":
            await self._send_file(writer, os.path.join(STATIC_DIR, "manifest.json"), "application/json; charset=utf-8")
            return
        if path in ("/apple-touch-icon.png", "/icon.png"):
            await self._send_file(writer, os.path.join(STATIC_DIR, "icon.png"), "image/png")
            return
        if path.startswith("/captures/"):
            fname = os.path.basename(path)
            await self._send_file(writer, os.path.join(CAPTURES_DIR, fname), "image/jpeg")
            return
        if path.startswith("/saved/"):
            rel = path.split("/saved/", 1)[1]
            parts = rel.split("/")
            if len(parts) == 2 and parts[0] in ("Camara_1", "Camara_2"):
                fpath = os.path.join(SAVED_DIR, parts[0], os.path.basename(parts[1]))
                ctype = "video/mp4" if fpath.endswith(".mp4") else "image/jpeg"
                await self._send_file(writer, fpath, ctype)
                return

        if path == "/api/status" and method == "GET":
            await self._send_json(writer, {
                "cameras": [self.ysxlite.get_summary(), self.v380.get_summary()],
                "detector": self.detector.stats,
                "detections": self.detector.latest_detections,
                "recordings": self.detector.get_recording_status(),
                "gdrive": self.gdrive.get_status(),
                "ntfy_topic": self.config.get("ntfy_topic", ""),
                "public_tunnel_url": self.config.get("public_tunnel_url", ""),
            })
            return

        if path == "/api/events" and method == "GET":
            await self._send_json(writer, {"events": self.detector.events[:30]})
            return

        if path == "/api/events/clear" and method == "POST":
            self.detector.clear_temp_events()
            await self._send_json(writer, {"ok": True})
            return

        if path.startswith("/api/events/") and path.endswith("/save") and method == "POST":
            ev_id = path.split("/api/events/", 1)[1].replace("/save", "")
            saved_item = self.detector.promote_detection_to_drive(ev_id)
            if saved_item:
                await self._send_json(writer, {"ok": True, "item": saved_item})
            else:
                await self._send_json(writer, {"ok": False, "error": "Captura no encontrada"}, "404 Not Found")
            return

        if path.startswith("/api/events/") and method == "DELETE":
            ev_id = path.split("/api/events/", 1)[1]
            ok = self.detector.delete_event(ev_id)
            await self._send_json(writer, {"ok": ok})
            return

        if path == "/api/saved" and method == "GET":
            await self._send_json(writer, {
                "items": self.gdrive.items,
                "gdrive": self.gdrive.get_status(),
            })
            return

        if path.startswith("/api/saved/") and method == "DELETE":
            item_id = path.split("/api/saved/", 1)[1]
            ok = self.gdrive.delete_saved_item(item_id)
            await self._send_json(writer, {"ok": ok})
            return

        if path == "/api/notify/test" and method == "POST":
            topic = self.config.get("ntfy_topic", "michicam-alertas-934480")
            ok_ntfy = await asyncio.to_thread(
                self.detector.send_ntfy_alert,
                topic,
                "",
                "MichiCam 🐾 Prueba de Alerta",
                "¡Las notificaciones en tu iPhone 17 están funcionando perfectamente!",
            )
            await self._send_json(writer, {"ok": ok_ntfy, "topic": topic})
            return

        if path == "/api/config":
            if method == "GET":
                await self._send_json(writer, self.config)
                return
            if method == "POST":
                updates = json.loads(body.decode("utf-8") or "{}")
                self.config.update(updates)
                save_config(self.config)
                self.v380.update_credentials(
                    dev_id=self.config.get("v380_dev_id"),
                    ip=self.config.get("v380_ip"),
                    username=self.config.get("v380_username"),
                    password=self.config.get("v380_password"),
                )
                await self._send_json(writer, {"ok": True, "config": self.config})
                return

        if path.startswith("/api/stream/") and path.endswith(".mjpg"):
            cam_id = path.split("/api/stream/", 1)[1].replace(".mjpg", "")
            cam = self.cameras.get(cam_id)
            if not cam:
                await self._send_json(writer, {"error": "Camera not found"}, "404 Not Found")
                return
            draw_ai = query.get("ai", ["1"])[0] == "1"
            await self._stream_mjpeg(writer, cam_id, cam, draw_ai)
            return

        if path.startswith("/api/audio/") and path.endswith(".pcm"):
            cam_id = path.split("/api/audio/", 1)[1].replace(".pcm", "")
            cam = self.cameras.get(cam_id)
            if cam:
                await self._stream_audio_pcm(writer, cam)
                return

        if path.startswith("/api/camera/"):
            segs = path.strip("/").split("/")
            if len(segs) == 4:
                cam_id, action = segs[2], segs[3]
                cam = self.cameras.get(cam_id)
                if not cam:
                    await self._send_json(writer, {"ok": False, "error": "Cámara no encontrada"}, "404 Not Found")
                    return
                if action == "push_frame" and body:
                    if hasattr(cam, "ingest_pushed_frame"):
                        cam.ingest_pushed_frame(body)
                    await self._send_json(writer, {"ok": True})
                    return
                payload = json.loads(body.decode("utf-8") or "{}") if body else {}
                if action == "ptz":
                    self.detector.notify_ptz(cam_id)
                    res = await cam.ptz(payload.get("direction", "stop"))
                    await self._send_json(writer, res)
                    return
                if action == "control":
                    res = await cam.control(payload.get("setting", ""), payload.get("value"))
                    await self._send_json(writer, res)
                    return
                if action == "capture":
                    if not cam.last_frame:
                        await self._send_json(writer, {"ok": False, "error": "La cámara aún no tiene imagen en vivo"})
                        return
                    arr = np.frombuffer(cam.last_frame, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    item = self.detector.save_manual_photo_to_drive(
                        cam_id=cam_id,
                        cam_name=cam.name,
                        frame=frame,
                        detections=self.detector.latest_detections.get(cam_id),
                    )
                    await self._send_json(writer, {"ok": True, "item": item})
                    return
                if action == "record":
                    if cam_id in self.detector.active_recorders:
                        res = self.detector.stop_recording(cam_id)
                    else:
                        res = self.detector.start_recording(cam_id, cam.name)
                    await self._send_json(writer, res)
                    return

        await self._send_json(writer, {"error": "Endpoint not found"}, "404 Not Found")

    async def _stream_mjpeg(self, writer: asyncio.StreamWriter, cam_id: str, cam, draw_ai: bool):
        hdr = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
            b"Cache-Control: no-cache, no-store, must-revalidate\r\n"
            b"Pragma: no-cache\r\n"
            b"Connection: close\r\n\r\n"
        )
        writer.write(hdr)
        await writer.drain()

        last_sent_ts = 0.0
        while True:
            if cam.connected and cam.last_frame:
                if cam.last_frame_ts == last_sent_ts:
                    await asyncio.sleep(0.04)
                    continue
                last_sent_ts = cam.last_frame_ts
                jpg = cam.last_frame
                if draw_ai and self.detector.latest_detections.get(cam_id):
                    jpg = await asyncio.to_thread(self.detector.render_annotated_jpeg, cam_id, jpg)
            else:
                jpg = self._placeholders.get(cam_id, b"")
                await asyncio.sleep(1.0)

            chunk = (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpg)}\r\n\r\n".encode("latin-1")
                + jpg
                + b"\r\n"
            )
            writer.write(chunk)
            await writer.drain()
            await asyncio.sleep(0.03)

    async def _stream_audio_pcm(self, writer: asyncio.StreamWriter, cam):
        await cam.ensure_audio()
        q: asyncio.Queue = asyncio.Queue(maxsize=40)
        cam.audio_subscribers.add(q)
        hdr = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/octet-stream\r\n"
            b"Cache-Control: no-cache\r\n"
            b"Connection: close\r\n\r\n"
        )
        writer.write(hdr)
        await writer.drain()
        try:
            while True:
                pcm = await asyncio.wait_for(q.get(), timeout=15.0)
                writer.write(pcm)
                await writer.drain()
        finally:
            cam.audio_subscribers.discard(q)


async def main():
    port = int(os.environ.get("PORT", 8080))
    app = MichiCamApp()
    await app.start()
    server = await asyncio.start_server(app.handle_client, "0.0.0.0", port)
    print("==========================================================")
    print(f"🐾 MichiCam corriendo en: http://localhost:{port}")
    print("==========================================================")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
