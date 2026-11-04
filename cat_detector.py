import os
import json
import time
import shutil
import threading
import urllib.request
from datetime import datetime
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
EVENTS_FILE = os.path.join(CAPTURES_DIR, "events.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov5n.onnx")

os.makedirs(CAPTURES_DIR, exist_ok=True)

MAX_TEMP_DETECTIONS = 30

COCO_CLASSES = {
    0: ("person", "Persona", (255, 180, 50)),
    14: ("bird", "Ave", (80, 220, 255)),
    15: ("cat", "Michi", (180, 105, 255)),
    16: ("dog", "Perro", (80, 200, 120)),
}


class VideoRecorder:
    def __init__(self, cam_id: str, cam_name: str, filepath: str, filename: str):
        self.cam_id = cam_id
        self.cam_name = cam_name
        self.filepath = filepath
        self.filename = filename
        self.started_at = time.time()
        self.writer: cv2.VideoWriter | None = None
        self.frame_count = 0
        self.lock = threading.Lock()
        self.closed = False

    def write_jpeg(self, jpeg_bytes: bytes):
        with self.lock:
            if self.closed:
                return
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return
            h, w = frame.shape[:2]
            if self.writer is None:
                for fourcc_str in ("avc1", "mp4v"):
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                    wtr = cv2.VideoWriter(self.filepath, fourcc, 10.0, (w, h))
                    if wtr.isOpened():
                        self.writer = wtr
                        break
            if self.writer is not None:
                self.writer.write(frame)
                self.frame_count += 1

    def stop(self) -> float:
        with self.lock:
            self.closed = True
            if self.writer is not None:
                self.writer.release()
                self.writer = None
            return round(time.time() - self.started_at, 1)


class CatDetector:
    def __init__(self, config_getter, gdrive_manager=None):
        self.config_getter = config_getter
        self.gdrive = gdrive_manager
        self.lock = threading.Lock()
        self.net = None
        if os.path.exists(MODEL_PATH):
            try:
                self.net = cv2.dnn.readNetFromONNX(MODEL_PATH)
            except Exception as e:
                print(f"[CatDetector] Error loading ONNX model: {e}")

        self.prev_gray = {}
        self.last_analysis_ts = {}
        self.last_capture_ts = {}
        self.last_ptz_ts = {}
        self.latest_detections = {}
        self.active_recorders: dict[str, VideoRecorder] = {}

        self.stats = {
            "cats_detected_today": 0,
            "motion_events_today": 0,
            "last_cat_seen": None,
            "last_event": None,
        }
        self.events = [ev for ev in self._load_events() if ev.get("type") in ("cat", "person", "motion")]
        self._prune_temp_events()
        self._save_events()
        self._recalculate_stats()

    def notify_ptz(self, cam_id: str):
        """Pause detection while the camera motor is moving so panning is never mistaken for activity."""
        self.last_ptz_ts[cam_id] = time.time()
        self.prev_gray[cam_id] = None
        self.latest_detections[cam_id] = []

    def _load_events(self):
        if os.path.exists(EVENTS_FILE):
            try:
                with open(EVENTS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_events(self):
        try:
            with open(EVENTS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.events, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[CatDetector] Error saving events: {e}")

    def _prune_temp_events(self):
        """Keep at most MAX_TEMP_DETECTIONS temporary detections and delete older files automatically."""
        while len(self.events) > MAX_TEMP_DETECTIONS:
            old = self.events.pop()
            fname = old.get("filename")
            if fname:
                fpath = os.path.join(CAPTURES_DIR, fname)
                if os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass

    def _recalculate_stats(self):
        today = datetime.now().strftime("%Y-%m-%d")
        cats = 0
        persons = 0
        last_cat = None
        for ev in self.events:
            if ev.get("timestamp", "").startswith(today):
                if ev.get("type") == "cat":
                    cats += 1
                elif ev.get("type") in ("person", "motion"):
                    persons += 1
            if ev.get("type") in ("cat", "person", "motion") and not last_cat:
                last_cat = ev.get("timestamp_human")
        self.stats["cats_detected_today"] = cats
        self.stats["motion_events_today"] = persons
        self.stats["last_cat_seen"] = last_cat
        self.stats["last_event"] = self.events[0] if self.events else None

    def on_raw_frame(self, cam_id: str, jpeg_bytes: bytes):
        """Called on every incoming frame so active video recordings get smooth ~10 FPS frames."""
        rec = self.active_recorders.get(cam_id)
        if rec and not rec.closed:
            rec.write_jpeg(jpeg_bytes)
            # Auto-save 15-minute (900s) blocks to Google Drive so unlimited recording never loses data
            if time.time() - rec.started_at >= 900:
                cam_name = rec.cam_name
                self.stop_recording(cam_id)
                self.start_recording(cam_id, cam_name)

    def start_recording(self, cam_id: str, cam_name: str) -> dict:
        if cam_id in self.active_recorders:
            return {"ok": False, "error": "Ya se está grabando en esta cámara"}
        ts_now = datetime.now()
        rec_id = ts_now.strftime("%Y%m%d_%H%M%S") + f"_{cam_id}"
        filename = f"Grabacion_{rec_id}.mp4"
        sub_dir = self.gdrive.get_cam_subfolder(cam_id)
        filepath = os.path.join(sub_dir, filename)
        self.active_recorders[cam_id] = VideoRecorder(cam_id, cam_name, filepath, filename)
        return {"ok": True, "recording": True, "cam_id": cam_id}

    def stop_recording(self, cam_id: str) -> dict:
        rec = self.active_recorders.pop(cam_id, None)
        if not rec:
            return {"ok": False, "error": "No hay grabación activa en esta cámara"}
        duration = rec.stop()
        ts_now = datetime.now()
        sub_name = "Camara_1" if cam_id == "ysxlite" else "Camara_2"
        item = {
            "id": rec.filename.replace(".mp4", ""),
            "cam_id": cam_id,
            "cam_name": rec.cam_name,
            "media_type": "video",
            "title": f"🎥 Video grabado ({duration}s)",
            "filename": rec.filename,
            "url": f"/saved/{sub_name}/{rec.filename}",
            "timestamp": ts_now.isoformat(),
            "timestamp_human": ts_now.strftime("%d/%m/%Y %H:%M:%S"),
            "drive_synced": False,
            "drive_url": None,
        }
        if self.gdrive and os.path.exists(rec.filepath):
            self.gdrive.register_saved_item(item, rec.filepath)
        return {"ok": True, "recording": False, "item": item}

    def get_recording_status(self) -> dict:
        out = {}
        now = time.time()
        for cid, rec in list(self.active_recorders.items()):
            out[cid] = {
                "recording": True,
                "seconds": int(now - rec.started_at),
            }
        return out

    def process_frame(self, cam_id: str, cam_name: str, jpeg_bytes: bytes):
        now = time.time()
        if now - self.last_ptz_ts.get(cam_id, 0) < 3.5:
            self.latest_detections[cam_id] = []
            return

        last_ts = self.last_analysis_ts.get(cam_id, 0)
        if now - last_ts < 0.65:
            return
        self.last_analysis_ts[cam_id] = now

        cfg = self.config_getter()
        if not cfg.get("ai_enabled", True):
            self.latest_detections[cam_id] = []
            return

        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        small = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)
        prev = self.prev_gray.get(cam_id)
        self.prev_gray[cam_id] = gray
        global_shift = 0.0
        if prev is not None and prev.shape == gray.shape:
            diff = cv2.absdiff(prev, gray)
            thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            global_shift = float(np.count_nonzero(thresh)) / (320.0 * 180.0) * 100.0
            if global_shift > 35.0:
                self.latest_detections[cam_id] = []
                return

        detections = []
        if self.net is not None:
            with self.lock:
                try:
                    detections = self._run_yolo(frame, float(cfg.get("cat_confidence", 0.32)))
                except Exception as e:
                    print(f"[CatDetector] YOLO error: {e}")

        self.latest_detections[cam_id] = detections

        cooldown = float(cfg.get("capture_cooldown", 25))
        if now - self.last_capture_ts.get(cam_id, 0) < cooldown:
            return

        cat_dets = [d for d in detections if d["class"] in ("cat", "dog", "bird")]
        person_dets = [d for d in detections if d["class"] == "person"]
        motion_thresh = float(cfg.get("motion_threshold", 2.0))

        if cat_dets and cfg.get("auto_capture_cat", True):
            best = max(cat_dets, key=lambda d: d["confidence"])
            self.last_capture_ts[cam_id] = now
            label_txt = f"🐱 {best['label']} detectado ({int(best['confidence']*100)}%)"
            self.save_temp_detection(
                cam_id=cam_id,
                cam_name=cam_name,
                frame=frame,
                detections=detections,
                event_type="cat",
                title=label_txt,
                confidence=best["confidence"],
            )
        elif person_dets and cfg.get("auto_capture_person", cfg.get("auto_capture_motion", True)):
            best = max(person_dets, key=lambda d: d["confidence"])
            self.last_capture_ts[cam_id] = now
            label_txt = f"🧑 Persona detectada ({int(best['confidence']*100)}%)"
            self.save_temp_detection(
                cam_id=cam_id,
                cam_name=cam_name,
                frame=frame,
                detections=detections,
                event_type="person",
                title=label_txt,
                confidence=best["confidence"],
            )
        elif global_shift >= motion_thresh and cfg.get("auto_capture_motion", True):
            self.last_capture_ts[cam_id] = now
            label_txt = f"👀 Movimiento detectado ({round(global_shift, 1)}%)"
            self.save_temp_detection(
                cam_id=cam_id,
                cam_name=cam_name,
                frame=frame,
                detections=detections,
                event_type="motion",
                title=label_txt,
                confidence=min(1.0, global_shift / 15.0),
            )

    def _run_yolo(self, frame, conf_thres=0.32):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = self.net.forward()[0]

        mask = preds[:, 4] > (conf_thres * 0.8)
        rows = preds[mask]
        if len(rows) == 0:
            return []

        boxes = []
        confidences = []
        class_ids = []

        x_factor = w / 640.0
        y_factor = h / 640.0

        for row in rows:
            obj_conf = row[4]
            scores = row[5:]
            cid = int(np.argmax(scores))
            if cid not in COCO_CLASSES:
                continue
            conf = float(obj_conf * scores[cid])
            effective_thres = conf_thres * 0.85 if cid == 15 else conf_thres
            if conf < effective_thres:
                continue

            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            left = int((cx - 0.5 * bw) * x_factor)
            top = int((cy - 0.5 * bh) * y_factor)
            width = int(bw * x_factor)
            height = int(bh * y_factor)

            boxes.append([left, top, width, height])
            confidences.append(conf)
            class_ids.append(cid)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres * 0.8, 0.45)
        results = []
        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                cid = class_ids[idx]
                cls_key, cls_es, color = COCO_CLASSES[cid]
                results.append({
                    "class": cls_key,
                    "label": cls_es,
                    "confidence": round(confidences[idx], 3),
                    "box": boxes[idx],
                    "color": list(color),
                })
        return results

    def render_annotated_jpeg(self, cam_id: str, jpeg_bytes: bytes) -> bytes:
        dets = self.latest_detections.get(cam_id, [])
        if not dets:
            return jpeg_bytes
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jpeg_bytes
        annotated = self._draw_boxes(frame, dets)
        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return buf.tobytes() if ok else jpeg_bytes

    def _draw_boxes(self, frame, detections):
        img = frame.copy()
        h, w = img.shape[:2]
        for d in detections:
            x, y, bw, bh = d["box"]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w - 1, x + bw)
            y2 = min(h - 1, y + bh)
            color = tuple(d.get("color", [180, 105, 255]))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{d['label']} {int(d.get('confidence', 0)*100)}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            ty = max(y1, th + 10)
            cv2.rectangle(img, (x1, ty - th - 10), (x1 + tw + 12, ty), color, -1)
            cv2.putText(img, label, (x1 + 6, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        return img

    def save_temp_detection(self, cam_id: str, cam_name: str, frame, detections=None, event_type="cat", title="🐱 Michi detectado", confidence=1.0):
        """Saves a temporary detection (NOT uploaded to Google Drive unless user clicks 'Guardar')."""
        ts_now = datetime.now()
        ev_id = ts_now.strftime("%Y%m%d_%H%M%S_%f")[:19] + f"_{cam_id}"
        filename = f"{ev_id}.jpg"
        filepath = os.path.join(CAPTURES_DIR, filename)

        out_img = self._draw_boxes(frame, detections) if detections else frame
        cv2.imwrite(filepath, out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        ev = {
            "id": ev_id,
            "cam_id": cam_id,
            "cam_name": cam_name,
            "type": event_type,
            "title": title,
            "confidence": round(float(confidence), 2),
            "filename": filename,
            "url": f"/captures/{filename}",
            "timestamp": ts_now.isoformat(),
            "timestamp_human": ts_now.strftime("%d/%m/%Y %H:%M:%S"),
            "saved_to_drive": False,
        }
        self.events.insert(0, ev)
        self._prune_temp_events()
        self._save_events()
        self._recalculate_stats()

        cfg = self.config_getter()
        # 1. iPhone Notification via ntfy.sh (zero-registration push notification to iPhone)
        if cfg.get("ntfy_enabled", True) and cfg.get("ntfy_topic"):
            threading.Thread(
                target=self.send_ntfy_alert,
                args=(cfg["ntfy_topic"], filepath, f"MichiCam • {cam_name}", f"{title} ({ev['timestamp_human']})"),
                daemon=True,
            ).start()

        # 2. Telegram Photo Alert
        if cfg.get("telegram_enabled") and cfg.get("telegram_bot_token") and cfg.get("telegram_chat_id"):
            threading.Thread(
                target=self.send_telegram_photo,
                args=(cfg["telegram_bot_token"], cfg["telegram_chat_id"], filepath, f"🐾 MichiCam ({cam_name})\n{title}\n🕒 {ev['timestamp_human']}"),
                daemon=True,
            ).start()
        return ev

    def save_manual_photo_to_drive(self, cam_id: str, cam_name: str, frame, detections=None):
        """Manual snapshot goes directly to Saved / Google Drive folder."""
        ts_now = datetime.now()
        item_id = ts_now.strftime("%Y%m%d_%H%M%S") + f"_{cam_id}"
        filename = f"Foto_{item_id}.jpg"
        sub_dir = self.gdrive.get_cam_subfolder(cam_id)
        filepath = os.path.join(sub_dir, filename)
        sub_name = "Camara_1" if cam_id == "ysxlite" else "Camara_2"

        out_img = self._draw_boxes(frame, detections) if detections else frame
        cv2.imwrite(filepath, out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        item = {
            "id": item_id,
            "cam_id": cam_id,
            "cam_name": cam_name,
            "media_type": "photo",
            "title": "📸 Foto guardada",
            "filename": filename,
            "url": f"/saved/{sub_name}/{filename}",
            "timestamp": ts_now.isoformat(),
            "timestamp_human": ts_now.strftime("%d/%m/%Y %H:%M:%S"),
            "drive_synced": False,
            "drive_url": None,
        }
        if self.gdrive:
            self.gdrive.register_saved_item(item, filepath)
        return item

    def promote_detection_to_drive(self, ev_id: str) -> dict | None:
        """Copies a temporary detection photo into the permanent Google Drive folder when user clicks 'Guardar'."""
        for ev in self.events:
            if ev.get("id") == ev_id:
                src_path = os.path.join(CAPTURES_DIR, ev["filename"])
                if not os.path.exists(src_path):
                    return None
                cam_id = ev["cam_id"]
                sub_dir = self.gdrive.get_cam_subfolder(cam_id)
                sub_name = "Camara_1" if cam_id == "ysxlite" else "Camara_2"
                dest_filename = f"Deteccion_{ev['filename']}"
                dest_path = os.path.join(sub_dir, dest_filename)
                shutil.copy2(src_path, dest_path)
                ev["saved_to_drive"] = True
                self._save_events()

                item = {
                    "id": f"saved_{ev_id}",
                    "cam_id": cam_id,
                    "cam_name": ev["cam_name"],
                    "media_type": "photo",
                    "title": ev["title"],
                    "filename": dest_filename,
                    "url": f"/saved/{sub_name}/{dest_filename}",
                    "timestamp": ev["timestamp"],
                    "timestamp_human": ev["timestamp_human"],
                    "drive_synced": False,
                    "drive_url": None,
                }
                if self.gdrive:
                    self.gdrive.register_saved_item(item, dest_path)
                return item
        return None

    def delete_event(self, ev_id: str) -> bool:
        for i, ev in enumerate(self.events):
            if ev.get("id") == ev_id:
                fname = ev.get("filename")
                if fname:
                    fpath = os.path.join(CAPTURES_DIR, fname)
                    if os.path.exists(fpath):
                        try:
                            os.remove(fpath)
                        except Exception:
                            pass
                self.events.pop(i)
                self._save_events()
                self._recalculate_stats()
                return True
        return False

    def clear_temp_events(self):
        for ev in self.events:
            fname = ev.get("filename")
            if fname:
                fpath = os.path.join(CAPTURES_DIR, fname)
                if os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
        self.events.clear()
        self._save_events()
        self._recalculate_stats()

    @staticmethod
    def send_ntfy_alert(topic: str, photo_path: str, title: str, message: str):
        """Send instant push notification to iPhone via ntfy.sh JSON API (supports UTF-8 emojis & accents)."""
        try:
            payload = json.dumps({
                "topic": topic.strip(),
                "title": title,
                "message": message,
                "priority": 4,
                "tags": ["cat", "camera"],
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://ntfy.sh",
                data=payload,
                headers={"Content-Type": "application/json; charset=utf-8"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                return resp.status == 200
        except Exception as e:
            print(f"[ntfy] Error sending notification: {e}")
            return False

    @staticmethod
    def send_telegram_photo(bot_token: str, chat_id: str, photo_path: str, caption: str):
        try:
            url = f"https://api.telegram.org/bot{bot_token.strip()}/sendPhoto"
            boundary = f"----MichiCamBoundary{int(time.time()*1000)}"
            with open(photo_path, "rb") as f:
                photo_bytes = f.read()

            body = bytearray()
            for name, val in [("chat_id", chat_id.strip()), ("caption", caption)]:
                body.extend(f"--{boundary}\r\n".encode())
                body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
                body.extend(f"{val}\r\n".encode())

            body.extend(f"--{boundary}\r\n".encode())
            body.extend(b'Content-Disposition: form-data; name="photo"; filename="cat.jpg"\r\n')
            body.extend(b"Content-Type: image/jpeg\r\n\r\n")
            body.extend(photo_bytes)
            body.extend(f"\r\n--{boundary}--\r\n".encode())

            req = urllib.request.Request(url, data=bytes(body), method="POST")
            req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            print(f"[Telegram] Error sending photo: {e}")
            return False
