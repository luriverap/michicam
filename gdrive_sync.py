import os
import re
import json
import shutil
import subprocess
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(BASE_DIR, "saved_drive")
SAVED_INDEX_FILE = os.path.join(SAVED_DIR, "saved_items.json")

os.makedirs(os.path.join(SAVED_DIR, "Camara_1"), exist_ok=True)
os.makedirs(os.path.join(SAVED_DIR, "Camara_2"), exist_ok=True)


class GoogleDriveManager:
    """
    Manages permanent recordings (.mp4) and user-saved photos (.jpg),
    organized into 'MichiCam_Cámaras / Cámara 1' and 'MichiCam_Cámaras / Cámara 2',
    and syncs them to Google Drive using the connected `gdrive` CLI.
    """

    def __init__(self, config_getter):
        self.config_getter = config_getter
        self.lock = threading.Lock()
        self.items = self._load_items()
        self.drive_folder_ids = {
            "root": "1BTJJdt8T_-JWE3Syjw2cARNJB1Bg4OBE",
            "root_link": "https://drive.google.com/drive/folders/1BTJJdt8T_-JWE3Syjw2cARNJB1Bg4OBE",
            "ysxlite": "16w1B3ElzHjp_X6hNVGxunZCtOtsCR8Mr",
            "v380": "1DOVQ9GFVLcs-CPhuPUlMp8vwxwst040q",
        }
        self._last_connected_check = 0.0
        self._is_connected_cache = True

    def _load_items(self) -> list:
        if os.path.exists(SAVED_INDEX_FILE):
            try:
                with open(SAVED_INDEX_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_items(self):
        try:
            with open(SAVED_INDEX_FILE, "w", encoding="utf-8") as f:
                json.dump(self.items, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[GoogleDrive] Error saving index: {e}")

    def get_cam_subfolder(self, cam_id: str) -> str:
        sub = "Camara_1" if cam_id == "ysxlite" else "Camara_2"
        path = os.path.join(SAVED_DIR, sub)
        os.makedirs(path, exist_ok=True)
        return path

    def is_gdrive_connected(self) -> bool:
        import time
        now = time.time()
        if now - self._last_connected_check < 30.0:
            return self._is_connected_cache
        try:
            r = subprocess.run(
                ["gdrive", "readonly", "quota"],
                capture_output=True,
                text=True,
                timeout=8,
            )
            self._is_connected_cache = r.returncode == 0
            self._last_connected_check = now
            return self._is_connected_cache
        except Exception:
            self._is_connected_cache = False
            self._last_connected_check = now
            return False

    def _ensure_gdrive_folders(self) -> dict:
        if self.drive_folder_ids.get("root") and self.drive_folder_ids.get("ysxlite") and self.drive_folder_ids.get("v380"):
            return self.drive_folder_ids
        try:
            cfg = self.config_getter()
            folder_name = cfg.get("gdrive_folder_name", "MichiCam_Cámaras")
            r = subprocess.run(
                ["gdrive", "readonly", "search", "--name-exact", folder_name, "--type", "folder", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            root_id = None
            if r.returncode == 0 and r.stdout.strip():
                files = json.loads(r.stdout)
                if files:
                    root_id = files[0].get("id")
            if not root_id:
                rc = subprocess.run(
                    ["gdrive", "mutate", "mkdir", folder_name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                m = re.search(r"ID:\s*([A-Za-z0-9_-]+)", rc.stdout or "")
                if m:
                    root_id = m.group(1)
            if root_id:
                self.drive_folder_ids["root"] = root_id
                self.drive_folder_ids["root_link"] = f"https://drive.google.com/drive/folders/{root_id}"
                for cid, cname in [("ysxlite", "Cámara 1"), ("v380", "Cámara 2")]:
                    rs = subprocess.run(
                        ["gdrive", "readonly", "search", "--name-exact", cname, "--parent-id", root_id, "--type", "folder", "--json"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    sub_id = None
                    if rs.returncode == 0 and rs.stdout.strip():
                        sfiles = json.loads(rs.stdout)
                        if sfiles:
                            sub_id = sfiles[0].get("id")
                    if not sub_id:
                        rcs = subprocess.run(
                            ["gdrive", "mutate", "mkdir", cname, "--parent", root_id],
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                        ms = re.search(r"ID:\s*([A-Za-z0-9_-]+)", rcs.stdout or "")
                        if ms:
                            sub_id = ms.group(1)
                    if sub_id:
                        self.drive_folder_ids[cid] = sub_id
        except Exception as e:
            print(f"[GoogleDrive] Folder setup notice: {e}")
        return self.drive_folder_ids

    def upload_file_async(self, item_entry: dict, local_path: str):
        def _worker():
            # 1. Check if user configured a local synced Google Drive folder path
            cfg = self.config_getter()
            mirror_dir = cfg.get("gdrive_local_mirror", "").strip()
            if mirror_dir and os.path.isdir(mirror_dir):
                cam_folder = "Cámara 1" if item_entry["cam_id"] == "ysxlite" else "Cámara 2"
                target_dir = os.path.join(mirror_dir, cfg.get("gdrive_folder_name", "MichiCam_Cámaras"), cam_folder)
                try:
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.copy2(local_path, os.path.join(target_dir, item_entry["filename"]))
                    item_entry["drive_synced"] = True
                except Exception as e:
                    print(f"[GoogleDrive] Mirror copy error: {e}")

            # 2. Upload via connected `gdrive` CLI
            if self.is_gdrive_connected():
                folders = self._ensure_gdrive_folders()
                parent_id = folders.get(item_entry["cam_id"]) or folders.get("root")
                if parent_id:
                    try:
                        r = subprocess.run(
                            [
                                "gdrive",
                                "mutate",
                                "upload",
                                local_path,
                                "--parent",
                                parent_id,
                                "--json",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=600,
                        )
                        if r.returncode == 0:
                            out = r.stdout.strip()
                            file_id = None
                            web_link = None
                            try:
                                data = json.loads(out)
                                file_id = data.get("id")
                                web_link = data.get("webViewLink")
                            except Exception:
                                m = re.search(r"ID:\s*([A-Za-z0-9_-]+)", out)
                                if m:
                                    file_id = m.group(1)
                            if file_id:
                                item_entry["drive_file_id"] = file_id
                                item_entry["drive_url"] = web_link or f"https://drive.google.com/file/d/{file_id}/view"
                            item_entry["drive_synced"] = True
                            with self.lock:
                                self._save_items()
                    except Exception as e:
                        print(f"[GoogleDrive] Upload error: {e}")

        threading.Thread(target=_worker, daemon=True).start()

    def register_saved_item(self, item: dict, local_path: str) -> dict:
        with self.lock:
            self.items.insert(0, item)
            self._save_items()
        self.upload_file_async(item, local_path)
        return item

    def delete_saved_item(self, item_id: str) -> bool:
        with self.lock:
            for i, it in enumerate(self.items):
                if it.get("id") == item_id:
                    local_path = os.path.join(self.get_cam_subfolder(it["cam_id"]), it["filename"])
                    if os.path.exists(local_path):
                        try:
                            os.remove(local_path)
                        except Exception:
                            pass
                    drive_fid = it.get("drive_file_id")
                    if drive_fid:
                        threading.Thread(
                            target=lambda fid=drive_fid: subprocess.run(
                                ["gdrive", "mutate", "trash", fid],
                                capture_output=True,
                                timeout=15,
                            ),
                            daemon=True,
                        ).start()
                    self.items.pop(i)
                    self._save_items()
                    return True
        return False

    def get_status(self) -> dict:
        g_ok = self.is_gdrive_connected()
        if g_ok and not self.drive_folder_ids.get("root"):
            self._ensure_gdrive_folders()
        return {
            "connected": g_ok,
            "account": "lunariverap97@gmail.com" if g_ok else None,
            "folder_name": self.config_getter().get("gdrive_folder_name", "MichiCam_Cámaras"),
            "folder_url": self.drive_folder_ids.get("root_link", "https://drive.google.com/drive/my-drive"),
            "saved_count": len(self.items),
        }
