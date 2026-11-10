"""
Frame Cropper - offline desktop app
-----------------------------------
Reads every screen-recording video in a folder (any length, e.g. 100 min),
extracts screenshots at a chosen interval, optionally crops each one to a
region you draw, and skips duplicate images.

Install once (needs internet only for this step):
    pip install opencv-python numpy
Run:
    python frame_cropper.py
Make a standalone .exe (optional):
    pip install pyinstaller
    pyinstaller --onefile --windowed frame_cropper.py
"""
import base64
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
try:
    import cv2
    import numpy as np
except ImportError:
    messagebox.showerror(
        "Missing packages",
        "Please install the required packages first.\n\n"
        "Open Command Prompt and run:\n    pip install opencv-python numpy",
    )
    raise SystemExit(1)

VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".flv", ".m4v", ".ts", ".mpg", ".mpeg", ".mts", ".m2ts", ".3gp", ".ogv", ".vob", ".f4v"}
THUMB = (256, 144)        # size used to compare images (width, height)
PIXEL_DIFF = 14          # a pixel counts as "changed" if gray level differs by more than this
PREVIEW_MAX = (640, 360)


def list_videos(folder, recursive):
    found = []
    if recursive:
        for root, _, files in os.walk(folder):
            for f in files:
                if os.path.splitext(f)[1].lower() in VIDEO_EXT:
                    found.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            p = os.path.join(folder, f)
            if os.path.isfile(p) and os.path.splitext(f)[1].lower() in VIDEO_EXT:
                found.append(p)
    return sorted(found)


def make_thumb(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, THUMB, interpolation=cv2.INTER_AREA).astype(np.int16).ravel()


def _short_path(path):
    """Windows 8.3 path: helps OpenCV open files whose names have special characters."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(32768)
        n = ctypes.windll.kernel32.GetShortPathNameW(path, buf, 32768)
        return buf.value if n else None
    except Exception:
        return None


class CvSource:
    """OpenCV reader (fast: skips decoding of frames it does not need)."""

    def __init__(self, path, backend):
        self.cap = None
        paths = [path]
        sp = _short_path(path)
        if sp and sp != path:
            paths.append(sp)
        for p in paths:
            cap = cv2.VideoCapture(p, backend)
            if cap.isOpened():
                self.cap = cap
                break
            cap.release()
        if self.cap is None:
            raise IOError("could not open")
        ok, first = self.cap.read()
        if not ok or first is None:
            self.cap.release()
            raise IOError("opened but no frame could be decoded")
        self._first = first
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps == fps and 1 <= fps <= 240 else 30.0
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.total = total if total > 0 else 0

    def first_frame(self):
        return self._first

    def iter_samples(self, step):
        yield 0, self._first
        idx = 1
        while self.cap.grab():
            if idx % step == 0:
                ok, f = self.cap.retrieve()
                if ok:
                    yield idx, f
            idx += 1

    def close(self):
        self.cap.release()


class AvSource:
    """PyAV reader (bundles its own FFmpeg: handles HEVC/H.265, AV1, etc.)."""

    def __init__(self, path):
        import av
        self.c = av.open(path)
        st = self.c.streams.video[0]
        st.thread_type = "AUTO"
        rate = st.average_rate or st.guessed_rate
        fps = float(rate) if rate else 30.0
        self.fps = fps if 1 <= fps <= 240 else 30.0
        dur = (float(self.c.duration) / 1_000_000) if self.c.duration else 0
        self.total = int(st.frames or 0) or int(dur * self.fps)
        self._it = self.c.decode(st)
        self._first = next(self._it).to_ndarray(format="bgr24")

    def first_frame(self):
        return self._first

    def iter_samples(self, step):
        yield 0, self._first
        for idx, fr in enumerate(self._it, start=1):
            if idx % step == 0:
                yield idx, fr.to_ndarray(format="bgr24")

    def close(self):
        try:
            self.c.close()
        except Exception:
            pass


def open_source(path):
    """Try several readers until one can really decode the video."""
    errs = []
    backends = [("OpenCV-FFmpeg", cv2.CAP_FFMPEG)]
    if os.name == "nt":
        backends.append(("Windows Media Foundation", cv2.CAP_MSMF))
    for name, b in backends:
        try:
            return CvSource(path, b), name
        except Exception as e:
            errs.append(f"{name}: {e}")
    try:
        return AvSource(path), "PyAV"
    except ImportError:
        errs.append("PyAV: not installed")
    except Exception as e:
        errs.append(f"PyAV: {e}")
    raise IOError("; ".join(errs))


class History:
    """Remembers thumbnails of every saved image to detect duplicates."""

    def __init__(self):
        self.arr = np.zeros((256, THUMB[0] * THUMB[1]), np.int16)
        self.n = 0
        self.last = None

    def is_duplicate(self, t, threshold, check_all):
        if self.last is not None:
            if (np.abs(self.last - t) > PIXEL_DIFF).mean() <= threshold:
                return True
        if check_all and self.n:
            d = (np.abs(self.arr[: self.n] - t) > PIXEL_DIFF).mean(axis=1)
            if d.min() <= threshold:
                return True
        return False

    def add(self, t):
        if self.n == len(self.arr):
            self.arr = np.vstack([self.arr, np.zeros_like(self.arr)])
        self.arr[self.n] = t
        self.n += 1
        self.last = t


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Frame Cropper (offline)")
        self.geometry("980x640")
        self.q = queue.Queue()
        self.stop_flag = threading.Event()
        self.worker = None

        self.in_var = tk.StringVar()
        self.out_var = tk.StringVar()
        self.interval = tk.DoubleVar(value=1.0)
        self.sens = tk.DoubleVar(value=0.2)
        self.fmt = tk.StringVar(value="png")
        self.recursive = tk.BooleanVar(value=False)
        self.check_all = tk.BooleanVar(value=True)
        self.across = tk.BooleanVar(value=True)
        self.subfolders = tk.BooleanVar(value=True)

        self.crop = None            # (x0, y0, x1, y1) as fractions 0..1
        self.preview_img = None
        self.prev_scale = (1, 1)
        self.rect_id = None
        self.drag_start = None

        self.build_ui()
        self.after(100, self.poll)

    # ---------------- UI ----------------
    def build_ui(self):
        left = ttk.Frame(self, padding=10)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=10)
        right.pack(side="left", fill="both", expand=True)

        def folder_row(label, var, cmd):
            ttk.Label(left, text=label).pack(anchor="w")
            row = ttk.Frame(left)
            row.pack(fill="x", pady=(0, 8))
            ttk.Entry(row, textvariable=var, width=34).pack(side="left", fill="x", expand=True)
            ttk.Button(row, text="Browse", command=cmd).pack(side="left", padx=(4, 0))

        folder_row("Video folder", self.in_var, self.pick_in)
        folder_row("Output folder", self.out_var, self.pick_out)

        hdr = ttk.Frame(left); hdr.pack(fill="x")
        ttk.Label(hdr, text="Videos found (all are processed)").pack(side="left")
        ttk.Button(hdr, text="Rescan", command=self.rescan).pack(side="right")
        lf = ttk.Frame(left); lf.pack(fill="x", pady=(0, 8))
        self.vid_list = tk.Listbox(lf, height=6, width=44, exportselection=False)
        sb = ttk.Scrollbar(lf, orient="vertical", command=self.vid_list.yview)
        self.vid_list.config(yscrollcommand=sb.set)
        self.vid_list.pack(side="left", fill="x", expand=True)
        sb.pack(side="left", fill="y")
        self.vid_list.bind("<<ListboxSelect>>", self.on_pick_video)
        self.videos = []

        r = ttk.Frame(left); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Take a picture every (seconds)").pack(side="left")
        ttk.Spinbox(r, from_=0.1, to=600, increment=0.5, width=6, textvariable=self.interval).pack(side="right")

        r = ttk.Frame(left); r.pack(fill="x", pady=2)
        ttk.Label(r, text="Duplicate sensitivity (% pixels)").pack(side="left")
        ttk.Spinbox(r, from_=0.01, to=20, increment=0.05, width=6, textvariable=self.sens).pack(side="right")
        ttk.Label(left, text="Lower = stricter (keeps more near-identical images)\n"
                             "Higher = removes more look-alikes", foreground="#666").pack(anchor="w")

        r = ttk.Frame(left); r.pack(fill="x", pady=6)
        ttk.Label(r, text="Image format").pack(side="left")
        ttk.Combobox(r, values=["png", "jpg"], textvariable=self.fmt, width=6, state="readonly").pack(side="right")

        for text, var in [
            ("Include sub-folders", self.recursive),
            ("Skip images that match ANY earlier image", self.check_all),
            ("Skip duplicates across different videos", self.across),
            ("Save each video in its own sub-folder", self.subfolders),
        ]:
            ttk.Checkbutton(left, text=text, variable=var,
                            command=self.rescan if var is self.recursive else None).pack(anchor="w")

        self.start_btn = ttk.Button(left, text="Start", command=self.start)
        self.start_btn.pack(fill="x", pady=(14, 2))
        self.stop_btn = ttk.Button(left, text="Stop", command=self.stop_flag.set, state="disabled")
        self.stop_btn.pack(fill="x")

        self.bar = ttk.Progressbar(left, maximum=1000)
        self.bar.pack(fill="x", pady=(10, 2))
        self.status = ttk.Label(left, text="Idle")
        self.status.pack(anchor="w")

        ttk.Label(right, text="Crop area: drag a box on the preview (no box = full frame)").pack(anchor="w")
        self.canvas = tk.Canvas(right, width=PREVIEW_MAX[0], height=PREVIEW_MAX[1], bg="#222", cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        row = ttk.Frame(right); row.pack(fill="x", pady=4)
        ttk.Button(row, text="Reset crop (full frame)", command=self.reset_crop).pack(side="left")
        self.crop_lbl = ttk.Label(row, text="Crop: full frame")
        self.crop_lbl.pack(side="left", padx=10)

        self.log = tk.Text(right, height=10, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True, pady=(6, 0))

    def write_log(self, s):
        self.log.configure(state="normal")
        self.log.insert("end", s + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # ---------------- folder + preview ----------------
    def pick_in(self):
        d = filedialog.askdirectory(title="Select folder with videos")
        if not d:
            return
        self.in_var.set(d)
        if not self.out_var.get():
            self.out_var.set(os.path.join(d, "extracted_images"))
        self.rescan()

    def rescan(self):
        d = self.in_var.get().strip()
        self.vid_list.delete(0, "end")
        self.videos = []
        if not os.path.isdir(d):
            self.write_log("Video folder not found. Check the path.")
            return
        self.videos = list_videos(d, self.recursive.get())
        for v in self.videos:
            self.vid_list.insert("end", os.path.relpath(v, d))
        self.write_log(f"Found {len(self.videos)} video(s).")
        if self.videos:
            self.vid_list.selection_set(0)
            self.load_preview(self.videos[0])
        else:
            self.write_log("No videos found. If they are inside other folders, tick 'Include sub-folders'. "
                           "Supported: " + ", ".join(sorted(VIDEO_EXT)))

    def on_pick_video(self, _e=None):
        sel = self.vid_list.curselection()
        if sel:
            self.load_preview(self.videos[sel[0]])

    def pick_out(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.out_var.set(d)

    def load_preview(self, path):
        try:
            src, backend = open_source(path)
        except Exception as e:
            self.canvas.delete("all")
            self.preview_img = None
            self.write_log(f"Cannot read {os.path.basename(path)}: {e}\n"
                           "(If the file is still being rendered/saved, wait until it finishes.)")
            return
        frame = src.first_frame()
        src.close()
        self.write_log(f"Preview OK ({backend}).")
        h, w = frame.shape[:2]
        s = min(PREVIEW_MAX[0] / w, PREVIEW_MAX[1] / h)
        dw, dh = int(w * s), int(h * s)
        small = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".png", small)
        self.preview_img = tk.PhotoImage(data=base64.b64encode(buf.tobytes()))
        self.canvas.delete("all")
        self.canvas.config(width=dw, height=dh)
        self.canvas.create_image(0, 0, anchor="nw", image=self.preview_img)
        self.prev_size = (dw, dh)
        self.reset_crop()

    def on_press(self, e):
        if self.preview_img is None:
            return
        self.drag_start = (e.x, e.y)
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="#00ff66", width=2)

    def on_drag(self, e):
        if self.drag_start:
            self.canvas.coords(self.rect_id, *self.drag_start, e.x, e.y)

    def on_release(self, e):
        if not self.drag_start:
            return
        dw, dh = self.prev_size
        x0, y0 = self.drag_start
        x1, y1 = e.x, e.y
        self.drag_start = None
        x0, x1 = sorted((max(0, min(dw, x0)), max(0, min(dw, x1))))
        y0, y1 = sorted((max(0, min(dh, y0)), max(0, min(dh, y1))))
        if x1 - x0 < 10 or y1 - y0 < 10:
            self.reset_crop()
            return
        self.crop = (x0 / dw, y0 / dh, x1 / dw, y1 / dh)
        self.crop_lbl.config(text=f"Crop: {int((x1 - x0) / dw * 100)}% x {int((y1 - y0) / dh * 100)}% of frame")

    def reset_crop(self):
        self.crop = None
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        self.crop_lbl.config(text="Crop: full frame")

    # ---------------- processing ----------------
    def start(self):
        in_dir, out_dir = self.in_var.get(), self.out_var.get()
        if not os.path.isdir(in_dir):
            messagebox.showerror("Error", "Choose a valid video folder.")
            return
        if not out_dir:
            messagebox.showerror("Error", "Choose an output folder.")
            return
        try:
            interval = float(self.interval.get())
            thr = float(self.sens.get()) / 100.0
        except tk.TclError:
            messagebox.showerror("Error", "Interval and sensitivity must be numbers.")
            return
        vids = list_videos(in_dir, self.recursive.get())
        if not vids:
            messagebox.showinfo("Nothing to do", "No video files found.")
            return
        if self.preview_img is None:
            self.load_preview(vids[0])
        os.makedirs(out_dir, exist_ok=True)
        self.stop_flag.clear()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        opts = dict(interval=max(0.05, interval), thr=thr, fmt=self.fmt.get(), check_all=self.check_all.get(),
                    across=self.across.get(), sub=self.subfolders.get(), crop=self.crop, out=out_dir)
        self.worker = threading.Thread(target=self.run, args=(vids, opts), daemon=True)
        self.worker.start()

    def run(self, vids, o):
        import traceback
        try:
            self._run(vids, o)
        except Exception:
            self.q.put(("log", "ERROR:\n" + traceback.format_exc()))
            self.q.put(("done", 0, True))

    def _run(self, vids, o):
        q = self.q
        shared = History()
        total_saved = 0
        for vi, path in enumerate(vids, 1):
            if self.stop_flag.is_set():
                break
            base = os.path.basename(path)
            name = os.path.splitext(base)[0]
            out = os.path.join(o["out"], name) if o["sub"] else o["out"]
            os.makedirs(out, exist_ok=True)
            hist = shared if o["across"] else History()
            try:
                src, backend = open_source(path)
            except Exception as e:
                q.put(("log", f"[{vi}/{len(vids)}] SKIPPED {base}: {e}"))
                continue
            fps, total = src.fps, src.total
            step = max(1, int(round(fps * o["interval"])))
            q.put(("log", f"[{vi}/{len(vids)}] {base}  ({total / fps / 60:.1f} min, reader: {backend})"))
            saved = skipped = 0
            try:
                for idx, frame in src.iter_samples(step):
                    if self.stop_flag.is_set():
                        break
                    if o["crop"]:
                        h, w = frame.shape[:2]
                        x0, y0, x1, y1 = o["crop"]
                        frame = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
                    if frame.size:
                        t = make_thumb(frame)
                        if hist.is_duplicate(t, o["thr"], o["check_all"]):
                            skipped += 1
                        else:
                            sec = idx / fps
                            hh, rem = divmod(sec, 3600)
                            mm, ss = divmod(rem, 60)
                            fname = f"{name}_{int(hh):02d}-{int(mm):02d}-{int(ss):02d}-{int((ss % 1) * 1000):03d}.{o['fmt']}"
                            ext = ".jpg" if o["fmt"] == "jpg" else ".png"
                            params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext == ".jpg" else []
                            ok2, buf = cv2.imencode(ext, frame, params)
                            if ok2:
                                with open(os.path.join(out, fname), "wb") as f:
                                    f.write(buf.tobytes())
                                hist.add(t)
                                saved += 1
                                total_saved += 1
                    if total:
                        q.put(("progress", min(1.0, (vi - 1 + idx / total) / len(vids)),
                               f"Video {vi}/{len(vids)} | saved {total_saved} | skipped dupes {skipped}"))
            except Exception as e:
                q.put(("log", f"    stopped early on this video: {e}"))
            finally:
                src.close()
            q.put(("log", f"    saved {saved}, skipped {skipped} duplicates"))
        q.put(("done", total_saved, self.stop_flag.is_set()))

    def poll(self):
        try:
            while True:
                m = self.q.get_nowait()
                if m[0] == "log":
                    self.write_log(m[1])
                elif m[0] == "progress":
                    self.bar["value"] = m[1] * 1000
                    self.status.config(text=m[2])
                elif m[0] == "done":
                    self.bar["value"] = 1000 if not m[2] else self.bar["value"]
                    self.status.config(text=("Stopped. " if m[2] else "Finished. ") + f"{m[1]} images saved.")
                    self.write_log(self.status.cget("text"))
                    self.start_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
        except queue.Empty:
            pass
        self.after(100, self.poll)


if __name__ == "__main__":
    App().mainloop()
