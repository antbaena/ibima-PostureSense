#!/usr/bin/env python3
import os
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk
import numpy as np
import cv2

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

try:
    from cv_bridge import CvBridge
    _HAS_CV_BRIDGE = True
    _BRIDGE = CvBridge()
except Exception:
    _HAS_CV_BRIDGE = False
    _BRIDGE = None

IMAGE_TYPES = {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}


def _normalize_bag_uri(path: str) -> str:
    path = os.path.abspath(path)

    # Solo carpeta (no archivo)
    if os.path.isfile(path):
        raise IsADirectoryError("Selecciona una carpeta del rosbag2 (no un archivo).")

    if not os.path.isdir(path):
        raise FileNotFoundError("Ruta no válida (debe ser una carpeta).")

    return path


from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
try:
    from rosbag2_py import SequentialCompressionReader
    _HAS_COMP_READER = True
except Exception:
    SequentialCompressionReader = None
    _HAS_COMP_READER = False


def _metadata_says_file_compression(uri: str) -> bool:
    meta = os.path.join(uri, "metadata.yaml")
    if not os.path.exists(meta):
        return False
    txt = open(meta, "r", encoding="utf-8", errors="ignore").read().lower()
    # cubre variantes tipo: compression_mode: file / "file" / FILE
    return ("compression_mode" in txt) and ("file" in txt)


def _open_reader(uri: str):
    # Si el bag está comprimido a nivel de fichero, usa el reader con descompresión
    if _HAS_COMP_READER and _metadata_says_file_compression(uri):
        reader = SequentialCompressionReader()
    else:
        reader = SequentialReader()

    # Mejor dejar que lo coja de metadata.yaml (más robusto)
    storage_options = StorageOptions(uri=uri, storage_id="")
    converter_options = ConverterOptions(input_serialization_format="", output_serialization_format="")
    reader.open(storage_options, converter_options)
    return reader


def _list_image_topics(uri: str):
    reader = _open_reader(uri)
    topics = reader.get_all_topics_and_types()
    image_topics = [(t.name, t.type) for t in topics if t.type in IMAGE_TYPES]

    preferred = [it for it in image_topics
                if it[0].startswith("/cam") and ("color" in it[0].lower())]
    preferred.sort(key=lambda x: x[0])

    others = [it for it in image_topics if it not in preferred]
    return preferred + others



def _read_first_msg_of_topic(uri: str, topic_name: str):
    reader = _open_reader(uri)
    reader.set_filter(StorageFilter(topics=[topic_name]))
    while reader.has_next():
        topic, data, _t = reader.read_next()
        if topic == topic_name:
            return data
    return None


def _msg_to_pil(msg, msg_type: str) -> Image.Image:
    if msg_type == "sensor_msgs/msg/CompressedImage":
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("No se pudo decodificar CompressedImage.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    if msg_type == "sensor_msgs/msg/Image":
        if _HAS_CV_BRIDGE:
            cv_img = _BRIDGE.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if len(cv_img.shape) == 2:
                return Image.fromarray(cv_img)
            if msg.encoding.lower().startswith("rgb"):
                return Image.fromarray(cv_img)
            if cv_img.shape[2] == 4:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
            else:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(cv_img)

        # Fallback mínimo sin cv_bridge (solo encodings comunes)
        enc = msg.encoding.lower()
        w, h = msg.width, msg.height
        data = np.frombuffer(msg.data, dtype=np.uint8)

        if enc == "mono8":
            return Image.fromarray(data.reshape((h, w)))
        if enc in ("rgb8", "bgr8"):
            img = data.reshape((h, w, 3))
            if enc == "bgr8":
                img = img[:, :, ::-1]
            return Image.fromarray(img)
        if enc in ("rgba8", "bgra8"):
            img = data.reshape((h, w, 4))
            if enc == "bgra8":
                img = img[:, :, [2, 1, 0, 3]]
            return Image.fromarray(img)

        raise ValueError(f"Encoding no soportado sin cv_bridge: {msg.encoding}")

    raise ValueError(f"Tipo no soportado: {msg_type}")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Rosbag2 - Preview cámaras (solo carpeta)")
        self.geometry("1000x700")

        self.path_var = tk.StringVar(value="")
        self._photo_refs = [None] * 4  # evitar GC

        top = tk.Frame(self, padx=10, pady=10)
        top.pack(fill="x")
        tk.Label(top, text="Carpeta rosbag2 (contiene metadata.yaml y *.mcap/*.mcap.zstd):").pack(anchor="w")

        row = tk.Frame(top)
        row.pack(fill="x", pady=(6, 0))

        tk.Entry(row, textvariable=self.path_var).pack(side="left", fill="x", expand=True)
        tk.Button(row, text="Carpeta…", command=self.pick_dir).pack(side="left", padx=6)
        tk.Button(row, text="Cargar", command=self.load_and_show).pack(side="left", padx=6)

        self.grid_frame = tk.Frame(self, padx=10, pady=10)
        self.grid_frame.pack(fill="both", expand=True)

        self.cells = []
        for i in range(4):
            cell = tk.Frame(self.grid_frame, bd=1, relief="groove", padx=8, pady=8)
            r, c = divmod(i, 2)
            cell.grid(row=r, column=c, sticky="nsew", padx=8, pady=8)

            img_label = tk.Label(cell, text="(sin imagen)", width=60, height=18)
            img_label.pack()

            topic_label = tk.Label(cell, text="(topic)", font=("TkDefaultFont", 10, "bold"))
            topic_label.pack(pady=(8, 0))

            self.cells.append((img_label, topic_label))

        for c in (0, 1):
            self.grid_frame.columnconfigure(c, weight=1)
        for r in (0, 1):
            self.grid_frame.rowconfigure(r, weight=1)

    def pick_dir(self):
        d = filedialog.askdirectory(title="Selecciona carpeta del rosbag2")
        if d:
            self.path_var.set(d)

    def load_and_show(self):
        user_path = self.path_var.get().strip()
        if not user_path:
            messagebox.showerror("Error", "Selecciona una carpeta.")
            return

        try:
            uri = _normalize_bag_uri(user_path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        # Limpia celdas
        for i in range(4):
            img_label, topic_label = self.cells[i]
            img_label.configure(image="", text="(sin imagen)")
            topic_label.configure(text="(topic)")
            self._photo_refs[i] = None

        try:
            image_topics = _list_image_topics(uri)
        except Exception as e:
            messagebox.showerror(
                "Error abriendo bag",
                f"{e}\n\nTip: si hay .mcap.zstd, asegúrate de tener instalado rosbag2_compression_zstd."
            )
            return

        if not image_topics:
            messagebox.showinfo("Sin topics", "No se encontraron topics de imagen (Image/CompressedImage).")
            return

        for i, (topic_name, topic_type) in enumerate(image_topics[:4]):
            print(f"Cargando topic: {topic_name} ({topic_type})")
            raw = _read_first_msg_of_topic(uri, topic_name)
            if raw is None:
                continue
            try:
                msg_cls = get_message(topic_type)
                msg = deserialize_message(raw, msg_cls)
                pil = _msg_to_pil(msg, topic_type)

                max_w, max_h = 600, 450  # prueba 600,450 si lo quieres aún más grande

                w, h = pil.size
                scale = min(max_w / w, max_h / h)      # <= esto permite >1 (ampliar)
                new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

                # Pillow moderno:
                try:
                    pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
                except AttributeError:
                    pil = pil.resize((new_w, new_h), Image.LANCZOS)

                photo = ImageTk.PhotoImage(pil)

                img_label, topic_label = self.cells[i]
                img_label.configure(image=photo, text="")
                topic_label.configure(text=topic_name)
                self._photo_refs[i] = photo
            except Exception as e:
                img_label, topic_label = self.cells[i]
                img_label.configure(text=f"Error mostrando imagen:\n{e}", image="")
                topic_label.configure(text=topic_name)


if __name__ == "__main__":
    App().mainloop()
